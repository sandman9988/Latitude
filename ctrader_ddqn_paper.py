#!/usr/bin/env python3
# ctrader_ddqn_paper.py
# Dual FIX sessions (QUOTE+TRADE) for cTrader/Pepperstone demo.
# Builds BTCUSD (symbolId=10028) M15 bars from best bid/ask, then trades 0.10 qty target-position via TRADE.
#
# Requires QuickFIX built/installed into the venv (you already did this).
#
# Run:
#   source ~/Documents/.venv/bin/activate
#   export CTRADER_USERNAME="5179095"
#   export CTRADER_PASSWORD_QUOTE="***"
#   export CTRADER_PASSWORD_TRADE="***"
#   export CTRADER_CFG_QUOTE="ctrader_quote.cfg"
#   export CTRADER_CFG_TRADE="ctrader_trade.cfg"
#   export CTRADER_BTC_SYMBOL_ID="10028"
#   export CTRADER_QTY="0.10"
#   python3 ctrader_ddqn_paper.py

import datetime as dt
import logging
import os
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import quickfix44 as fix44

import quickfix as fix

from performance_tracker import PerformanceTracker


# ----------------------------
# Logging
# ----------------------------
def _redact_fix(s: str) -> str:
    # redact tag 554=Password (and anything else you want)
    return s.replace("554=", "554=REDACTED")


def setup_logging() -> logging.Logger:
    logdir = os.environ.get("PY_LOGDIR", "ctrader_py_logs").strip()
    Path(logdir).mkdir(parents=True, exist_ok=True)

    # Python 3.12: utcnow() deprecates; use timezone-aware UTC.
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logdir, f"ctrader_{ts}.log")

    logger = logging.getLogger("ctrader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Executable: %s", sys.executable)
    logger.info("CWD: %s", os.getcwd())
    logger.info("PY_LOGDIR: %s", os.path.abspath(logdir))
    logger.info("Logfile: %s", logfile)
    return logger


LOG = setup_logging()


# ----------------------------
# Time helpers (UTC required)
# ----------------------------
def utc_ts_ms() -> str:
    # FIX UTCTimestamp: YYYYMMDD-HH:MM:SS.sss (UTC)
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# ----------------------------
# Bar builder (configurable timeframe)
# ----------------------------
class BarBuilder:
    def __init__(self, timeframe_minutes: int = 15):
        self.timeframe_minutes = timeframe_minutes
        self.bucket = None
        self.o = self.h = self.l = self.c = None

    def bucket_start(self, t: dt.datetime) -> dt.datetime:
        m = (t.minute // self.timeframe_minutes) * self.timeframe_minutes
        return t.replace(minute=m, second=0, microsecond=0)

    def update(self, t: dt.datetime, mid: float):
        b = self.bucket_start(t)
        if self.bucket is None:
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return None

        if b != self.bucket:
            closed = (self.bucket, self.o, self.h, self.l, self.c)
            self.bucket = b
            self.o = self.h = self.l = self.c = mid
            return closed

        self.c = mid
        if mid > self.h:
            self.h = mid
        if mid < self.l:
            self.l = mid
        return None


# ----------------------------
# Minimal policy wrapper
# ----------------------------
class Policy:
    """
    Discrete actions: 0=SHORT, 1=FLAT, 2=LONG
    Default: FLAT until you load a DDQN model (optional).
    """

    def __init__(self):
        self.use_torch = False
        self.model = None
        self.window = 64

        model_path = os.environ.get("DDQN_MODEL_PATH", "").strip()
        if model_path:
            try:
                import torch
                import torch.nn as nn

                class QNet(nn.Module):
                    def __init__(
                        self, window: int, n_features: int, n_actions: int = 3
                    ):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.Conv1d(64, 64, kernel_size=5, padding=2),
                            nn.ReLU(),
                            nn.AdaptiveAvgPool1d(1),
                            nn.Flatten(),
                            nn.Linear(64, 128),
                            nn.ReLU(),
                            nn.Linear(128, n_actions),
                        )

                    def forward(self, x):
                        # x: (B,T,F) -> (B,F,T)
                        return self.net(x.transpose(1, 2))

                self.torch = torch
                self.model = QNet(window=self.window, n_features=4, n_actions=3)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                self.use_torch = True
                LOG.info("[POLICY] Loaded DDQN model: %s", model_path)
            except Exception as e:
                LOG.warning(
                    "[POLICY] Failed to load model, running fallback. Error: %s", e
                )
                self.use_torch = False

    def decide(self, bars: deque) -> int:
        if len(bars) < 70:
            return 1  # FLAT

        closes = [b[4] for b in bars]
        import numpy as np

        c = np.array(closes, dtype=np.float64)

        ret1 = np.zeros_like(c)
        ret1[1:] = (c[1:] / c[:-1]) - 1.0

        ret5 = np.zeros_like(c)
        ret5[5:] = (c[5:] / c[:-5]) - 1.0

        def rolling_mean(x, n):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) >= n:
                cs = np.cumsum(np.insert(x, 0, 0.0))
                out[n - 1 :] = (cs[n:] - cs[:-n]) / n
            return out

        def rolling_std(x, n):
            out = np.full_like(x, np.nan, dtype=np.float64)
            if len(x) >= n:
                for i in range(n - 1, len(x)):
                    w = x[i - n + 1 : i + 1]
                    out[i] = np.std(w)
            return out

        ma_fast = rolling_mean(c, 10)
        ma_slow = rolling_mean(c, 30)
        ma_diff = (ma_fast / ma_slow) - 1.0
        vol = rolling_std(ret1, 20)

        feats = np.vstack([ret1, ret5, np.nan_to_num(ma_diff), np.nan_to_num(vol)]).T
        feats = feats[-self.window :].astype(np.float32)

        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-8
        x = (feats - mu) / sd

        if not self.use_torch:
            md = float(x[-1, 2])
            if md > 0.2:
                return 2
            if md < -0.2:
                return 0
            return 1

        with self.torch.no_grad():
            t = self.torch.from_numpy(x).unsqueeze(0)
            q = self.model(t).squeeze(0).numpy()
            return int(q.argmax())


# ----------------------------
# Path Recorder
# ----------------------------
class PathRecorder:
    """Record M1 OHLC path during entire trade lifecycle."""
    def __init__(self):
        self.recording = False
        self.entry_time = None
        self.entry_price = None
        self.direction = None
        self.path = []  # List of (timestamp, o, h, l, c) tuples
        self.trade_counter = 0

    def start_recording(self, entry_time: dt.datetime, entry_price: float, direction: int):
        """Start recording path for a new trade."""
        self.recording = True
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.path = []
        LOG.info("[PATH] Started recording for %s trade at %.2f",
                 "LONG" if direction == 1 else "SHORT", entry_price)

    def add_bar(self, bar):
        """Add a bar to the path. bar is tuple: (timestamp, o, h, l, c)"""
        if not self.recording:
            return
        self.path.append(bar)

    def stop_recording(self, exit_time: dt.datetime, exit_price: float, pnl: float) -> dict:
        """Stop recording and return trade summary with path."""
        if not self.recording:
            return None

        self.recording = False
        self.trade_counter += 1

        # Calculate trade duration
        duration_seconds = (exit_time - self.entry_time).total_seconds() if self.entry_time else 0

        trade_record = {
            "trade_id": self.trade_counter,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "pnl": pnl,
            "duration_seconds": duration_seconds,
            "bars_count": len(self.path),
            "path": [
                {"timestamp": t.isoformat(), "open": o, "high": h, "low": l, "close": c}
                for t, o, h, l, c in self.path
            ]
        }

        # Save to JSON file
        self._save_to_file(trade_record)

        LOG.info("[PATH] Stopped recording. Trade #%d: %d bars, %.2f seconds, PnL=%.2f",
                 self.trade_counter, len(self.path), duration_seconds, pnl)

        return trade_record

    def _save_to_file(self, trade_record: dict):
        """Save trade record to JSON file."""
        import json
        from pathlib import Path

        trades_dir = Path("trades")
        trades_dir.mkdir(exist_ok=True)

        filename = trades_dir / f"trade_{trade_record['trade_id']:04d}_{trade_record['direction'].lower()}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(trade_record, f, indent=2)
            LOG.info("[PATH] Saved to %s", filename)
        except Exception as e:
            LOG.error("[PATH] Failed to save: %s", e)


# ----------------------------
# MFE/MAE Tracker
# ----------------------------
class MFEMAETracker:
    """Track Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) per position."""
    def __init__(self):
        self.entry_price = None
        self.direction = None  # 1=long, -1=short
        self.mfe = 0.0  # max favorable move in $ (profit)
        self.mae = 0.0  # max adverse move in $ (loss, stored as positive)
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def start_tracking(self, entry_price: float, direction: int):
        """direction: 1=long, -1=short"""
        self.entry_price = entry_price
        self.direction = direction
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False

    def update(self, current_price: float):
        """Update with current market price during open position."""
        if self.entry_price is None:
            return

        # Calculate P&L
        if self.direction == 1:  # long
            pnl = current_price - self.entry_price
        else:  # short
            pnl = self.entry_price - current_price

        # Track best profit (MFE)
        if pnl > self.best_profit:
            self.best_profit = pnl
            self.mfe = pnl

        # Track worst loss (MAE)
        if pnl < self.worst_loss:
            self.worst_loss = pnl
            self.mae = abs(pnl)

        # Detect winner-to-loser (was profitable, now losing)
        if self.best_profit > 0 and pnl < 0:
            self.winner_to_loser = True

    def get_summary(self) -> dict:
        """Return summary of MFE/MAE metrics."""
        return {
            "entry_price": self.entry_price,
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "mfe": self.mfe,
            "mae": self.mae,
            "best_profit": self.best_profit,
            "worst_loss": self.worst_loss,
            "winner_to_loser": self.winner_to_loser
        }

    def reset(self):
        """Reset tracker for next position."""
        self.entry_price = None
        self.direction = None
        self.mfe = 0.0
        self.mae = 0.0
        self.best_profit = 0.0
        self.worst_loss = 0.0
        self.winner_to_loser = False


# ----------------------------
# FIX application
# ----------------------------
class CTraderFixApp(fix.Application):
    def __init__(self, symbol_id: int, qty: float, timeframe_minutes: int = 15):
        super().__init__()
        self.symbol_id = symbol_id
        self.qty = qty
        self.timeframe_minutes = timeframe_minutes

        self.quote_sid = None
        self.trade_sid = None

        self.policy = Policy()
        self.bars = deque(maxlen=2000)
        self.builder = BarBuilder(timeframe_minutes)
        self.mfe_mae_tracker = MFEMAETracker()
        self.path_recorder = PathRecorder()
        self.performance = PerformanceTracker()

        self.best_bid = None
        self.best_ask = None

        self.cur_pos = 0
        self.pos_req_id = None
        self.clord_counter = 0
        self.trade_entry_time = None  # Track entry time for performance metrics

    # ---- helpers ----
    @staticmethod
    def _qual(sessionID) -> str:
        # Route strictly by SessionQualifier (NOT SubIDs).
        try:
            q = sessionID.getSessionQualifier()
            return (q or "").upper()
        except Exception:
            return ""

    # ---- session events ----
    def onCreate(self, sessionID):
        LOG.info("[CREATE] %s qual=%s", sessionID.toString(), self._qual(sessionID))

    def onLogon(self, sessionID):
        qual = self._qual(sessionID)
        LOG.info("[LOGON] %s qual=%s", sessionID.toString(), qual)

        if qual == "QUOTE":
            self.quote_sid = sessionID
            self.send_md_subscribe_spot()
        elif qual == "TRADE":
            self.trade_sid = sessionID
            self.request_positions()
        else:
            LOG.warning(
                "[LOGON] Unknown qualifier; not routing: %s", sessionID.toString()
            )

    def onLogout(self, sessionID):
        LOG.warning("[LOGOUT] %s qual=%s", sessionID.toString(), self._qual(sessionID))

    # ---- admin hooks ----
    def toAdmin(self, message, sessionID):
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)

        if msg_type.getValue() != fix.MsgType_Logon:
            return

        qual = self._qual(sessionID)

        # Reset seq nums
        message.setField(fix.ResetSeqNumFlag(True))

        user = os.environ.get("CTRADER_USERNAME", "").strip()
        if qual == "QUOTE":
            pwd = os.environ.get("CTRADER_PASSWORD_QUOTE", "").strip()
            message.getHeader().setField(fix.TargetSubID("QUOTE"))
        else:
            pwd = os.environ.get("CTRADER_PASSWORD_TRADE", "").strip()
            message.getHeader().setField(fix.TargetSubID("TRADE"))

        if user:
            message.setField(fix.Username(user))
        if pwd:
            message.setField(fix.Password(pwd))

        LOG.info("[ADMIN][LOGON OUT] qual=%s %s", qual, _redact_fix(message.toString()))

    def fromAdmin(self, message, sessionID):
        qual = self._qual(sessionID)
        LOG.info("[ADMIN][IN] qual=%s %s", qual, _redact_fix(message.toString()))

    def toApp(self, message, sessionID):
        qual = self._qual(sessionID)
        # Keep this INFO until stable; you can reduce to DEBUG later.
        LOG.info("[APP][OUT] qual=%s %s", qual, _redact_fix(message.toString()))

    def fromApp(self, message, sessionID):
        qual = self._qual(sessionID)
        LOG.info("[APP][IN] qual=%s %s", qual, _redact_fix(message.toString()))

        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        t = msg_type.getValue()

        if t == "W":
            self.on_md_snapshot(message)
        elif t == "X":
            self.on_md_incremental(message)
        elif t == "8":
            self.on_exec_report(message)
        elif t == "AP":
            self.on_position_report(message)
        elif t == "j":
            self.on_biz_reject(message)
        elif t == "Y":
            self.on_md_reject(message)

    # ----------------------------
    # QUOTE: Market data subscribe
    # ----------------------------
    def send_md_subscribe_spot(self):
        if not self.quote_sid:
            return

        req = fix44.MarketDataRequest()
        req.setField(fix.MDReqID("BTCUSD_SPOT"))
        req.setField(fix.SubscriptionRequestType("1"))
        req.setField(fix.MarketDepth(1))
        req.setField(fix.MDUpdateType(1))
        req.setField(fix.NoMDEntryTypes(2))

        g1 = fix44.MarketDataRequest.NoMDEntryTypes()
        g1.setField(fix.MDEntryType("0"))
        req.addGroup(g1)

        g2 = fix44.MarketDataRequest.NoMDEntryTypes()
        g2.setField(fix.MDEntryType("1"))
        req.addGroup(g2)

        req.setField(fix.NoRelatedSym(1))
        sym = fix44.MarketDataRequest.NoRelatedSym()
        sym.setField(fix.Symbol(str(self.symbol_id)))
        req.addGroup(sym)

        fix.Session.sendToTarget(req, self.quote_sid)
        LOG.info("[QUOTE] Subscribed spot for symbolId=%s", self.symbol_id)

    def on_md_snapshot(self, msg: fix.Message):
        try:
            no = fix.NoMDEntries()
            msg.getField(no)
            n = int(no.getValue())
        except Exception:
            return

        bid = ask = None
        for i in range(1, n + 1):
            g = fix44.MarketDataSnapshotFullRefresh().NoMDEntries()
            msg.getGroup(i, g)

            et = fix.MDEntryType()
            px = fix.MDEntryPx()
            if g.isSetField(et) and g.isSetField(px):
                g.getField(et)
                g.getField(px)
                if et.getValue() == "0":
                    bid = float(px.getValue())
                if et.getValue() == "1":
                    ask = float(px.getValue())

        if bid is not None:
            self.best_bid = bid
        if ask is not None:
            self.best_ask = ask

        self.try_bar_update()

    def on_md_incremental(self, msg: fix.Message):
        try:
            no = fix.NoMDEntries()
            msg.getField(no)
            n = int(no.getValue())
        except Exception:
            return

        for i in range(1, n + 1):
            g = fix44.MarketDataIncrementalRefresh().NoMDEntries()
            msg.getGroup(i, g)

            act = fix.MDUpdateAction()
            g.getField(act)
            action = act.getValue()

            et = fix.MDEntryType()
            sym = fix.Symbol()

            if g.isSetField(et):
                g.getField(et)
            if g.isSetField(sym):
                g.getField(sym)

            if sym.getValue() and sym.getValue() != str(self.symbol_id):
                continue

            if action == "0":
                px = fix.MDEntryPx()
                if g.isSetField(px):
                    g.getField(px)
                    p = float(px.getValue())
                    if et.getValue() == "0":
                        self.best_bid = p
                    if et.getValue() == "1":
                        self.best_ask = p
            elif action == "2":
                if et.getValue() == "0":
                    self.best_bid = None
                if et.getValue() == "1":
                    self.best_ask = None

        self.try_bar_update()

    def try_bar_update(self):
        if self.best_bid is None or self.best_ask is None:
            return

        mid = (self.best_bid + self.best_ask) / 2.0
        
        # Update MFE/MAE if we have an open position
        if self.cur_pos != 0:
            self.mfe_mae_tracker.update(mid)
        
        closed = self.builder.update(utc_now(), mid)
        if closed:
            self.bars.append(closed)
            self.on_bar_close(closed)

    # ----------------------------
    # TRADE: positions + orders
    # ----------------------------
    def request_positions(self):
        if not self.trade_sid:
            return

        self.pos_req_id = f"pos_{uuid.uuid4().hex[:10]}"
        req = fix44.RequestForPositions()
        req.setField(fix.PosReqID(self.pos_req_id))
        fix.Session.sendToTarget(req, self.trade_sid)
        LOG.info("[TRADE] Requested positions")

    def on_position_report(self, msg: fix.Message):
        try:
            sym = fix.Symbol()
            if msg.isSetField(sym):
                msg.getField(sym)
                if sym.getValue() != str(self.symbol_id):
                    return
        except Exception:
            return

        long_qty = 0.0
        short_qty = 0.0

        f704 = fix.StringField(704)  # LongQty
        f705 = fix.StringField(705)  # ShortQty

        if msg.isSetField(f704):
            msg.getField(f704)
            long_qty = float(f704.getValue())
        if msg.isSetField(f705):
            msg.getField(f705)
            short_qty = float(f705.getValue())

        net = long_qty - short_qty
        old_pos = self.cur_pos
        if abs(net) < self.qty * 0.5:
            self.cur_pos = 0
        elif net > 0:
            self.cur_pos = 1
        else:
            self.cur_pos = -1

        LOG.info("[TRADE] PositionReport net=%0.6f -> cur_pos=%s", net, self.cur_pos)
        
        # Log MFE/MAE summary and save path when position closes
        if old_pos != 0 and self.cur_pos == 0:
            summary = self.mfe_mae_tracker.get_summary()
            LOG.info(
                "[MFE/MAE] Entry=%.5f %s | MFE=%.5f MAE=%.5f | Best=%.5f Worst=%.5f | WTL=%s",
                summary["entry_price"], summary["direction"],
                summary["mfe"], summary["mae"],
                summary["best_profit"], summary["worst_loss"],
                summary["winner_to_loser"]
            )
            
            # Stop path recording and save trade
            if self.best_bid and self.best_ask:
                exit_price = (self.best_bid + self.best_ask) / 2.0
                exit_time = utc_now()
                direction_sign = 1 if summary["direction"] == "LONG" else -1
                pnl = (exit_price - summary["entry_price"]) * direction_sign
                
                # Save path
                self.path_recorder.stop_recording(exit_time, exit_price, pnl)
                
                # Add to performance tracker
                if self.trade_entry_time:
                    self.performance.add_trade(
                        pnl=pnl,
                        entry_time=self.trade_entry_time,
                        exit_time=exit_time,
                        direction=summary["direction"],
                        entry_price=summary["entry_price"],
                        exit_price=exit_price,
                        mfe=summary["mfe"],
                        mae=summary["mae"],
                        winner_to_loser=summary["winner_to_loser"]
                    )
                    
                    # Log performance dashboard every 5 trades
                    if self.performance.total_trades % 5 == 0:
                        LOG.info("\n" + self.performance.print_dashboard())
                    else:
                        metrics = self.performance.get_metrics()
                        LOG.info(
                            "[PERF] Trades: %d | Win Rate: %.1f%% | Total PnL: $%.2f | Sharpe: %.3f | Max DD: %.1f%%",
                            metrics['total_trades'],
                            metrics['win_rate'] * 100,
                            metrics['total_pnl'],
                            metrics['sharpe_ratio'],
                            metrics['max_drawdown'] * 100
                        )
                
            self.mfe_mae_tracker.reset()
            self.trade_entry_time = None

    def on_exec_report(self, msg: fix.Message):
        ex = fix.ExecType()
        if not msg.isSetField(ex):
            return
        msg.getField(ex)

        if ex.getValue() == "8":
            txt = fix.Text()
            if msg.isSetField(txt):
                msg.getField(txt)
                LOG.warning("[TRADE] Order rejected: %s", txt.getValue())
            return

        if ex.getValue() != "F":
            return

        sym = fix.Symbol()
        if msg.isSetField(sym):
            msg.getField(sym)
            if sym.getValue() != str(self.symbol_id):
                return

        self.request_positions()

    def on_biz_reject(self, msg: fix.Message):
        txt = fix.Text()
        if msg.isSetField(txt):
            msg.getField(txt)
            LOG.warning("[REJECT] BusinessMessageReject: %s", txt.getValue())

    def on_md_reject(self, msg: fix.Message):
        txt = fix.Text()
        if msg.isSetField(txt):
            msg.getField(txt)
            LOG.warning("[REJECT] MarketDataRequestReject: %s", txt.getValue())

    # ----------------------------
    # Strategy: run on M15 close
    # ----------------------------
    def on_bar_close(self, bar):
        t, o, h, l, c = bar
        
        # Record bar if position is open
        if self.cur_pos != 0:
            self.path_recorder.add_bar(bar)
        
        action = self.policy.decide(self.bars)
        desired = -1 if action == 0 else (0 if action == 1 else 1)

        LOG.info(
            "[BAR M%d] %s O=%.2f H=%.2f L=%.2f C=%.2f | desired=%s cur=%s",
            self.timeframe_minutes,
            t.isoformat(),
            o,
            h,
            l,
            c,
            desired,
            self.cur_pos,
        )

        if not self.trade_sid:
            return
        if desired == self.cur_pos:
            return

        delta = desired - self.cur_pos
        side = "1" if delta > 0 else "2"
        order_qty = abs(delta) * self.qty
        self.send_market_order(side=side, qty=order_qty)

    def send_market_order(self, side: str, qty: float):
        self.clord_counter += 1
        clid = f"cl_{int(time.time())}_{self.clord_counter}"

        order = fix44.NewOrderSingle()
        order.setField(fix.ClOrdID(clid))
        order.setField(fix.Symbol(str(self.symbol_id)))
        order.setField(fix.Side(side))
        order.setField(fix.TransactTime(utc_ts_ms()))
        order.setField(fix.OrdType("1"))
        order.setField(fix.OrderQty(qty))
        
        # Start MFE/MAE tracking and path recording
        if self.best_bid and self.best_ask:
            entry_price = (self.best_bid + self.best_ask) / 2.0
            direction = 1 if side == "1" else -1
            self.trade_entry_time = utc_now()  # Store for performance tracker
            self.mfe_mae_tracker.start_tracking(entry_price, direction)
            self.path_recorder.start_recording(self.trade_entry_time, entry_price, direction)

        fix.Session.sendToTarget(order, self.trade_sid)
        LOG.info(
            "[TRADE] Sent MKT %s qty=%.6f clOrdID=%s",
            ("BUY" if side == "1" else "SELL"),
            qty,
            clid,
        )


# ----------------------------
# Main: start two initiators
# ----------------------------
def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def main():
    # Require creds explicitly so you don't get silent logouts.
    user = require_env("CTRADER_USERNAME")
    _ = require_env("CTRADER_PASSWORD_QUOTE")
    _ = require_env("CTRADER_PASSWORD_TRADE")

    symbol_id = int(os.environ.get("CTRADER_BTC_SYMBOL_ID", "10028"))
    qty = float(os.environ.get("CTRADER_QTY", "0.10"))
    timeframe_minutes = int(os.environ.get("CTRADER_TIMEFRAME_MIN", "15"))

    cfg_quote = os.environ.get("CTRADER_CFG_QUOTE", "ctrader_quote.cfg")
    cfg_trade = os.environ.get("CTRADER_CFG_TRADE", "ctrader_trade.cfg")

    LOG.info("symbol_id=%s qty=%s timeframe=M%d", symbol_id, qty, timeframe_minutes)
    LOG.info("cfg_quote=%s", cfg_quote)
    LOG.info("cfg_trade=%s", cfg_trade)
    LOG.info("CTRADER_USERNAME=%s", user)

    app = CTraderFixApp(symbol_id=symbol_id, qty=qty, timeframe_minutes=timeframe_minutes)

    settings_q = fix.SessionSettings(cfg_quote)
    store_q = fix.FileStoreFactory(settings_q)
    log_q = fix.FileLogFactory(settings_q)
    initiator_q = fix.SocketInitiator(app, store_q, settings_q, log_q)

    settings_t = fix.SessionSettings(cfg_trade)
    store_t = fix.FileStoreFactory(settings_t)
    log_t = fix.FileLogFactory(settings_t)
    initiator_t = fix.SocketInitiator(app, store_t, settings_t, log_t)

    initiator_q.start()
    initiator_t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        initiator_q.stop()
        initiator_t.stop()


if __name__ == "__main__":
    main()
