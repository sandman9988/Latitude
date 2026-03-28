"""
Tick data and deal history downloader.
Supplements OHLCV bars with:
  - Historical bid + ask tick streams (for VPIN, order flow classification)
  - Deal history (actual fills, commissions, swap charges — friction calibration)
  - Cash flow history (deposits, withdrawals, actual swap deductions)

Historical DOM is NOT available from cTrader — DOM is live-only (subscribe/stream).
Record live DOM during paper/live trading via ctrader/dom_recorder.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from core.logger import get_logger
from core.math_utils import safe_div
from core.numeric import non_negative

logger = get_logger("tick_downloader")

INTER_REQUEST_SLEEP_S = 0.25
TICK_CHUNK_HOURS = 6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TickBar:
    """Single bid or ask tick."""
    timestamp: float   # unix seconds
    price: float
    side: str          # "bid" or "ask"
    symbol: str = ""


@dataclass
class DealRecord:
    """
    Actual executed deal from deal history.
    Used to calibrate friction model with real broker fills.
    """
    deal_id: int
    order_id: int
    position_id: int
    symbol_id: int
    timestamp: float
    execution_price: float
    volume: float          # in lots
    side: str              # "buy" or "sell"
    commission: float      # in account currency (negative = cost)
    swap: float            # in account currency
    gross_pnl: float       # gross P&L if closing trade
    margin_rate: float
    is_closing: bool


@dataclass
class CashFlowRecord:
    """Deposit, withdrawal, or swap deduction from cash flow history."""
    timestamp: float
    amount: float          # positive = credit, negative = debit
    flow_type: str         # "deposit", "withdrawal", "swap", "commission", "bonus"
    balance_after: float


# ---------------------------------------------------------------------------
# Tick data download
# ---------------------------------------------------------------------------

def download_ticks(
    conn,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    output_dir: Optional[Path] = None,
) -> Tuple[List[TickBar], List[TickBar]]:
    """
    Download historical bid AND ask tick data.
    Returns (bid_ticks, ask_ticks) sorted by timestamp.
    Saves CSVs to output_dir/{symbol}/ticks/ if provided.
    """
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
    except ImportError:
        raise ImportError("ctrader_open_api not installed")

    symbol = symbol.strip().upper()
    symbol_id = conn.find_symbol_id(symbol)
    if symbol_id is None:
        raise ValueError(f"Symbol not found: {symbol}")
    time.sleep(INTER_REQUEST_SLEEP_S)

    digits = conn.get_digits(symbol_id)
    pip_position = max(digits - 1, 0)
    scale = float(10 ** max(digits, 0))
    time.sleep(INTER_REQUEST_SLEEP_S)

    bid_ticks = _download_tick_side(conn, account_id=conn.credentials.account_id,
                                     symbol_id=symbol_id, symbol=symbol,
                                     side="bid", tick_type=1,
                                     start_dt=start_dt, end_dt=end_dt, scale=scale)
    time.sleep(INTER_REQUEST_SLEEP_S)

    ask_ticks = _download_tick_side(conn, account_id=conn.credentials.account_id,
                                     symbol_id=symbol_id, symbol=symbol,
                                     side="ask", tick_type=2,
                                     start_dt=start_dt, end_dt=end_dt, scale=scale)

    logger.info(
        f"Downloaded ticks: {len(bid_ticks)} bid, {len(ask_ticks)} ask for {symbol}",
        symbol=symbol, component="tick_downloader"
    )

    if output_dir:
        _save_ticks_csv(bid_ticks, ask_ticks, symbol, output_dir)

    return bid_ticks, ask_ticks


def _download_tick_side(
    conn,
    account_id: int,
    symbol_id: int,
    symbol: str,
    side: str,
    tick_type: int,   # 1=bid, 2=ask
    start_dt: datetime,
    end_dt: datetime,
    scale: float,
) -> List[TickBar]:
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
    except ImportError:
        return []

    ticks: List[TickBar] = []
    chunk_span = timedelta(hours=TICK_CHUNK_HOURS)
    cur = start_dt

    while cur < end_dt:
        chunk_end = min(cur + chunk_span, end_dt)
        page_start = cur

        while page_start < chunk_end:
            req = api_msgs.ProtoOAGetTickDataReq()
            req.ctidTraderAccountId = account_id
            req.symbolId = symbol_id
            req.type = tick_type
            req.fromTimestamp = int(page_start.timestamp() * 1000)
            req.toTimestamp = int(chunk_end.timestamp() * 1000)

            resp = conn.send_and_wait(req, timeout_s=30.0)
            if resp is None or hasattr(resp, "errorCode"):
                break

            page = _decode_tick_page(getattr(resp, "tickData", []), scale, side, symbol)
            if not page:
                break
            ticks.extend(page)

            if not bool(getattr(resp, "hasMore", False)):
                break
            page_start = datetime.fromtimestamp(page[-1].timestamp + 0.001, tz=timezone.utc)
            time.sleep(INTER_REQUEST_SLEEP_S)

        cur = chunk_end
        time.sleep(INTER_REQUEST_SLEEP_S)

    return sorted(ticks, key=lambda t: t.timestamp)


def _decode_tick_page(tick_rows, scale: float, side: str, symbol: str) -> List[TickBar]:
    ticks = []
    ts_cur: Optional[int] = None
    price_cur: Optional[int] = None

    for idx, row in enumerate(tick_rows):
        ts_raw = int(getattr(row, "timestamp", 0) or 0)
        tick_raw = int(getattr(row, "tick", 0) or 0)

        if idx == 0 or ts_cur is None or price_cur is None:
            ts_cur = ts_raw
            price_cur = tick_raw
        else:
            ts_cur += ts_raw
            price_cur += tick_raw

        price = price_cur / scale
        if price > 0:
            ticks.append(TickBar(
                timestamp=ts_cur / 1000.0,
                price=price,
                side=side,
                symbol=symbol,
            ))

    return ticks


def _save_ticks_csv(
    bid_ticks: List[TickBar],
    ask_ticks: List[TickBar],
    symbol: str,
    output_dir: Path,
) -> None:
    tick_dir = output_dir / symbol / "ticks"
    tick_dir.mkdir(parents=True, exist_ok=True)

    for side, ticks in [("bid", bid_ticks), ("ask", ask_ticks)]:
        if not ticks:
            continue
        t0 = datetime.fromtimestamp(ticks[0].timestamp, tz=timezone.utc).strftime("%Y%m%d")
        t1 = datetime.fromtimestamp(ticks[-1].timestamp, tz=timezone.utc).strftime("%Y%m%d")
        path = tick_dir / f"{symbol}_{side}_{t0}_{t1}.csv"
        with open(path, "w") as f:
            f.write("timestamp,price\n")
            for tick in ticks:
                dt = datetime.fromtimestamp(tick.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"{dt},{tick.price}\n")
        logger.info(f"Saved {len(ticks)} {side} ticks to {path}", component="tick_downloader")


# ---------------------------------------------------------------------------
# Deal history download
# ---------------------------------------------------------------------------

def download_deals(
    conn,
    start_dt: datetime,
    end_dt: datetime,
    max_rows: int = 5000,
    output_dir: Optional[Path] = None,
) -> List[DealRecord]:
    """
    Download historical deal fills from the account.
    Used to calibrate friction costs and measure real slippage.
    """
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
    except ImportError:
        raise ImportError("ctrader_open_api not installed")

    account_id = conn.credentials.account_id
    deals: List[DealRecord] = []
    chunk_days = 30
    cur = start_dt

    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=chunk_days), end_dt)

        req = api_msgs.ProtoOADealListReq()
        req.ctidTraderAccountId = account_id
        req.fromTimestamp = int(cur.timestamp() * 1000)
        req.toTimestamp = int(chunk_end.timestamp() * 1000)
        req.maxRows = max_rows

        resp = conn.send_and_wait(req, timeout_s=30.0)
        if resp and not hasattr(resp, "errorCode"):
            for d in getattr(resp, "deal", []):
                record = _decode_deal(d)
                if record:
                    deals.append(record)

        cur = chunk_end
        time.sleep(INTER_REQUEST_SLEEP_S)

    deals.sort(key=lambda d: d.timestamp)
    logger.info(f"Downloaded {len(deals)} deals", component="tick_downloader")

    if output_dir and deals:
        _save_deals_csv(deals, output_dir)

    return deals


def _decode_deal(d) -> Optional[DealRecord]:
    deal_id = int(getattr(d, "dealId", 0) or 0)
    if deal_id == 0:
        return None

    close_detail = getattr(d, "closePositionDetail", None)
    gross_pnl = 0.0
    if close_detail:
        gross_pnl = float(getattr(close_detail, "grossProfit", 0) or 0) / 100.0

    # cTrader stores money in 1/100 units
    commission_raw = int(getattr(d, "commission", 0) or 0)
    commission = float(commission_raw) / 100.0

    volume_raw = int(getattr(d, "filledVolume", 0) or getattr(d, "volume", 0) or 0)
    volume_lots = float(volume_raw) / 100.0  # centilots to lots

    trade_side = int(getattr(d, "tradeSide", 1) or 1)
    side = "buy" if trade_side == 1 else "sell"

    status = int(getattr(d, "dealStatus", 2) or 2)
    if status != 2:  # only FILLED deals
        return None

    return DealRecord(
        deal_id=deal_id,
        order_id=int(getattr(d, "orderId", 0) or 0),
        position_id=int(getattr(d, "positionId", 0) or 0),
        symbol_id=int(getattr(d, "symbolId", 0) or 0),
        timestamp=int(getattr(d, "executionTimestamp", 0) or 0) / 1000.0,
        execution_price=float(getattr(d, "executionPrice", 0.0) or 0.0),
        volume=volume_lots,
        side=side,
        commission=commission,
        swap=0.0,  # swap is in position, not deal
        gross_pnl=gross_pnl,
        margin_rate=float(getattr(d, "marginRate", 0.0) or 0.0),
        is_closing=close_detail is not None,
    )


def _save_deals_csv(deals: List[DealRecord], output_dir: Path) -> None:
    path = output_dir / "deals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("timestamp,deal_id,symbol_id,side,price,volume,commission,gross_pnl,margin_rate,is_closing\n")
        for d in deals:
            dt = datetime.fromtimestamp(d.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{dt},{d.deal_id},{d.symbol_id},{d.side},{d.execution_price},"
                    f"{d.volume},{d.commission},{d.gross_pnl},{d.margin_rate},{d.is_closing}\n")


# ---------------------------------------------------------------------------
# Cash flow history download
# ---------------------------------------------------------------------------

def download_cash_flow(
    conn,
    start_dt: datetime,
    end_dt: datetime,
    output_dir: Optional[Path] = None,
) -> List[CashFlowRecord]:
    """
    Download deposit/withdrawal/swap history.
    Useful for measuring actual swap costs over time.
    """
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
    except ImportError:
        raise ImportError("ctrader_open_api not installed")

    account_id = conn.credentials.account_id
    records: List[CashFlowRecord] = []

    req = api_msgs.ProtoOACashFlowHistoryListReq()
    req.ctidTraderAccountId = account_id
    req.fromTimestamp = int(start_dt.timestamp() * 1000)
    req.toTimestamp = int(end_dt.timestamp() * 1000)

    resp = conn.send_and_wait(req, timeout_s=30.0)
    if resp and not hasattr(resp, "errorCode"):
        for dw in getattr(resp, "depositWithdraw", []):
            record = _decode_cash_flow(dw)
            if record:
                records.append(record)

    records.sort(key=lambda r: r.timestamp)
    logger.info(f"Downloaded {len(records)} cash flow records", component="tick_downloader")

    if output_dir and records:
        _save_cash_flow_csv(records, output_dir)

    return records


def _decode_cash_flow(dw) -> Optional[CashFlowRecord]:
    ts = int(getattr(dw, "changeBalanceTimestamp", 0) or 0)
    if ts == 0:
        return None

    amount_raw = int(getattr(dw, "delta", 0) or 0)
    balance_raw = int(getattr(dw, "balance", 0) or 0)
    amount = float(amount_raw) / 100.0
    balance = float(balance_raw) / 100.0

    # ExternalId or description gives the type
    reason = int(getattr(dw, "externalId", 0) or 0)
    external_note = str(getattr(dw, "externalNote", "") or "").lower()

    if "swap" in external_note:
        flow_type = "swap"
    elif "commission" in external_note:
        flow_type = "commission"
    elif amount > 0:
        flow_type = "deposit"
    else:
        flow_type = "withdrawal"

    return CashFlowRecord(
        timestamp=ts / 1000.0,
        amount=amount,
        flow_type=flow_type,
        balance_after=balance,
    )


def _save_cash_flow_csv(records: List[CashFlowRecord], output_dir: Path) -> None:
    path = output_dir / "cash_flow.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("timestamp,amount,flow_type,balance_after\n")
        for r in records:
            dt = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{dt},{r.amount},{r.flow_type},{r.balance_after}\n")


# ---------------------------------------------------------------------------
# Friction calibrator — uses deal history to measure real vs estimated costs
# ---------------------------------------------------------------------------

def calibrate_friction_from_deals(
    deals: List[DealRecord],
    spec_commission_per_lot: float,
) -> Dict[str, float]:
    """
    Compare actual commissions from deal history vs spec estimate.
    Returns calibration metrics for the runway predictor.
    """
    if not deals:
        return {}

    commissions = [abs(d.commission) for d in deals if d.commission != 0 and d.volume > 0]
    commission_per_lot = [safe_div(abs(d.commission), d.volume) for d in deals
                          if d.commission != 0 and d.volume > 0]

    if not commission_per_lot:
        return {}

    mean_commission = sum(commission_per_lot) / len(commission_per_lot)
    ratio = safe_div(mean_commission, spec_commission_per_lot, fallback=1.0)

    # Slippage: compare intended vs actual (requires order price — approximate from deal price variance)
    return {
        "mean_commission_per_lot": mean_commission,
        "commission_ratio_vs_spec": ratio,
        "n_deals": float(len(deals)),
        "total_commission_paid": sum(commissions),
    }
