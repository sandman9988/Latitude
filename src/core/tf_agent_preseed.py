"""Cold-start replay-buffer seeding logic for :class:`TFAgent`.

Extracted verbatim from ``openapi_hub`` as a behaviour-preserving mixin. These
methods run only during warm-up (before live trading) to seed the harvester and
trigger replay buffers with synthetic HOLD/CLOSE and ENTRY/NO_ENTRY experiences
derived from bar history, so the agents do not start from a cold buffer.
"""

import logging
import math
from collections import deque
from typing import Any

import numpy as np

from src.utils.safe_math import SAFE_EPSILON, SAFE_SMALL

LOG = logging.getLogger(__name__)


class TFAgentPreseedMixin:
    """HOLD/CLOSE and trigger replay-buffer seeding from bar history."""

    _PRESEED_STOP_PCT: float = 0.003    # 0.3% adverse move = stop-out
    _PRESEED_TARGET_PCT: float = 0.002  # 0.2% favourable = target hit
    _PRESEED_MAX_HOLD: int = 20         # bars before force-exit

    def _compute_preseed_vol(self, bars_list: list, idx: int) -> float:
        """Rolling std of log-returns over preceding 10 bars at position idx."""
        window = bars_list[max(0, idx - 10):idx]
        if len(window) < 3:
            return 0.005
        closes = [b[4] for b in window if b[4] > 0]
        if len(closes) < 3:
            return 0.005
        try:
            rets = np.diff(np.log(np.array(closes, dtype=float)))
            v = float(np.std(rets))
            return v if v > 0 else 0.005
        except Exception:
            return 0.005

    def _preseed_harvester_from_bars(self) -> None:
        """Seed harvester replay buffer with realistic HOLD+CLOSE experiences.

        Simulates paper positions at every 3rd bar using stop-loss / take-profit logic
        so the harvester learns from real exit scenarios, not neutral no-ops.
        Ported from legacy ctrader_ddqn_paper.py _preseed_harvester_from_bars().
        """
        from src.core.openapi_hub import _MIN_BARS_BEFORE_TRADE
        self._harvester_preseeded = True
        bars_list = list(self.bars)
        n = len(bars_list)
        if n < _MIN_BARS_BEFORE_TRADE + 5:
            return

        added_hold = 0
        added_close = 0
        directions = [1, -1, 1, -1]  # alternate LONG/SHORT

        for d_idx, entry_idx in enumerate(range(_MIN_BARS_BEFORE_TRADE, n - 2, 3)):
            direction = directions[d_idx % len(directions)]
            hold_count, close_count = self._preseed_one_harvester_entry(bars_list, entry_idx, direction)
            added_hold += hold_count
            added_close += close_count

        LOG.info("[%s %s] Harvester preseed: %d HOLD + %d CLOSE experiences from %d bars",
                 self.symbol, self.tf_label, added_hold, added_close, n)

    def _preseed_one_harvester_entry(self, bars_list: list, entry_idx: int, direction: int) -> tuple[int, int]:
        entry_price = float(bars_list[entry_idx][4])
        if not math.isfinite(entry_price) or entry_price <= 0:
            return 0, 0

        stop_dist = entry_price * self._PRESEED_STOP_PCT
        target_dist = entry_price * self._PRESEED_TARGET_PCT
        stop_price = entry_price - direction * stop_dist
        target_price = entry_price + direction * target_dist
        vol_entry = self._compute_preseed_vol(bars_list, entry_idx)

        prev_harv_state = None
        prev_mfe = 0.0
        prev_mae = 0.0
        pnl_pts = 0.0
        added_hold = 0
        for hold_step in range(1, self._PRESEED_MAX_HOLD + 1):
            bar_idx = entry_idx + hold_step
            if bar_idx >= len(bars_list):
                break
            bar_high, bar_low, bar_close = map(float, bars_list[bar_idx][2:5])
            if not math.isfinite(bar_close) or bar_close <= 0:
                break

            cur_mfe, cur_mae = self._preseed_mfe_mae(
                direction,
                entry_price,
                bar_high,
                bar_low,
                prev_mfe,
                prev_mae,
            )
            harv_state = self._build_preseed_harvester_state(
                bars_list,
                bar_idx,
                vol_entry,
                cur_mfe,
                cur_mae,
                hold_step,
                entry_price,
            )
            if harv_state is None:
                break
            if prev_harv_state is not None and self._add_preseed_hold_experience(
                prev_harv_state,
                harv_state,
                entry_price,
                bar_close,
                direction,
                cur_mfe,
                cur_mae,
                prev_mfe,
                prev_mae,
                vol_entry,
                hold_step,
            ):
                added_hold += 1

            prev_harv_state, prev_mfe, prev_mae = harv_state, cur_mfe, cur_mae
            pnl_pts = self._preseed_exit_pnl(
                direction,
                bar_high,
                bar_low,
                bar_close,
                entry_price,
                stop_price,
                target_price,
                stop_dist,
                target_dist,
                hold_step,
            )
            if pnl_pts is not None:
                break

        added_close = self._add_preseed_close_experience(prev_harv_state, pnl_pts or 0.0, prev_mfe)
        return added_hold, added_close

    def _preseed_mfe_mae(
        self,
        direction: int,
        entry_price: float,
        bar_high: float,
        bar_low: float,
        prev_mfe: float,
        prev_mae: float,
    ) -> tuple[float, float]:
        fav_price = bar_high if direction == 1 else bar_low
        adv_price = bar_low if direction == 1 else bar_high
        cur_mfe = max(prev_mfe, (fav_price - entry_price) * direction)
        cur_mae = max(prev_mae, (entry_price - adv_price) * direction)
        return max(cur_mfe, 0.0), max(cur_mae, 0.0)

    def _build_preseed_harvester_state(
        self,
        bars_list: list,
        bar_idx: int,
        vol_entry: float,
        cur_mfe: float,
        cur_mae: float,
        hold_step: int,
        entry_price: float,
    ) -> Any | None:
        window_step = deque(bars_list[:bar_idx + 1], maxlen=2000)
        try:
            market = self.policy._build_state(
                window_step,
                imbalance=0.0,
                vpin_z=0.0,
                depth_ratio=1.0,
                realized_vol=vol_entry,
                event_features=None,
            )
            harvester = getattr(self.policy, "harvester", None)
            if harvester is None:
                return None
            return harvester._build_full_state(
                market,
                mfe=cur_mfe,
                mae=cur_mae,
                ticks_held=hold_step,
                entry_price=entry_price,
            ).copy()
        except Exception:
            return None

    def _add_preseed_hold_experience(
        self,
        prev_harv_state: Any,
        harv_state: Any,
        entry_price: float,
        bar_close: float,
        direction: int,
        cur_mfe: float,
        cur_mae: float,
        prev_mfe: float,
        prev_mae: float,
        vol_entry: float,
        hold_step: int,
    ) -> bool:
        capture_ratio = ((bar_close - entry_price) * direction) / cur_mfe if cur_mfe > SAFE_EPSILON else 0.0
        capture_c = float(np.clip(capture_ratio * 0.4, 0.0, 0.4))
        mfe_delta = (cur_mfe - prev_mfe) / max(abs(entry_price), 1.0)
        mfe_g = float(np.clip(mfe_delta / max(vol_entry, SAFE_SMALL) * 0.3, -0.3, 0.3))
        mae_delta = (cur_mae - prev_mae) / max(abs(entry_price), 1.0)
        mae_p = float(-np.clip(mae_delta / max(vol_entry, SAFE_SMALL) * 0.4, 0.0, 0.4))
        bars_per_day = max(10, 1440 // max(1, self.timeframe_minutes))
        t_decay = -0.02 * min(hold_step / max(1, bars_per_day // 10), 10.0)
        hold_reward = float(np.clip(capture_c + mfe_g + mae_p + t_decay, -1.0, 1.0))
        try:
            self.policy.add_harvester_experience(
                state=prev_harv_state,
                action=0,
                reward=hold_reward,
                next_state=harv_state,
                done=False,
            )
            return True
        except Exception:
            return False

    def _preseed_exit_pnl(
        self,
        direction: int,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        entry_price: float,
        stop_price: float,
        target_price: float,
        stop_dist: float,
        target_dist: float,
        hold_step: int,
    ) -> float | None:
        hit_stop = (direction == 1 and bar_low <= stop_price) or (direction == -1 and bar_high >= stop_price)
        hit_target = (direction == 1 and bar_high >= target_price) or (direction == -1 and bar_low <= target_price)
        if hit_target:
            return target_dist
        if hit_stop:
            return -stop_dist
        if hold_step == self._PRESEED_MAX_HOLD:
            return (bar_close - entry_price) * direction
        return None

    def _add_preseed_close_experience(self, prev_harv_state: Any, pnl_pts: float, prev_mfe: float) -> int:
        if prev_harv_state is None:
            return 0
        capture_at_close = min(1.0, pnl_pts / prev_mfe) if prev_mfe > SAFE_EPSILON else 0.0
        close_reward = float(np.clip(capture_at_close, -1.0, 1.0))
        try:
            self.policy.add_harvester_experience(
                state=prev_harv_state,
                action=1,
                reward=close_reward,
                next_state=prev_harv_state,
                done=True,
            )
            return 1
        except Exception:
            return 0

    def _preseed_trigger_buffer(self) -> None:
        """Seed trigger replay buffer from bar history with synthetic LONG/SHORT/NO_ENTRY.

        Alternates between ENTRY experiences (forward-looking return reward) and
        NO_ENTRY experiences (reward=0, done=True) up to 50% buffer capacity.
        Ported from legacy ctrader_ddqn_paper.py _preseed_trigger_buffer().
        """
        from src.core.openapi_hub import _MIN_BARS_BEFORE_TRADE
        bars_list = list(self.bars)
        n = len(bars_list)
        if n < _MIN_BARS_BEFORE_TRADE + 5:
            return

        trig = getattr(self.policy, "trigger", None)
        if trig is None:
            return
        trig_buf = getattr(trig, "buffer", None)
        buf_capacity = getattr(trig_buf, "capacity", 10000)
        buf_size = getattr(trig_buf, "size", 0)
        if buf_size >= buf_capacity * 0.5:
            LOG.debug("[%s %s] Trigger buffer ≥50%% full — skip preseed", self.symbol, self.tf_label)
            return

        added_entry = 0
        added_no_entry = 0
        directions = [1, 2, 1, 2]
        d_idx = 0

        for entry_idx in range(_MIN_BARS_BEFORE_TRADE, n - 3, 2):
            entry_bar = bars_list[entry_idx]
            entry_price = float(entry_bar[4])
            if entry_price <= 0:
                continue

            window = deque(bars_list[:entry_idx + 1], maxlen=2000)
            vol = self._compute_preseed_vol(bars_list, entry_idx)
            try:
                trig_state = self.policy._build_state(
                    window, imbalance=0.0, vpin_z=0.0, depth_ratio=1.0,
                    realized_vol=vol, event_features=None,
                )
            except Exception:
                continue

            action = directions[d_idx % len(directions)]
            d_idx += 1
            direction = 1 if action == 1 else -1

            # Forward-looking reward: 3-bar return normalised by vol
            fwd_bar = bars_list[min(entry_idx + 3, n - 1)]
            fwd_price = float(fwd_bar[4])
            if fwd_price > 0 and entry_price > 0 and vol > 0:
                ret = (fwd_price - entry_price) * direction / entry_price
                trig_reward = float(np.clip(ret / vol, -2.0, 2.0))
            else:
                trig_reward = 0.0

            try:
                self.policy.add_trigger_experience(
                    state=trig_state, action=action,
                    reward=trig_reward, next_state=trig_state, done=True,
                )
                added_entry += 1
            except Exception:
                pass

            # Paired NO_ENTRY — keeps class balance
            if not self._trigger_no_entry_saturated():
                try:
                    self.policy.add_trigger_experience(
                        state=trig_state, action=0,
                        reward=0.0, next_state=trig_state, done=True,
                    )
                    added_no_entry += 1
                except Exception:
                    pass

        LOG.info("[%s %s] Trigger preseed: %d ENTRY + %d NO_ENTRY experiences from %d bars",
                 self.symbol, self.tf_label, added_entry, added_no_entry, n)
