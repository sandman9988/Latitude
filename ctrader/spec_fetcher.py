"""
Symbol contract spec fetcher.
Pulls the complete ProtoOASymbol + dynamic leverage + asset/category metadata
and builds a fully populated BrokerSpec.

Usage:
    creds = CTraderCredentials.from_env()
    conn = CTraderConnector(creds)
    conn.start()
    spec = fetch_spec(conn, "XAUUSD")
    spec = fetch_spec(conn, "DE40")
    conn.stop()
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.validator import (
    BrokerSpec, LeverageTier, TradingInterval, Holiday
)
from core.math_utils import safe_div
from core.logger import get_logger

logger = get_logger("spec_fetcher")

INTER_REQUEST_SLEEP_S = 0.25


def fetch_spec(
    conn,
    symbol: str,
    account_currency: str = "USD",
    money_digits: int = 2,
) -> BrokerSpec:
    """
    Fetch a complete BrokerSpec for a symbol from the cTrader Open API.
    Combines: symbol spec, asset info, category, dynamic leverage, conversion chain.
    """
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
    except ImportError:
        raise ImportError("ctrader_open_api not installed. Run: pip install ctrader-open-api")

    symbol = symbol.strip().upper()
    account_id = conn.credentials.account_id

    # 1. Find symbol ID
    symbol_id = conn.find_symbol_id(symbol)
    if symbol_id is None:
        raise ValueError(f"Symbol not found: {symbol}")
    time.sleep(INTER_REQUEST_SLEEP_S)

    # 2. Fetch full symbol spec
    req = api_msgs.ProtoOASymbolByIdReq()
    req.ctidTraderAccountId = account_id
    req.symbolId.append(symbol_id)
    resp = conn.send_and_wait(req, timeout_s=20.0)
    if resp is None or hasattr(resp, "errorCode") or not getattr(resp, "symbol", []):
        raise RuntimeError(f"Failed to fetch symbol spec for {symbol}")
    sym = resp.symbol[0]
    time.sleep(INTER_REQUEST_SLEEP_S)

    # 3. Fetch assets map (for base/quote asset names)
    assets = _fetch_assets(conn, account_id)
    time.sleep(INTER_REQUEST_SLEEP_S)

    # 4. Fetch symbol categories
    categories = _fetch_categories(conn, account_id)
    time.sleep(INTER_REQUEST_SLEEP_S)

    # 5. Fetch asset classes
    asset_classes = _fetch_asset_classes(conn, account_id)
    time.sleep(INTER_REQUEST_SLEEP_S)

    # 6. Fetch dynamic leverage if available
    leverage_tiers: List[LeverageTier] = []
    leverage_id = int(getattr(sym, "leverageId", 0) or 0)
    if leverage_id > 0:
        leverage_tiers = _fetch_leverage_tiers(conn, account_id, leverage_id)
        time.sleep(INTER_REQUEST_SLEEP_S)

    # 7. Fetch conversion chain (to account currency)
    conversion_symbols: List[str] = []
    base_asset_id = int(getattr(sym, "baseAssetId", 0) or 0)
    quote_asset_id = int(getattr(sym, "quoteAssetId", 0) or 0)
    conversion_symbols = _fetch_conversion_chain(
        conn, account_id, base_asset_id, quote_asset_id,
        assets, account_currency
    )
    time.sleep(INTER_REQUEST_SLEEP_S)

    # 8. Assemble BrokerSpec
    return _build_spec(
        sym=sym,
        symbol=symbol,
        symbol_id=symbol_id,
        assets=assets,
        categories=categories,
        asset_classes=asset_classes,
        leverage_tiers=leverage_tiers,
        conversion_symbols=conversion_symbols,
        account_currency=account_currency,
        money_digits=money_digits,
    )


def fetch_all_specs(
    conn,
    symbols: List[str],
    account_currency: str = "USD",
    money_digits: int = 2,
) -> Dict[str, BrokerSpec]:
    """Fetch specs for multiple symbols. Returns dict keyed by symbol name."""
    result = {}
    for symbol in symbols:
        try:
            spec = fetch_spec(conn, symbol, account_currency, money_digits)
            result[symbol.upper()] = spec
            logger.info(f"Fetched spec: {symbol}", component="spec_fetcher")
        except Exception as e:
            logger.error(f"Failed to fetch spec for {symbol}: {e}", component="spec_fetcher")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_assets(conn, account_id: int) -> Dict[int, str]:
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        req = api_msgs.ProtoOAAssetListReq()
        req.ctidTraderAccountId = account_id
        resp = conn.send_and_wait(req, timeout_s=15.0)
        if resp is None or hasattr(resp, "errorCode"):
            return {}
        return {
            int(getattr(a, "assetId", 0)): str(getattr(a, "name", "") or "").upper()
            for a in getattr(resp, "asset", [])
            if getattr(a, "assetId", None)
        }
    except Exception as e:
        logger.warning(f"Failed to fetch assets: {e}")
        return {}


def _fetch_categories(conn, account_id: int) -> Dict[int, dict]:
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        req = api_msgs.ProtoOASymbolCategoryListReq()
        req.ctidTraderAccountId = account_id
        resp = conn.send_and_wait(req, timeout_s=15.0)
        if resp is None or hasattr(resp, "errorCode"):
            return {}
        return {
            int(getattr(c, "id", 0)): {
                "name": str(getattr(c, "name", "") or ""),
                "asset_class_id": int(getattr(c, "assetClassId", 0) or 0),
            }
            for c in getattr(resp, "symbolCategory", [])
            if getattr(c, "id", None)
        }
    except Exception as e:
        logger.warning(f"Failed to fetch categories: {e}")
        return {}


def _fetch_asset_classes(conn, account_id: int) -> Dict[int, str]:
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        req = api_msgs.ProtoOAAssetClassListReq()
        req.ctidTraderAccountId = account_id
        resp = conn.send_and_wait(req, timeout_s=15.0)
        if resp is None or hasattr(resp, "errorCode"):
            return {}
        return {
            int(getattr(ac, "id", 0)): str(getattr(ac, "name", "") or "").upper()
            for ac in getattr(resp, "assetClass", [])
            if getattr(ac, "id", None)
        }
    except Exception as e:
        logger.warning(f"Failed to fetch asset classes: {e}")
        return {}


def _fetch_leverage_tiers(conn, account_id: int, leverage_id: int) -> List[LeverageTier]:
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        req = api_msgs.ProtoOAGetDynamicLeverageByIDReq()
        req.ctidTraderAccountId = account_id
        req.leverageId = leverage_id
        resp = conn.send_and_wait(req, timeout_s=15.0)
        if resp is None or hasattr(resp, "errorCode"):
            return []
        tiers = []
        for t in getattr(getattr(resp, "leverage", None), "tier", []):
            volume = float(getattr(t, "volume", 0) or 0) / 100.0  # cTrader stores in centilots
            leverage = float(getattr(t, "leverage", 0) or 0) / 100.0  # stored in basis points
            if leverage > 0:
                tiers.append(LeverageTier(volume_threshold=volume, leverage=leverage))
        return sorted(tiers, key=lambda x: x.volume_threshold)
    except Exception as e:
        logger.warning(f"Failed to fetch leverage tiers for id {leverage_id}: {e}")
        return []


def _fetch_conversion_chain(
    conn,
    account_id: int,
    base_asset_id: int,
    quote_asset_id: int,
    assets: Dict[int, str],
    account_currency: str,
) -> List[str]:
    """
    Returns symbol names needed to convert P&L from quote currency to account currency.
    E.g. DE40 (EUR) on USD account → ["EURUSD"]
    """
    quote_currency = assets.get(quote_asset_id, "")
    if not quote_currency or quote_currency == account_currency:
        return []
    try:
        from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
        # Find asset ID for account currency
        acc_asset_id = next(
            (aid for aid, name in assets.items() if name == account_currency.upper()),
            None
        )
        if acc_asset_id is None:
            return []
        req = api_msgs.ProtoOASymbolsForConversionReq()
        req.ctidTraderAccountId = account_id
        req.firstAssetId = quote_asset_id
        req.lastAssetId = acc_asset_id
        resp = conn.send_and_wait(req, timeout_s=15.0)
        if resp is None or hasattr(resp, "errorCode"):
            return []
        return [
            str(getattr(s, "symbolName", "") or "")
            for s in getattr(resp, "symbol", [])
            if getattr(s, "symbolName", None)
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch conversion chain: {e}")
        return []


def _build_spec(
    sym: Any,
    symbol: str,
    symbol_id: int,
    assets: Dict[int, str],
    categories: Dict[int, dict],
    asset_classes: Dict[int, str],
    leverage_tiers: List[LeverageTier],
    conversion_symbols: List[str],
    account_currency: str,
    money_digits: int,
) -> BrokerSpec:
    """Assemble BrokerSpec from raw ProtoOASymbol fields."""

    digits = int(getattr(sym, "digits", 5) or 5)
    pip_position = int(getattr(sym, "pipPosition", max(digits - 1, 0)) or max(digits - 1, 0))
    pip_size = 10 ** (-pip_position)
    tick_size = 10 ** (-digits)

    # Lot size — cTrader stores in units (e.g. 100000 for FX, 10 for DAX)
    lot_size_raw = int(getattr(sym, "lotSize", 0) or 0)
    lot_size = float(lot_size_raw) / 100.0 if lot_size_raw > 0 else 100000.0

    # Volume — cTrader stores in centilots (1 lot = 100 centilots)
    min_vol = float(getattr(sym, "minVolume", 100) or 100) / 100.0
    max_vol = float(getattr(sym, "maxVolume", 10000) or 10000) / 100.0
    step_vol = float(getattr(sym, "stepVolume", 1) or 1) / 100.0

    # Commission
    commission_type = int(getattr(sym, "commissionType", 2) or 2)
    commission_raw = int(getattr(sym, "preciseTradingCommissionRate", 0) or
                         getattr(sym, "commission", 0) or 0)
    # preciseTradingCommissionRate is in 1/100000 units
    commission_rate = float(commission_raw) / 100000.0

    min_commission_raw = int(getattr(sym, "preciseMinCommission", 0) or
                              getattr(sym, "minCommission", 0) or 0)
    min_commission = float(min_commission_raw) / 100.0
    min_commission_type = int(getattr(sym, "minCommissionType", 1) or 1)

    # Swap
    swap_type = int(getattr(sym, "swapCalculationType", 0) or 0)
    swap_long = float(getattr(sym, "swapLong", 0.0) or 0.0)
    swap_short = float(getattr(sym, "swapShort", 0.0) or 0.0)
    swap_rollover = int(getattr(sym, "swapRollover3Days", 3) or 3)
    swap_period = int(getattr(sym, "swapPeriod", 1) or 1)

    # SL/TP distance constraints
    sl_dist_raw = int(getattr(sym, "slDistance", 0) or 0)
    tp_dist_raw = int(getattr(sym, "tpDistance", 0) or 0)
    dist_type = int(getattr(sym, "distanceSetIn", 1) or 1)
    sl_min = float(sl_dist_raw) / 10.0  # stored in 1/10 points
    tp_min = float(tp_dist_raw) / 10.0

    # P&L conversion fee
    pnl_fee_raw = int(getattr(sym, "pnlConversionFeeRate", 0) or 0)
    pnl_fee = float(pnl_fee_raw) / 100000.0  # stored in 1/100000

    # Trading schedule
    schedule = [
        TradingInterval(
            start_second=int(getattr(iv, "startSecond", 0) or 0),
            end_second=int(getattr(iv, "endSecond", 0) or 0),
        )
        for iv in getattr(sym, "schedule", [])
    ]
    schedule_tz = str(getattr(sym, "scheduleTimeZone", "UTC") or "UTC")

    # Holidays
    holidays = [
        Holiday(
            name=str(getattr(h, "name", "") or ""),
            date_timestamp=int(getattr(h, "holidayDate", 0) or 0),
            is_recurring=bool(getattr(h, "isRecurring", False)),
            start_second=int(getattr(h, "startSecond", 0) or 0),
            end_second=int(getattr(h, "endSecond", 86400) or 86400),
        )
        for h in getattr(sym, "holiday", [])
    ]

    # Asset/category/class names
    base_asset_id = int(getattr(sym, "baseAssetId", 0) or 0)
    quote_asset_id = int(getattr(sym, "quoteAssetId", 0) or 0)
    base_asset = assets.get(base_asset_id, "")
    quote_asset = assets.get(quote_asset_id, "")

    cat_id = int(getattr(sym, "symbolCategoryId", 0) or 0)
    cat_info = categories.get(cat_id, {})
    category_name = cat_info.get("name", "")
    asset_class_id = cat_info.get("asset_class_id", 0)
    asset_class_name = asset_classes.get(asset_class_id, "")

    # Tick value: monetary value of 1 tick move for 1 lot in account currency
    # For FX: tick_value = lot_size * tick_size
    # For indices: depends on contract, approximate from lot_size
    tick_value = lot_size * tick_size

    # Margin rate from leverage
    leverage_id = int(getattr(sym, "leverageId", 0) or 0)
    if leverage_tiers:
        # Use first tier (base leverage) as default
        base_leverage = leverage_tiers[0].leverage if leverage_tiers else 30.0
        margin_rate = safe_div(1.0, base_leverage, fallback=0.01)
    else:
        margin_rate = 0.01  # fallback 1:100

    return BrokerSpec(
        symbol=symbol,
        symbol_id=symbol_id,
        description=str(getattr(sym, "description", "") or getattr(sym, "symbolName", symbol) or symbol),
        base_asset=base_asset,
        quote_asset=quote_asset,
        asset_class=asset_class_name,
        category=category_name,
        digits=digits,
        pip_size=pip_size,
        tick_size=tick_size,
        tick_value=tick_value,
        lot_size=lot_size,
        lot_step=step_vol,
        min_volume=min_vol,
        max_volume=max_vol,
        margin_rate=margin_rate,
        leverage_tiers=leverage_tiers,
        commission_type=commission_type,
        commission_rate=commission_rate,
        min_commission=min_commission,
        min_commission_type=min_commission_type,
        swap_type=swap_type,
        swap_long=swap_long,
        swap_short=swap_short,
        swap_rollover_day=swap_rollover,
        swap_period=swap_period,
        spread_pips=1.0,  # placeholder — overwritten by tick-based measurement
        sl_min_distance=sl_min,
        tp_min_distance=tp_min,
        distance_type=dist_type,
        pnl_conversion_fee_rate=pnl_fee,
        schedule=schedule,
        schedule_timezone=schedule_tz,
        holidays=holidays,
        trading_mode=int(getattr(sym, "tradingMode", 0) or 0),
        short_selling_enabled=bool(getattr(sym, "enableShortSelling", True)),
        guaranteed_sl_available=bool(getattr(sym, "guaranteedStopLoss", False)),
        swap_free=False,
        currency=account_currency,
        money_digits=money_digits,
        conversion_symbols=conversion_symbols,
    )
