"""Ortak yardımcılar: zaman aralıkları ve sayı biçimlendirme."""

from __future__ import annotations

import math

INTERVALS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT", "SUIUSDT",
]


def interval_seconds(interval: str) -> int:
    if interval not in INTERVALS:
        raise ValueError(f"Geçersiz zaman aralığı: {interval}")
    return INTERVALS[interval]


def is_finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_price(value) -> str:
    if not is_finite(value):
        return "-"
    v = float(value)
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:,.4f}"
    text = f"{v:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def fmt_money(value, asset: str = "") -> str:
    if not is_finite(value):
        return "-"
    text = f"{float(value):,.2f}"
    return f"{text} {asset}".strip()


def fmt_pct(value, signed: bool = True) -> str:
    if value is None:
        return "-"
    v = float(value)
    if math.isinf(v):
        return "∞"
    if math.isnan(v):
        return "-"
    return f"{v:+.2f}%" if signed else f"{v:.2f}%"


def fmt_qty(value) -> str:
    if not is_finite(value):
        return "-"
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"
