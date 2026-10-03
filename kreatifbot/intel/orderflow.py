"""Order flow, open interest, funding ve tasfiye motorları.

Bu motorlar tek başına AL/SAT üretmez; yalnızca onay / risk özelliği üretir.
Örnek kombinasyonlar (fiyat↑+OI↑ vb.) yalnızca 'aday' olarak işaretlenir;
kârlı oldukları geçmiş verideki doğrulamayla ölçülmelidir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FundingConfig


# ---------------------------------------------------------------------- order book
def orderbook_features(book: dict | None, band_pct: float = 0.5, wall_mult: float = 5.0) -> dict:
    if not book or not book.get("bids") or not book.get("asks"):
        return {"available": False, "reason": "Order book verisi yok"}
    bids = np.array([[float(p), float(q)] for p, q in book["bids"]])
    asks = np.array([[float(p), float(q)] for p, q in book["asks"]])
    best_bid, best_ask = bids[0, 0], asks[0, 0]
    mid = (best_bid + best_ask) / 2
    lo, hi = mid * (1 - band_pct / 100), mid * (1 + band_pct / 100)
    bid_in = bids[bids[:, 0] >= lo]
    ask_in = asks[asks[:, 0] <= hi]
    bid_vol = float((bid_in[:, 0] * bid_in[:, 1]).sum())
    ask_vol = float((ask_in[:, 0] * ask_in[:, 1]).sum())
    total = bid_vol + ask_vol
    all_q = np.concatenate([bids[:, 1], asks[:, 1]])
    med = float(np.median(all_q)) if len(all_q) else 0.0
    walls_bid = [(float(p), float(q)) for p, q in bids if med and q >= wall_mult * med][:3]
    walls_ask = [(float(p), float(q)) for p, q in asks if med and q >= wall_mult * med][:3]
    return {
        "available": True,
        "best_bid": best_bid, "best_ask": best_ask, "mid": mid,
        "spread_pct": (best_ask - best_bid) / mid * 100,
        "bid_depth_quote": bid_vol, "ask_depth_quote": ask_vol,
        "imbalance": (bid_vol - ask_vol) / total if total else 0.0,
        "bid_walls": walls_bid, "ask_walls": walls_ask,
    }


def estimate_slippage_pct(book: dict | None, side: str, quote_amount: float) -> float | None:
    """Order book'u yürüyerek piyasa emrinin ortalama fiyat kaymasını tahmin eder. Veri yoksa None."""
    if not book:
        return None
    levels = book["asks"] if side.upper() in ("BUY", "LONG") else book["bids"]
    if not levels:
        return None
    best = float(levels[0][0])
    remaining, cost, qty = quote_amount, 0.0, 0.0
    for p, q in levels:
        p, q = float(p), float(q)
        take = min(q * p, remaining)
        cost += take
        qty += take / p
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or qty == 0:
        return None  # Derinlik yetersiz
    avg = cost / qty
    return abs(avg / best - 1) * 100


def aggressive_flow(agg_trades: list | None) -> dict:
    """aggTrades: m=True → alıcı maker, yani agresif SATIŞ."""
    if not agg_trades:
        return {"available": False, "reason": "aggTrades verisi yok"}
    buy = sum(float(t["q"]) for t in agg_trades if not t.get("m"))
    sell = sum(float(t["q"]) for t in agg_trades if t.get("m"))
    return {"available": True, "aggressive_buy": buy, "aggressive_sell": sell, "delta": buy - sell,
            "buy_ratio": buy / (buy + sell) if buy + sell else float("nan")}


def cvd_divergence(price_change: float, cvd_change: float) -> str:
    if np.isnan(price_change) or np.isnan(cvd_change):
        return "UNAVAILABLE"
    if price_change > 0 and cvd_change > 0:
        return "BULLISH_CONFIRMATION"
    if price_change > 0 > cvd_change:
        return "BEARISH_DIVERGENCE_CANDIDATE"
    if price_change < 0 < cvd_change:
        return "BULLISH_DIVERGENCE_CANDIDATE"
    if price_change < 0 and cvd_change < 0:
        return "BEARISH_CONFIRMATION"
    return "NEUTRAL"


def absorption(f: pd.DataFrame) -> pd.Series:
    """Yüksek hacim + dar mum aralığı: emilim (absorption) adayı."""
    return (f["rvol"] > 2) & (f["bar_range_atr"] < 0.6)


# ---------------------------------------------------------------------- funding
def classify_funding(rate: float | None, cfg: FundingConfig | None = None) -> str:
    cfg = cfg or FundingConfig()
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):
        return "UNAVAILABLE"
    a = abs(rate)
    if a <= cfg.neutral_abs:
        return "NEUTRAL"
    sign = "POSITIVE" if rate > 0 else "NEGATIVE"
    if a >= cfg.extreme_abs:
        return f"EXTREME_{sign}"
    if a >= cfg.elevated_abs:
        return f"ELEVATED_{sign}"
    return sign


def funding_series_class(rates: pd.Series, cfg: FundingConfig | None = None) -> pd.Series:
    return rates.map(lambda r: classify_funding(r, cfg))


# ---------------------------------------------------------------------- open interest
def oi_quadrant(price_change: float, oi_change: float) -> str:
    if price_change is None or oi_change is None or np.isnan(price_change) or np.isnan(oi_change):
        return "UNAVAILABLE"
    p = "PRICE_UP" if price_change > 0 else "PRICE_DOWN"
    o = "OI_UP" if oi_change > 0 else "OI_DOWN"
    return f"{p}+{o}"


OI_QUADRANT_NOTES = {
    "PRICE_UP+OI_UP": "Yeni long pozisyonlar açılıyor olabilir (momentum onayı adayı)",
    "PRICE_UP+OI_DOWN": "Short kapanışı / short squeeze olabilir (zayıf yükseliş adayı)",
    "PRICE_DOWN+OI_UP": "Yeni short pozisyonlar açılıyor olabilir (düşüş momentumu adayı)",
    "PRICE_DOWN+OI_DOWN": "Long tasfiyesi / pozisyon kapanışı olabilir (kapitülasyon adayı)",
}


# ---------------------------------------------------------------------- tasfiyeler
@dataclass
class LiquidationStats:
    available: bool
    reason: str = ""
    total: float = float("nan")
    long_liq: float = float("nan")
    short_liq: float = float("nan")
    spike_z: float = float("nan")
    imbalance: float = float("nan")
    velocity: float = float("nan")
    notes: list = field(default_factory=list)


def liquidation_stats(liq: pd.DataFrame | None, window: int = 50) -> LiquidationStats:
    """liq sütunları: liq_long, liq_short (USDT). Veri yoksa UNAVAILABLE."""
    if liq is None or liq.empty or "liq_long" not in liq or liq["liq_long"].notna().sum() < 10:
        return LiquidationStats(False, "Tasfiye verisi yok (Binance REST bu veriyi sağlamıyor)")
    total = liq["liq_long"].fillna(0) + liq["liq_short"].fillna(0)
    hist = total.iloc[-window - 1:-1]
    z = (total.iloc[-1] - hist.mean()) / (hist.std() or 1)
    ll, sl = float(liq["liq_long"].iloc[-1] or 0), float(liq["liq_short"].iloc[-1] or 0)
    return LiquidationStats(True, total=float(total.iloc[-1]), long_liq=ll, short_liq=sl, spike_z=float(z),
                            imbalance=(ll - sl) / (ll + sl) if ll + sl else 0.0,
                            velocity=float(total.iloc[-3:].mean() - total.iloc[-6:-3].mean()))
