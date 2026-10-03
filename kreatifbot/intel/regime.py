"""Piyasa rejimi motoru.

EMA dizilimi ve eğimi, ADX, ATR yüzdeliği, gerçekleşen volatilite, momentum,
hacim, VWAP ve piyasa yapısından her mum için rejim üretir (nedensel). BTC trendi,
funding ve OI son mumdaki sınıflandırmayı bağlam olarak düzeltir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .types import Regime


@dataclass
class RegimeResult:
    regime: Regime
    bull_frac: float = float("nan")
    bear_frac: float = float("nan")
    adx: float = float("nan")
    vol_percentile: float = float("nan")
    high_vol: bool = False
    reasons: list = field(default_factory=list)


def _b(s) -> pd.Series:
    return s.fillna(False).astype(float) if hasattr(s, "fillna") else s


def classify_regime(f: pd.DataFrame) -> pd.DataFrame:
    close = f["close"]
    ema200_ok = f["ema200"].notna()
    bull_parts = [
        close > f["ema20"], f["ema20"] > f["ema50"], (f["ema50"] > f["ema200"]) & ema200_ok,
        f["ema50_slope"] > 0.1, f["st_dir"] == 1, f["structure_trend"] == 1, close > f["vwap"],
        f["di_plus"] > f["di_minus"],
    ]
    bear_parts = [
        close < f["ema20"], f["ema20"] < f["ema50"], (f["ema50"] < f["ema200"]) & ema200_ok,
        f["ema50_slope"] < -0.1, f["st_dir"] == -1, f["structure_trend"] == -1, close < f["vwap"],
        f["di_minus"] > f["di_plus"],
    ]
    denom = 7 + ema200_ok.astype(float)
    bull = sum(_b(p) for p in bull_parts) / denom
    bear = sum(_b(p) for p in bear_parts) / denom
    adx = f["adx"]
    vp = f["vol_percentile"]
    slope = f["ema50_slope"]
    crosses = (np.sign(close - f["ema20"]).diff().abs() > 0).astype(float).rolling(20, min_periods=20).sum()
    ret3 = (close / close.shift(3) - 1) * 100
    panic = (ret3 < -3 * f["atr_pct"].shift(3)) & (vp >= 0.9) & (f["rvol"] > 1.5)

    valid = f["ema50"].notna() & adx.notna() & vp.notna()
    conds = [
        ~valid,
        panic,
        (bull >= 0.85) & (adx >= 30) & (slope > 0.3),
        (bear >= 0.85) & (adx >= 30) & (slope < -0.3),
        vp >= 0.9,
        (bull >= 0.65) & (adx >= 20),
        (bear >= 0.65) & (adx >= 20),
        (adx < 20) & (crosses >= 6),
        vp <= 0.1,
        (bull >= 0.55) & (slope > 0),
        (bear >= 0.65),
    ]
    choices = [Regime.UNKNOWN.value, Regime.PANIC.value, Regime.STRONG_BULL.value, Regime.STRONG_BEAR.value,
               Regime.HIGH_VOLATILITY.value, Regime.BULL.value, Regime.BEAR.value, Regime.CHOP.value,
               Regime.LOW_VOLATILITY.value, Regime.WEAK_BULL.value, Regime.BEAR.value]
    regime = np.select(conds, choices, Regime.SIDEWAYS.value)
    out = pd.DataFrame({"regime": regime, "bull_frac": bull, "bear_frac": bear,
                        "high_vol": (vp >= 0.9).fillna(False)}, index=f.index)
    return out


_DOWNGRADE = {Regime.STRONG_BULL: Regime.BULL, Regime.BULL: Regime.WEAK_BULL, Regime.WEAK_BULL: Regime.SIDEWAYS,
              Regime.STRONG_BEAR: Regime.BEAR}


def regime_at(f: pd.DataFrame, reg: pd.DataFrame, i: int = -1, btc_bias: float | None = None,
              funding_class: str | None = None, oi_change_pct: float | None = None) -> RegimeResult:
    row = reg.iloc[i]
    r = Regime(row["regime"])
    fr = f.iloc[i]
    res = RegimeResult(regime=r, bull_frac=float(row["bull_frac"]), bear_frac=float(row["bear_frac"]),
                       adx=float(fr["adx"]) if pd.notna(fr["adx"]) else float("nan"),
                       vol_percentile=float(fr["vol_percentile"]) if pd.notna(fr["vol_percentile"]) else float("nan"),
                       high_vol=bool(row["high_vol"]))
    if r == Regime.UNKNOWN:
        res.reasons.append("Yetersiz gösterge geçmişi (EMA50/ADX/volatilite yüzdeliği yok)")
        return res
    res.reasons.append(f"EMA/yapı yükseliş oranı {res.bull_frac:.0%}, düşüş oranı {res.bear_frac:.0%}")
    res.reasons.append(f"ADX {res.adx:.1f}, ATR yüzdeliği {res.vol_percentile:.0%}")
    if btc_bias is not None and not np.isnan(btc_bias):
        if btc_bias <= -0.5 and res.regime in _DOWNGRADE and res.regime != Regime.STRONG_BEAR:
            res.reasons.append(f"BTC trendi düşüşte ({btc_bias:+.2f}) → rejim bir kademe düşürüldü")
            res.regime = _DOWNGRADE[res.regime]
        else:
            res.reasons.append(f"BTC eğilimi {btc_bias:+.2f}")
    if funding_class in ("EXTREME_POSITIVE", "EXTREME_NEGATIVE"):
        res.reasons.append(f"Funding aşırı ({funding_class}): kalabalık pozisyon riski")
        if res.regime in (Regime.STRONG_BULL, Regime.STRONG_BEAR):
            res.regime = _DOWNGRADE[res.regime]
    if oi_change_pct is not None and not np.isnan(oi_change_pct):
        res.reasons.append(f"OI değişimi %{oi_change_pct:+.2f}")
    return res
