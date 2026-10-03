"""Piyasa analizi: trend, rejim, volatilite, destek/direnç ve birleşik skor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .utils import fmt_pct, fmt_price

MIN_BARS = 60


@dataclass
class MarketAnalysis:
    symbol: str
    interval: str
    price: float
    change_pct: float
    trend: str
    adx: float
    regime: str
    volatility_pct: float
    rsi: float
    macd_hist: float
    bb_percent: float
    stoch_k: float
    support: float
    resistance: float
    volume_ratio: float
    score: float
    recommendation: str
    components: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            f"{self.symbol} ({self.interval})",
            f"Fiyat: {fmt_price(self.price)}   Değişim (son 24 mum): {fmt_pct(self.change_pct)}",
            "",
            f"GENEL SKOR: {self.score:+.0f} / 100  →  {self.recommendation}",
            "",
            f"Trend: {self.trend}   (ADX {self.adx:.1f})",
            f"Piyasa rejimi: {self.regime}",
            f"Volatilite (ATR/Fiyat): {self.volatility_pct:.2f}%",
            f"RSI(14): {self.rsi:.1f}   Stokastik %K: {self.stoch_k:.1f}",
            f"MACD histogram: {self.macd_hist:.6g}",
            f"Bollinger %B: {self.bb_percent:.2f}",
            f"Destek: {fmt_price(self.support)}   Direnç: {fmt_price(self.resistance)}",
            f"Hacim / ortalama: {self.volume_ratio:.2f}x",
            "",
            "Skor bileşenleri:",
        ]
        for name, val in self.components.items():
            lines.append(f"  • {name}: {val:+.2f}")
        if self.notes:
            lines += ["", "Yorumlar:"] + [f"  • {n}" for n in self.notes]
        lines += ["", "Not: Bu analiz yatırım tavsiyesi değildir."]
        return "\n".join(lines)


def recommendation_for(score: float) -> str:
    if score >= 50:
        return "GÜÇLÜ AL"
    if score >= 20:
        return "AL"
    if score <= -50:
        return "GÜÇLÜ SAT"
    if score <= -20:
        return "SAT"
    return "NÖTR"


def _num(value, default=0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(v) else v


def analyze(df: pd.DataFrame, symbol: str = "", interval: str = "") -> MarketAnalysis:
    if len(df) < MIN_BARS:
        raise ValueError(f"Analiz için en az {MIN_BARS} mum gerekli (mevcut: {len(df)}).")

    close = df["close"]
    price = float(close.iloc[-1])
    ema20, ema50 = ind.ema(close, 20), ind.ema(close, 50)
    ema200 = ind.ema(close, 200) if len(df) >= 200 else None
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    e200 = _num(ema200.iloc[-1], np.nan) if ema200 is not None else np.nan

    adx_s, plus_di, minus_di = ind.adx(df, 14)
    adx_val = _num(adx_s.iloc[-1])
    atr_s = ind.atr(df, 14)
    vol_pct_series = atr_s / close * 100
    vol_pct = _num(vol_pct_series.iloc[-1])
    vol_median = _num(vol_pct_series.tail(100).median(), vol_pct)
    rsi_val = _num(ind.rsi(close, 14).iloc[-1], 50)
    _, _, hist = ind.macd(close)
    hist_now, hist_prev = _num(hist.iloc[-1]), _num(hist.iloc[-2])
    pb = _num(ind.bollinger_percent_b(close).iloc[-1], 0.5)
    stoch_k = _num(ind.stochastic(df)[0].iloc[-1], 50)
    _, st_dir = ind.supertrend(df, 10, 3.0)
    st_now = int(st_dir.iloc[-1])

    lookback = min(50, len(df) - 1)
    window = df.iloc[-lookback - 1:-1]
    support, resistance = float(window["low"].min()), float(window["high"].max())

    avg_vol = _num(df["volume"].iloc[-21:-1].mean())
    volume_ratio = float(df["volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    n_change = min(24, len(df) - 1)
    change_pct = (price / float(close.iloc[-1 - n_change]) - 1) * 100

    # --- Trend
    if price > e50 and e20 > e50:
        trend = "Yükseliş"
    elif price < e50 and e20 < e50:
        trend = "Düşüş"
    else:
        trend = "Yatay / Kararsız"

    # --- Rejim
    if adx_val >= 25:
        regime = "Trend piyasası"
    elif vol_median > 0 and vol_pct > 1.8 * vol_median:
        regime = "Yüksek volatilite"
    else:
        regime = "Yatay piyasa (range)"

    # --- Skor bileşenleri (-1..1)
    trend_points = [1 if price > e20 else -1, 1 if e20 > e50 else -1]
    if not np.isnan(e200):
        trend_points.append(1 if e50 > e200 else -1)
    trend_score = float(np.mean(trend_points))

    macd_score = (0.6 if hist_now > 0 else -0.6) + (0.4 if hist_now > hist_prev else -0.4)

    rsi_score = float(np.clip((rsi_val - 50) / 25, -1, 1))
    if rsi_val > 75:
        rsi_score = -0.4
    elif rsi_val < 25:
        rsi_score = 0.4

    st_score = float(st_now)

    candle_dir = 1 if float(df["close"].iloc[-1]) >= float(df["open"].iloc[-1]) else -1
    volume_score = candle_dir * min(1.0, max(0.0, volume_ratio - 1.0)) if volume_ratio > 1.2 else 0.0

    components = {
        "Trend (EMA 20/50/200)": trend_score,
        "Momentum (MACD)": macd_score,
        "RSI": rsi_score,
        "Supertrend": st_score,
        "Hacim": volume_score,
    }
    weights = {
        "Trend (EMA 20/50/200)": 0.30,
        "Momentum (MACD)": 0.20,
        "RSI": 0.15,
        "Supertrend": 0.25,
        "Hacim": 0.10,
    }
    score = 100 * sum(components[k] * weights[k] for k in components)
    # Trend gücü düşükse skoru biraz zayıflat (yatay piyasada sinyaller güvenilmez).
    if adx_val < 20:
        score *= 0.75
    score = float(np.clip(score, -100, 100))

    notes = []
    if adx_val >= 40:
        notes.append("Çok güçlü trend (ADX ≥ 40). Trend takip stratejileri öne çıkar.")
    elif adx_val >= 25:
        notes.append("Belirgin trend var. EMA / Supertrend / MACD stratejileri uygundur.")
    else:
        notes.append("Trend zayıf. RSI ve Bollinger gibi dönüş stratejileri daha uygundur.")
    if rsi_val >= 70:
        notes.append("RSI aşırı alım bölgesinde: geri çekilme riski.")
    elif rsi_val <= 30:
        notes.append("RSI aşırı satım bölgesinde: tepki yükselişi olasılığı.")
    if pb > 1:
        notes.append("Fiyat Bollinger üst bandının üzerinde.")
    elif pb < 0:
        notes.append("Fiyat Bollinger alt bandının altında.")
    if volume_ratio >= 2:
        notes.append(f"Hacim ortalamanın {volume_ratio:.1f} katı: güçlü ilgi.")
    if vol_median > 0 and vol_pct > 1.8 * vol_median:
        notes.append("Volatilite normalin çok üzerinde: pozisyon boyutunu küçültün.")
    if resistance > 0 and (resistance - price) / price * 100 < 1:
        notes.append("Fiyat dirence çok yakın.")
    if support > 0 and (price - support) / price * 100 < 1:
        notes.append("Fiyat desteğe çok yakın.")
    if hist_now > 0 > hist_prev:
        notes.append("MACD histogramı pozitife döndü (al yönlü momentum).")
    elif hist_now < 0 < hist_prev:
        notes.append("MACD histogramı negatife döndü (sat yönlü momentum).")

    return MarketAnalysis(
        symbol=symbol, interval=interval, price=price, change_pct=change_pct,
        trend=trend, adx=adx_val, regime=regime, volatility_pct=vol_pct,
        rsi=rsi_val, macd_hist=hist_now, bb_percent=pb, stoch_k=stoch_k,
        support=support, resistance=resistance, volume_ratio=volume_ratio,
        score=score, recommendation=recommendation_for(score),
        components=components, notes=notes,
    )
