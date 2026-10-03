"""33 bağımsız strateji modülü (aday sinyal üreticileri).

Hiçbiri tek başına işlem açmaz. Parametreler başlangıç değerleridir; kârlı
oldukları varsayılmaz, walk-forward ile doğrulanmalıdır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..types import Regime
from .base import (
    NO_GO, R, TREND_LONG, IntelStrategy, StrategySpec, X, band_far, band_near, candle_dir, close_location, cross,
    edge, osc,
)

TREND_PREF = {R.STRONG_BULL, R.BULL}
TREND_FORBID = {R.SIDEWAYS, R.CHOP, R.LOW_VOLATILITY, R.STRONG_BEAR, R.BEAR}
MR_PREF = {R.SIDEWAYS, R.LOW_VOLATILITY, R.CHOP, R.WEAK_BULL}
MR_FORBID = {R.STRONG_BULL, R.STRONG_BEAR}  # Güçlü trendde ortalamaya dönüş kapalı
BO_PREF = {R.STRONG_BULL, R.BULL, R.WEAK_BULL, R.LOW_VOLATILITY, R.SIDEWAYS}
BO_FORBID = {R.CHOP, R.STRONG_BEAR}
VOLBO_PREF = {R.HIGH_VOLATILITY, R.LOW_VOLATILITY, R.STRONG_BULL, R.BULL}


def _spec(**kw) -> StrategySpec:
    return StrategySpec(**kw)


# ====================================================================== TREND
class TrendFollowing(IntelStrategy):
    spec = _spec(key="trend_following", name="Trend Takibi", family="ema", style="intraday",
                 preferred=TREND_PREF, forbidden=TREND_FORBID,
                 entry="EMA20>EMA50>EMA200, ADX güçlü, fiyat VWAP üstünde, HH/HL yapısı, hacim onayı",
                 confirmation="Kapanış onayı + hacim (RVOL)", invalidation="EMA20/50 ters kesişim veya Supertrend dönüşü",
                 exit="ATR trailing, EMA dönüşü, Supertrend dönüşü, yapı kırılımı, rejim değişimi, TP/SL, süre")
    default_params = {"adx": 22.0, "rvol": 1.0}

    def setup(self, f, p, s):
        c = (X(f["ema20"], f["ema50"], s) & X(f["ema50"], f["ema200"], s) & (f["adx"] > p["adx"])
             & X(f["close"], f["vwap"], s) & (f["structure_trend"] * s == 1) & (f["rvol"] > p["rvol"]))
        return edge(c)

    def invalidation(self, f, p, s):
        return cross(f["ema20"], f["ema50"], -s) | (f["st_dir"] * s == -1) & (f["st_dir"].shift() * s == 1)


class EmaTrend(IntelStrategy):
    spec = _spec(key="ema_trend", name="EMA Trend", family="ema", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR},
                 entry="EMA20, EMA50'yi kesiyor; fiyat EMA200 tarafında", invalidation="Ters EMA kesişimi")
    default_params = {"fast": "ema20", "slow": "ema50"}

    def setup(self, f, p, s):
        return cross(f[p["fast"]], f[p["slow"]], s) & X(f["close"], f["ema200"], s)


class EmaPullback(IntelStrategy):
    spec = _spec(key="ema_pullback", name="EMA Geri Çekilme", family="ema", style="intraday",
                 preferred=TREND_PREF, forbidden=TREND_FORBID,
                 entry="Trend içinde EMA20'ye geri çekilme ve yönlü mum kapanışı",
                 invalidation="Kapanış EMA50'nin ters tarafında")
    default_params = {"tol": 0.002}

    def setup(self, f, p, s):
        trend = X(f["ema20"], f["ema50"], s) & X(f["ema50"], f["ema200"], s)
        touch = (f["low"] <= f["ema20"] * (1 + p["tol"])) if s > 0 else (f["high"] >= f["ema20"] * (1 - p["tol"]))
        return trend & touch & X(f["close"], f["ema20"], s) & candle_dir(f, s)

    def invalidation(self, f, p, s):
        return X(f["ema50"], f["close"], s)


class MacdMomentumX(IntelStrategy):
    spec = _spec(key="macd_momentum", name="MACD Momentum", family="macd", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR},
                 entry="MACD histogramı sıfırı geçiyor, MACD>sinyal, fiyat EMA50 tarafında")
    default_params = {}

    def setup(self, f, p, s):
        return cross(f["macd_hist"], 0, s) & X(f["macd"], f["macd_signal"], s) & X(f["close"], f["ema50"], s)


class RsiMomentum(IntelStrategy):
    spec = _spec(key="rsi_momentum", name="RSI Momentum", family="momentum", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR},
                 entry="RSI 55'i yukarı kesiyor, EMA20>EMA50")
    default_params = {"level": 55.0}

    def setup(self, f, p, s):
        return cross(osc(f, "rsi", s), p["level"], 1) & X(f["ema20"], f["ema50"], s)

    def strength(self, f, p, s):
        return ((osc(f, "rsi", s) - 50) / 30).clip(0, 1).fillna(0)


class Momentum(IntelStrategy):
    spec = _spec(key="momentum", name="Momentum", family="momentum", style="scalp",
                 preferred=TREND_PREF | {R.HIGH_VOLATILITY}, forbidden={R.CHOP, R.SIDEWAYS, R.STRONG_BEAR},
                 entry="ROC eşiği aşıyor, fiyat ivmeleniyor, hacim ivmeleniyor",
                 exit="Momentum zayıflarsa azalt/çık (REDUCE / EXIT CANDIDATE)")
    default_params = {"roc": 1.0, "rvol": 1.2}

    def setup(self, f, p, s):
        c = (f["roc"] * s > p["roc"]) & (f["price_accel"] * s > 0) & (f["vol_accel"] > 0) & (f["rvol"] > p["rvol"])
        return edge(c)

    def invalidation(self, f, p, s):
        return (f["price_accel"] * s < 0) & (f["macd_hist"] * s < f["macd_hist"].shift() * s)


# ====================================================================== ORTALAMAYA DÖNÜŞ
class MeanReversion(IntelStrategy):
    spec = _spec(key="mean_reversion", name="Ortalamaya Dönüş", family="mean_reversion", style="intraday",
                 preferred=MR_PREF, forbidden=MR_FORBID | {R.BEAR},
                 entry="Bollinger dışından dönüş + RSI uç + Z-score uç + hacim tükenmesi",
                 invalidation="Yeni uç (fiyat bandın daha da dışına)", exit="Orta bant / VWAP hedefi")
    default_params = {"z": 2.0, "rsi_ext": 35.0}

    def setup(self, f, p, s):
        prev_ext = (f["zscore"].shift() * s < -p["z"]) & (osc(f, "rsi", s).shift() < p["rsi_ext"])
        exhaustion = f["rvol"] < f["rvol"].shift()
        return prev_ext & cross(f["close"], band_far(f, s), s) & exhaustion

    def invalidation(self, f, p, s):
        return (f["zscore"] * s < -(p["z"] + 1.0))


class RsiMeanReversion(IntelStrategy):
    spec = _spec(key="rsi_mean_reversion", name="RSI Ortalamaya Dönüş", family="mean_reversion",
                 style="intraday", preferred=MR_PREF, forbidden=MR_FORBID | {R.BEAR},
                 entry="RSI aşırı satımdan (30) yukarı dönüyor")
    default_params = {"oversold": 30.0}

    def setup(self, f, p, s):
        return cross(osc(f, "rsi", s), p["oversold"], 1)


class BollingerMeanReversion(IntelStrategy):
    spec = _spec(key="bollinger_mean_reversion", name="Bollinger Ortalamaya Dönüş", family="mean_reversion",
                 style="intraday", preferred=MR_PREF, forbidden=MR_FORBID | {R.BEAR},
                 entry="Kapanış alt banttan içeri dönüyor", exit="Orta bant")
    default_params = {}

    def setup(self, f, p, s):
        return cross(f["close"], band_far(f, s), s)


class VwapMeanReversion(IntelStrategy):
    spec = _spec(key="vwap_mean_reversion", name="VWAP Ortalamaya Dönüş", family="vwap", style="scalp",
                 preferred=MR_PREF, forbidden=MR_FORBID | {R.BEAR},
                 entry="Fiyat VWAP'tan 2 ATR uzaklaşıp geri dönüyor", exit="VWAP")
    default_params = {"dev": 2.0}

    def setup(self, f, p, s):
        dev = (f["close"] - f["vwap"]) / f["atr"].replace(0, np.nan) * s
        return cross(dev, -p["dev"], 1)


class VwapReclaim(IntelStrategy):
    spec = _spec(key="vwap_reclaim", name="VWAP Geri Alma", family="vwap", style="scalp",
                 preferred={R.BULL, R.WEAK_BULL, R.SIDEWAYS}, forbidden={R.STRONG_BEAR, R.CHOP},
                 entry="N mum VWAP altında kaldıktan sonra hacimle VWAP üstüne kapanış",
                 invalidation="Tekrar VWAP altına kapanış (VWAP rejection)")
    default_params = {"bars_below": 5, "rvol": 1.2}

    def setup(self, f, p, s):
        below = X(f["vwap"], f["close"], s).astype(float).shift().rolling(int(p["bars_below"])).sum() \
            == int(p["bars_below"])
        return cross(f["close"], f["vwap"], s) & below & (f["rvol"] > p["rvol"])

    def invalidation(self, f, p, s):
        return cross(f["close"], f["vwap"], -s)


# ====================================================================== TREND GÜCÜ
class AdxTrend(IntelStrategy):
    spec = _spec(key="adx_trend", name="ADX Trend", family="adx", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR},
                 entry="ADX 25'i yukarı kesiyor ve DI+ > DI-")
    default_params = {"level": 25.0}

    def setup(self, f, p, s):
        return cross(f["adx"], p["level"], 1) & X(f["di_plus"], f["di_minus"], s)


class SuperTrendX(IntelStrategy):
    spec = _spec(key="supertrend", name="SuperTrend", family="supertrend", style="intraday",
                 preferred=TREND_PREF | {R.HIGH_VOLATILITY}, forbidden={R.CHOP, R.SIDEWAYS, R.STRONG_BEAR},
                 stop_method="supertrend", entry="Supertrend yönü dönüyor")
    default_params = {}

    def setup(self, f, p, s):
        return (f["st_dir"] * s == 1) & (f["st_dir"].shift() * s == -1)


# ====================================================================== KIRILIM
class DonchianBreakoutX(IntelStrategy):
    spec = _spec(key="donchian_breakout", name="Donchian Kırılımı", family="breakout", style="swing",
                 preferred=BO_PREF, forbidden=BO_FORBID, entry="Kapanış önceki 20 mumun tepesini aşıyor")
    default_params = {}

    def setup(self, f, p, s):
        return edge(f["breakout_up"] if s > 0 else f["breakdown"])


class BollingerSqueezeBreakout(IntelStrategy):
    spec = _spec(key="bb_squeeze_breakout", name="Bollinger Sıkışma Kırılımı", family="volatility",
                 style="intraday", preferred=VOLBO_PREF | {R.SIDEWAYS}, forbidden={R.CHOP, R.STRONG_BEAR},
                 entry="Sıkışma (BB Keltner içinde / bant genişliği düşük) sonrası hacimli bant kırılımı")
    default_params = {"width_pct": 0.2, "rvol": 1.5}

    def setup(self, f, p, s):
        squeezed = (f["squeeze"] | (f["bb_width_pct"] < p["width_pct"])).astype(float).shift() \
            .rolling(5, min_periods=1).max() > 0
        return squeezed & cross(f["close"], band_near(f, s), s) & (f["rvol"] > p["rvol"])


class VolumeBreakout(IntelStrategy):
    spec = _spec(key="volume_breakout", name="Hacim Kırılımı", family="breakout", style="intraday",
                 preferred=BO_PREF | {R.HIGH_VOLATILITY}, forbidden=BO_FORBID,
                 entry="Kapanış 20 mumluk kapanış zirvesini 2x hacimle aşıyor")
    default_params = {"rvol": 2.0, "n": 20}

    def setup(self, f, p, s):
        n = int(p["n"])
        ext = f["close"].shift().rolling(n).max() if s > 0 else f["close"].shift().rolling(n).min()
        return X(f["close"], ext, s) & (f["rvol"] > p["rvol"])


class Breakout(IntelStrategy):
    spec = _spec(key="breakout", name="Aralık Kırılımı", family="breakout", style="intraday",
                 preferred=BO_PREF, forbidden=BO_FORBID,
                 entry="Konsolidasyon sonrası kapanışla kırılım, hacim genişlemesi",
                 confirmation="Sahte kırılım filtresi: kapanış mumun güçlü ucunda (≥%70), gövde büyük",
                 invalidation="Kapanış aralığın içine geri dönüyor")
    default_params = {"rvol": 1.3, "close_loc": 0.7}

    def setup(self, f, p, s):
        level = f["range_high"] if s > 0 else f["range_low"]
        return (f["consolidation"].shift(fill_value=False) & X(f["close"], level, s) & (f["rvol"] > p["rvol"])
                & (close_location(f, s) >= p["close_loc"]))

    def invalidation(self, f, p, s):
        level = f["range_high"] if s > 0 else f["range_low"]
        return X(level, f["close"], s)


class BreakoutRetest(IntelStrategy):
    spec = _spec(key="breakout_retest", name="Kırılım Sonrası Tekrar Test", family="breakout",
                 style="intraday", preferred=BO_PREF, forbidden=BO_FORBID,
                 entry="Aralık → kırılım → seviyeye geri test → seviye tutuyor → yönlü kapanış",
                 invalidation="Kapanış seviyenin 0.5 ATR ötesine geri dönüyor")
    default_params = {"window": 12, "tol_atr": 0.3, "fail_atr": 0.5, "rvol": 1.2}

    def setup(self, f, p, s):
        level = (f["range_high"] if s > 0 else f["range_low"]).to_numpy(dtype=float)
        close = f["close"].to_numpy(dtype=float)
        lo = f["low"].to_numpy(dtype=float) if s > 0 else f["high"].to_numpy(dtype=float)
        atr = f["atr"].to_numpy(dtype=float)
        rvol = f["rvol"].to_numpy(dtype=float)
        out = np.zeros(len(f), dtype=bool)
        active_level, since = np.nan, 0
        for i in range(1, len(f)):
            if not np.isnan(active_level):
                since += 1
                if since > p["window"] or (active_level - close[i]) * s > p["fail_atr"] * atr[i]:
                    active_level = np.nan
                elif ((lo[i] - active_level) * s <= p["tol_atr"] * atr[i]) and (close[i] - active_level) * s > 0 \
                        and since >= 2:
                    out[i] = True
                    active_level = np.nan
                    continue
            if not np.isnan(level[i]) and (close[i] - level[i]) * s > 0 and (close[i - 1] - level[i]) * s <= 0 \
                    and rvol[i] > p["rvol"]:
                active_level, since = level[i], 0
        return pd.Series(out, index=f.index)


class SupportResistanceBreakout(IntelStrategy):
    spec = _spec(key="sr_breakout", name="Destek/Direnç Kırılımı", family="breakout", style="intraday",
                 preferred=BO_PREF, forbidden=BO_FORBID,
                 entry="Önceki gün yükseği (direnç) hacimle kırılıyor")
    default_params = {"rvol": 1.2}

    def setup(self, f, p, s):
        level = f["prev_day_high"] if s > 0 else f["prev_day_low"]
        return cross(f["close"], level, s) & (f["rvol"] > p["rvol"])


class AtrVolatilityBreakout(IntelStrategy):
    spec = _spec(key="atr_volatility_breakout", name="ATR Volatilite Kırılımı", family="volatility",
                 style="scalp", preferred=VOLBO_PREF, forbidden={R.CHOP, R.STRONG_BEAR},
                 entry="Mum, önceki kapanıştan 1.5 ATR'den fazla yönlü hareket ediyor + hacim",
                 risk_multiplier=0.75)
    default_params = {"k": 1.5, "rvol": 1.5}

    def setup(self, f, p, s):
        move = (f["close"] - f["close"].shift()) * s
        return (move > p["k"] * f["atr"].shift()) & (f["rvol"] > p["rvol"])


# ====================================================================== PİYASA YAPISI
class MarketStructureBOS(IntelStrategy):
    spec = _spec(key="structure_bos", name="Piyasa Yapısı BOS", family="structure", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR}, stop_method="swing",
                 entry="Trend yönünde yapı kırılımı (Break of Structure) + hacim")
    default_params = {"rvol": 1.0}

    def setup(self, f, p, s):
        return (f["bos_bull"] if s > 0 else f["bos_bear"]) & (f["rvol"] > p["rvol"])


class ChochReversal(IntelStrategy):
    spec = _spec(key="choch_reversal", name="CHoCH Dönüşü", family="structure", style="intraday",
                 preferred={R.SIDEWAYS, R.WEAK_BULL, R.BEAR, R.LOW_VOLATILITY}, forbidden={R.STRONG_BEAR},
                 stop_method="swing", entry="Karakter değişimi (CHoCH) + EMA20 geri alındı")
    default_params = {}

    def setup(self, f, p, s):
        return (f["choch_bull"] if s > 0 else f["choch_bear"]) & X(f["close"], f["ema20"], s)


class LiquiditySweep(IntelStrategy):
    spec = _spec(key="liquidity_sweep", name="Likidite Süpürmesi", family="structure", style="scalp",
                 preferred={R.SIDEWAYS, R.BULL, R.WEAK_BULL, R.LOW_VOLATILITY}, forbidden={R.STRONG_BEAR},
                 stop_method="swing", entry="Önceki swing dibinin altına fitil, kapanış geri içeride + hacim")
    default_params = {"rvol": 1.2}

    def setup(self, f, p, s):
        return (f["sweep_bull"] if s > 0 else f["sweep_bear"]) & (f["rvol"] > p["rvol"]) & candle_dir(f, s)


# ====================================================================== ORDER FLOW
class OrderFlowMomentum(IntelStrategy):
    spec = _spec(key="orderflow_momentum", name="Order Flow Momentum", family="orderflow", style="scalp",
                 preferred=TREND_PREF | {R.WEAK_BULL, R.HIGH_VOLATILITY}, forbidden={R.CHOP, R.STRONG_BEAR},
                 requires=("delta",),
                 entry="Son 3 mumda pozitif delta, CVD yükseliyor, taker alış oranı > %55, fiyat yükseliyor")
    default_params = {"taker": 0.55}

    def setup(self, f, p, s):
        delta_ok = ((f["delta"] * s) > 0).astype(float).rolling(3).sum() == 3
        taker = f["taker_buy_ratio"] if s > 0 else 1 - f["taker_buy_ratio"]
        c = delta_ok & (f["cvd_slope"] * s > 0) & (taker > p["taker"]) & ((f["close"] - f["close"].shift(3)) * s > 0)
        return edge(c)


class CvdDivergence(IntelStrategy):
    spec = _spec(key="cvd_divergence", name="CVD Uyumsuzluğu", family="orderflow", style="intraday",
                 preferred=MR_PREF | {R.BULL}, forbidden={R.STRONG_BEAR}, requires=("cvd",),
                 entry="Fiyat yeni 20 mum dibi yaparken CVD önceki dibin üstünde + yönlü kapanış")
    default_params = {"n": 20}

    def setup(self, f, p, s):
        n = int(p["n"])
        price_ext = (f["close"] < f["close"].shift().rolling(n).min()) if s > 0 else \
            (f["close"] > f["close"].shift().rolling(n).max())
        cvd_ref = f["cvd"].shift().rolling(n).min() if s > 0 else f["cvd"].shift().rolling(n).max()
        div = price_ext & X(f["cvd"], cvd_ref, s)
        return div.shift(fill_value=False) & candle_dir(f, s)


# ====================================================================== TÜREVLER (FUTURES)
class OpenInterestMomentum(IntelStrategy):
    spec = _spec(key="oi_momentum", name="Open Interest Momentum", family="derivatives", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR},
                 requires=("open_interest",),
                 entry="Fiyat↑ + OI↑ + hacim↑ (+ CVD↑ varsa): momentum onayı adayı")
    default_params = {"oi": 1.0, "rvol": 1.0}

    def setup(self, f, p, s):
        c = (f["roc"] * s > 0) & (f["oi_change_pct"] > p["oi"]) & (f["rvol"] > p["rvol"])
        if f["cvd_slope"].notna().any():
            c &= f["cvd_slope"] * s > 0
        return edge(c)


class FundingCrowding(IntelStrategy):
    spec = _spec(key="funding_crowding", name="Funding / Kalabalık Pozisyon", family="derivatives",
                 style="intraday", preferred={R.SIDEWAYS, R.WEAK_BULL, R.BEAR, R.BULL}, forbidden={R.STRONG_BEAR},
                 requires=("funding_rate",), risk_multiplier=0.75,
                 entry="Funding yüksek negatif (short kalabalığı) + fiyat EMA20'yi geri alıyor (+ OI düşüyorsa)",
                 confirmation="Negatif funding tek başına LONG değildir; fiyat onayı şart")
    default_params = {"funding": 0.0003}

    def setup(self, f, p, s):
        crowded = (f["funding_rate"] * s <= -p["funding"])
        c = crowded & cross(f["close"], f["ema20"], s)
        if f["oi_change_pct"].notna().any():
            c &= f["oi_change_pct"].fillna(0) <= 0
        return c


class LiquidationReversal(IntelStrategy):
    spec = _spec(key="liquidation_reversal", name="Tasfiye Dönüşü", family="derivatives", style="scalp",
                 preferred={R.PANIC, R.HIGH_VOLATILITY, R.BEAR, R.SIDEWAYS}, forbidden=set(),
                 requires=("liq_long", "liq_short"), risk_multiplier=0.5,
                 entry="Long tasfiye patlaması + fiyat stabilizasyonu + CVD toparlanması + destek geri alındı")
    default_params = {"z": 4.0}

    def setup(self, f, p, s):
        liq = f["liq_long"] if s > 0 else f["liq_short"]
        z = (liq - liq.shift().rolling(50, min_periods=20).mean()) / liq.shift().rolling(50, min_periods=20).std()
        spike = (z > p["z"]).astype(float).shift().rolling(3, min_periods=1).max() > 0
        stabil = candle_dir(f, s) & X(f["low"] if s > 0 else f["high"], f["low"].shift() if s > 0 else f["high"].shift(), s)
        reclaim = X(f["close"], f["support"] if s > 0 else f["resistance"], s)
        c = spike & stabil & reclaim
        if f["cvd_slope"].notna().any():
            c &= f["cvd_slope"].diff() * s > 0
        return c


# ====================================================================== İSTATİSTİK
class ZScoreMeanReversion(IntelStrategy):
    spec = _spec(key="zscore_mean_reversion", name="Z-Score Ortalamaya Dönüş", family="statistical",
                 style="intraday", preferred=MR_PREF, forbidden=MR_FORBID | {R.BEAR},
                 entry="50 mumluk z-score -2'yi yukarı kesiyor")
    default_params = {"z": 2.0}

    def setup(self, f, p, s):
        return cross(f["zscore"] * s, -p["z"], 1)


class StatisticalMeanReversion(IntelStrategy):
    spec = _spec(key="statistical_mean_reversion", name="İstatistiksel Ortalamaya Dönüş", family="statistical",
                 style="scalp", preferred=MR_PREF, forbidden=MR_FORBID | {R.BEAR},
                 entry="Volatiliteye göre düzeltilmiş 5 mumluk getiri -2.5σ altında ve dönüş başladı, ADX<25")
    default_params = {"th": 2.5, "adx": 25.0}

    def setup(self, f, p, s):
        return ((f["vol_adj_ret"].shift() * s) < -p["th"]) & (f["norm_ret"] * s > 0) & (f["adx"] < p["adx"])


# ====================================================================== META
class MultiTimeframeConfluence(IntelStrategy):
    spec = _spec(key="mtf_confluence", name="Çoklu Zaman Dilimi Uyumu", family="meta", style="intraday",
                 preferred=TREND_PREF | {R.WEAK_BULL}, forbidden={R.CHOP, R.STRONG_BEAR},
                 requires=("major_bias", "trend_bias"),
                 entry="Ana+trend+onay zaman dilimleri aynı yönde, giriş TF'de EMA9/20 kesişimi")
    default_params = {"th": 0.5}

    def setup(self, f, p, s):
        parts = [f[c] for c in ("major_bias", "trend_bias", "confirmation_bias") if c in f]
        agg = pd.concat(parts, axis=1).mean(axis=1) * s
        return (agg > p["th"]) & cross(f["ema9"], f["ema20"], s)


class RegimeAdaptive(IntelStrategy):
    spec = _spec(key="regime_adaptive", name="Rejime Uyarlanan", family="meta", style="intraday",
                 preferred={R.STRONG_BULL, R.BULL, R.WEAK_BULL, R.SIDEWAYS, R.LOW_VOLATILITY},
                 forbidden={R.CHOP, R.STRONG_BEAR}, requires=("regime",),
                 entry="Trend rejiminde Trend Takibi, yatay rejimde Bollinger dönüşü kurulumunu kullanır")
    default_params = {}

    def setup(self, f, p, s):
        trend_regimes = {r.value for r in (TREND_LONG if s > 0 else {R.STRONG_BEAR, R.BEAR})}
        range_regimes = {R.SIDEWAYS.value, R.LOW_VOLATILITY.value}
        tf = TrendFollowing().setup(f, TrendFollowing.default_params, s)
        mr = BollingerMeanReversion().setup(f, {}, s)
        reg = f["regime"].astype(str)
        return (tf & reg.isin(trend_regimes)) | (mr & reg.isin(range_regimes))


class AIEnsemble(IntelStrategy):
    spec = _spec(key="ai_ensemble", name="AI/ML Topluluk", family="meta", style="intraday",
                 preferred={R.STRONG_BULL, R.BULL, R.WEAK_BULL, R.SIDEWAYS, R.LOW_VOLATILITY, R.HIGH_VOLATILITY},
                 forbidden={R.STRONG_BEAR}, requires=("ml_prob_long", "ml_prob_short"),
                 entry="Eğitilmiş modelin yön olasılığı eşiği aşıyor (model yoksa çalışmaz)")
    default_params = {"th": 0.6}

    def setup(self, f, p, s):
        col = "ml_prob_long" if s > 0 else "ml_prob_short"
        return cross(f[col], p["th"], 1)


ALL_STRATEGIES: list[type[IntelStrategy]] = [
    TrendFollowing, EmaTrend, EmaPullback, MacdMomentumX, RsiMomentum, Momentum, MeanReversion, RsiMeanReversion,
    BollingerMeanReversion, VwapMeanReversion, VwapReclaim, AdxTrend, SuperTrendX, DonchianBreakoutX,
    BollingerSqueezeBreakout, VolumeBreakout, Breakout, BreakoutRetest, SupportResistanceBreakout,
    MarketStructureBOS, ChochReversal, LiquiditySweep, OrderFlowMomentum, CvdDivergence, OpenInterestMomentum,
    FundingCrowding, LiquidationReversal, AtrVolatilityBreakout, ZScoreMeanReversion, StatisticalMeanReversion,
    MultiTimeframeConfluence, RegimeAdaptive, AIEnsemble,
]

REGISTRY: dict[str, type[IntelStrategy]] = {cls.spec.key: cls for cls in ALL_STRATEGIES}
assert len(REGISTRY) == 33, "Strateji anahtarları benzersiz olmalı"

_ = (NO_GO, Regime)  # dışa açık semboller
