"""Özellik (feature) motoru.

`compute_features` tek bir zaman dilimindeki mumlardan tüm teknik göstergeleri,
piyasa yapısını, mum formasyonlarını, hacim/akış ve istatistik özelliklerini
nedensel olarak hesaplar. Futures türevleri (funding, OI, long/short, taker)
ayrı sütunlarda ve kaynak etiketleriyle birleştirilir; yoksa sütun NaN kalır ve
`attrs["availability"]` içinde nedeni belirtilir. Sahte değer üretilmez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..utils import INTERVALS
from .structure import candlestick_patterns, market_structure

FEATURE_GROUPS = {
    "trend": ["ema9", "ema20", "ema50", "ema100", "ema200", "sma20", "sma50", "sma100", "sma200", "vwap",
              "avwap", "adx", "di_plus", "di_minus", "supertrend", "st_dir", "psar", "ichi_tenkan",
              "ichi_kijun", "ichi_span_a", "ichi_span_b", "ema50_slope", "ema200_slope"],
    "momentum": ["rsi", "stochrsi_k", "stochrsi_d", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist",
                 "roc", "mom", "cci", "willr", "mfi", "price_accel"],
    "volatility": ["atr", "atr_pct", "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_width_pct", "kc_upper",
                   "kc_lower", "hv", "rv", "vol_percentile", "squeeze"],
    "volume": ["vol_sma20", "rvol", "volume_spike", "vol_accel", "obv", "obv_slope"],
    "orderflow": ["delta", "cvd", "cvd_slope", "taker_buy_ratio"],
    "statistical": ["zscore", "ret_z", "norm_ret", "vol_adj_ret"],
    "derivatives": ["funding_rate", "open_interest", "oi_change_pct", "oi_accel", "long_short_ratio",
                    "fut_taker_ratio", "liq_long", "liq_short"],
}

SOURCES = {
    "spot": "binance_spot_klines",
    "futures": "binance_futures_klines",
    "funding_rate": "binance_futures_fundingRate",
    "open_interest": "binance_futures_openInterestHist",
    "long_short_ratio": "binance_futures_globalLongShortAccountRatio",
    "fut_taker_ratio": "binance_futures_takerlongshortRatio",
    "liquidations": "unavailable_rest",
}


def bars_per_year(interval: str) -> float:
    return 365 * 24 * 3600 / INTERVALS.get(interval, 3600)


def compute_features(df: pd.DataFrame, interval: str = "1h", market: str = "SPOT",
                     derivatives: dict | None = None, swing_k: int = 3) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DATA_UNAVAILABLE: mum verisi yok")
    df = df.reset_index(drop=True)
    f = df[[c for c in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume",
                        "trades", "taker_buy_base", "taker_buy_quote") if c in df]].copy()
    close = df["close"]
    a = ind.atr(df, 14)

    # ---------------- trend
    for n in (9, 20, 50, 100, 200):
        f[f"ema{n}"] = ind.ema(close, n)
    for n in (20, 50, 100, 200):
        f[f"sma{n}"] = ind.sma(close, n)
    f["vwap"] = ind.vwap_session(df)
    f["vwap_roll"] = ind.vwap_rolling(df, 50)
    adx, dip, dim = ind.adx(df, 14)
    f["adx"], f["di_plus"], f["di_minus"] = adx, dip, dim
    st, st_dir = ind.supertrend(df, 10, 3.0)
    f["supertrend"], f["st_dir"] = st, st_dir
    f["psar"] = ind.parabolic_sar(df)
    f["ichi_tenkan"], f["ichi_kijun"], f["ichi_span_a"], f["ichi_span_b"] = ind.ichimoku(df)
    f["ichi_chikou_ok"] = close > close.shift(26)
    f["ema50_slope"] = (f["ema50"] - f["ema50"].shift(10)) / a.replace(0, np.nan)
    f["ema200_slope"] = (f["ema200"] - f["ema200"].shift(20)) / a.replace(0, np.nan)
    f["ema20_slope"] = (f["ema20"] - f["ema20"].shift(5)) / a.replace(0, np.nan)

    # ---------------- momentum
    f["rsi"] = ind.rsi(close, 14)
    f["stochrsi_k"], f["stochrsi_d"] = ind.stoch_rsi(close)
    f["stoch_k"], f["stoch_d"] = ind.stochastic(df)
    f["macd"], f["macd_signal"], f["macd_hist"] = ind.macd(close)
    f["roc"] = ind.roc(close, 10)
    f["mom"] = ind.momentum(close, 10)
    f["cci"] = ind.cci(df, 20)
    f["willr"] = ind.williams_r(df, 14)
    f["mfi"] = ind.mfi(df, 14)
    f["price_accel"] = f["roc"].diff(3)

    # ---------------- volatilite
    f["atr"] = a
    f["atr_pct"] = a / close * 100
    f["bb_mid"], f["bb_upper"], f["bb_lower"] = ind.bollinger(close, 20, 2.0)
    f["bb_pctb"] = ind.bollinger_percent_b(close)
    f["bb_width"] = ind.bollinger_width(close)
    f["bb_width_pct"] = ind.rolling_percentile(f["bb_width"], 250)
    _, f["kc_upper"], f["kc_lower"] = ind.keltner(df)
    f["squeeze"] = (f["bb_upper"] < f["kc_upper"]) & (f["bb_lower"] > f["kc_lower"])
    bpy = bars_per_year(interval)
    f["hv"] = ind.historical_volatility(close, 30, bpy)
    f["rv"] = ind.realized_volatility(close, 30)
    f["vol_percentile"] = ind.rolling_percentile(f["atr_pct"], 250)
    f["rv_percentile"] = ind.rolling_percentile(f["rv"], 250)
    f["bar_range_atr"] = (df["high"] - df["low"]) / a.shift().replace(0, np.nan)

    # ---------------- hacim
    f["vol_sma20"] = ind.sma(df["volume"], 20)
    f["rvol"] = ind.relative_volume(df["volume"], 20)
    f["volume_spike"] = f["rvol"] > 2.0
    f["vol_accel"] = f["rvol"].rolling(3, min_periods=3).mean() - f["rvol"].shift(3).rolling(3, min_periods=3).mean()
    f["obv"] = ind.obv(df)
    f["obv_slope"] = np.sign(f["obv"] - f["obv"].shift(10))
    f["volume_z"] = ind.zscore(df["volume"], 50)

    # ---------------- order flow (kline taker verisi; yoksa NaN)
    f["delta"] = ind.volume_delta(df)
    f["cvd"] = ind.cvd(df)
    f["cvd_slope"] = f["cvd"] - f["cvd"].shift(10)
    f["taker_buy_ratio"] = (df["taker_buy_base"] / df["volume"].replace(0, np.nan)) if "taker_buy_base" in df \
        else np.nan

    # ---------------- istatistik
    ret = close.pct_change()
    f["zscore"] = ind.zscore(close, 50)
    std = ret.rolling(50, min_periods=50).std()
    f["ret_z"] = (ret - ret.rolling(50, min_periods=50).mean()) / std.replace(0, np.nan)
    f["norm_ret"] = ret / std.replace(0, np.nan)
    f["vol_adj_ret"] = (close / close.shift(5) - 1) / (std * np.sqrt(5)).replace(0, np.nan)

    # ---------------- yapı ve formasyonlar
    ms = market_structure(df, k=swing_k, atr_series=a)
    f = f.copy().join(ms)
    f["avwap"] = ind.anchored_vwap(df, ms["bos_bull"] | ms["bos_bear"])
    f = f.join(candlestick_patterns(df)).copy()

    # ---------------- futures türevleri (ayrı kaynaklar)
    availability = {
        "spot_klines" if market == "SPOT" else "futures_klines": "OK",
        "orderflow_taker": "OK" if "taker_buy_base" in df else "UNAVAILABLE: kline taker verisi yok",
    }
    f = pd.concat([f, pd.DataFrame(np.nan, index=f.index, columns=FEATURE_GROUPS["derivatives"])], axis=1)
    if derivatives:
        f, deriv_avail = merge_derivatives(f, derivatives)
        availability.update(deriv_avail)
    else:
        availability.update({k: "UNAVAILABLE: futures verisi sağlanmadı" for k in
                             ("funding_rate", "open_interest", "long_short_ratio", "fut_taker_ratio")})
    availability["liquidations"] = "UNAVAILABLE: Binance REST tasfiye verisi sunmuyor"
    f.attrs["availability"] = availability
    f.attrs["interval"] = interval
    f.attrs["market"] = market
    return f


def _asof(f: pd.DataFrame, data: pd.DataFrame, time_col: str, cols: list[str], delay=None) -> pd.DataFrame:
    if data is None or data.empty:
        return f
    d = data[[time_col] + cols].dropna(subset=[time_col]).copy()
    d["_avail_time"] = (pd.to_datetime(d[time_col], utc=True) + (delay or pd.Timedelta(0))).dt.as_unit("ns")
    d = d.sort_values("_avail_time")
    left = f[["close_time"]].copy()
    left["_row"] = np.arange(len(f))
    left["close_time"] = pd.to_datetime(left["close_time"], utc=True).dt.as_unit("ns")
    merged = pd.merge_asof(left.sort_values("close_time"), d.drop(columns=[time_col]),
                           left_on="close_time", right_on="_avail_time", direction="backward")
    merged = merged.sort_values("_row")
    for c in cols:
        f[c] = merged[c].to_numpy()
    return f


def merge_derivatives(f: pd.DataFrame, deriv: dict) -> tuple[pd.DataFrame, dict]:
    """Futures verilerini yalnızca mum kapanışında bilinir oldukları haliyle birleştirir.

    deriv anahtarları: funding (fundingTime, fundingRate), oi (timestamp, open_interest),
    long_short (timestamp, long_short_ratio), taker (timestamp, taker_buy_sell_ratio), period (str)
    """
    avail = {}
    period = deriv.get("period", "5m")
    delay = pd.Timedelta(seconds=INTERVALS.get(period, 300))
    fund = deriv.get("funding")
    if fund is not None and not fund.empty:
        f = _asof(f, fund.rename(columns={"fundingRate": "funding_rate"}), "fundingTime", ["funding_rate"])
        avail["funding_rate"] = "OK"
    else:
        avail["funding_rate"] = "UNAVAILABLE"
    oi = deriv.get("oi")
    if oi is not None and not oi.empty:
        f = _asof(f, oi, "timestamp", ["open_interest"], delay)
        f["oi_change_pct"] = f["open_interest"].pct_change(10, fill_method=None) * 100
        f["oi_accel"] = f["oi_change_pct"].diff(5)
        avail["open_interest"] = "OK" if f["open_interest"].notna().any() else \
            "UNAVAILABLE: OI geçmişi bu dönemi kapsamıyor (Binance ~30 gün sağlar)"
    else:
        avail["open_interest"] = "UNAVAILABLE"
    ls = deriv.get("long_short")
    if ls is not None and not ls.empty:
        f = _asof(f, ls, "timestamp", ["long_short_ratio"], delay)
        avail["long_short_ratio"] = "OK" if f["long_short_ratio"].notna().any() else "UNAVAILABLE"
    else:
        avail["long_short_ratio"] = "UNAVAILABLE"
    tk = deriv.get("taker")
    if tk is not None and not tk.empty:
        f = _asof(f, tk.rename(columns={"taker_buy_sell_ratio": "fut_taker_ratio"}), "timestamp",
                  ["fut_taker_ratio"], delay)
        avail["fut_taker_ratio"] = "OK" if f["fut_taker_ratio"].notna().any() else "UNAVAILABLE"
    else:
        avail["fut_taker_ratio"] = "UNAVAILABLE"
    liq = deriv.get("liquidations")
    if liq is not None and not liq.empty:
        f = _asof(f, liq, "timestamp", ["liq_long", "liq_short"])
        avail["liquidations"] = "OK"
    return f, avail


def feature_available(f: pd.DataFrame, col: str, i: int = -1) -> bool:
    if col not in f:
        return False
    v = f[col].iloc[i]
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def htf_bias(f: pd.DataFrame) -> pd.Series:
    """Bir zaman dilimi için -1..1 yön eğilimi (EMA dizilimi, eğim, Supertrend, yapı, MACD)."""
    parts = [
        np.sign(f["close"] - f["ema50"]),
        np.sign(f["ema20"] - f["ema50"]),
        np.sign(f["ema50"] - f["ema200"]).where(f["ema200"].notna(), 0),
        np.sign(f["ema50_slope"]),
        f["st_dir"].replace(0, np.nan),
        pd.Series(f["structure_trend"], index=f.index).astype(float),
        np.sign(f["macd_hist"]),
    ]
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1, skipna=True).where(f["ema50"].notna())
