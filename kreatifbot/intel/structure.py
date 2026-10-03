"""Piyasa yapısı ve mum formasyonları.

İleriye bakma kuralı: k-mumluk fraktal swing noktası (i) ancak i+k mumu kapandığında
bilinir. Bu yüzden swing bilgisi t anında yalnızca t-k ve öncesi için kullanılır.

SMC/ICT kavramları (BOS, CHoCH, order block, FVG, likidite süpürmesi) burada yalnızca
tanımlanır; doğru oldukları varsayılmaz. Katkıları backtest/ablasyonla ölçülmelidir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind


def swing_points(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Onaylanmış swing noktaları. Sütunlar t anında bilinen bilgiyi içerir.

    swing_high_confirmed[t] = True ise t-k mumundaki yüksek bir swing high'dır.
    """
    win = 2 * k + 1
    hi_c = df["high"].shift(k)
    lo_c = df["low"].shift(k)
    is_sh = (df["high"].rolling(win, min_periods=win).max() == hi_c) & hi_c.notna()
    is_sl = (df["low"].rolling(win, min_periods=win).min() == lo_c) & lo_c.notna()
    out = pd.DataFrame(index=df.index)
    out["swing_high_confirmed"] = is_sh
    out["swing_low_confirmed"] = is_sl
    out["last_swing_high"] = hi_c.where(is_sh).ffill()
    out["last_swing_low"] = lo_c.where(is_sl).ffill()
    sh_vals = hi_c.where(is_sh)
    sl_vals = lo_c.where(is_sl)
    out["prev_swing_high"] = _previous_event_value(sh_vals)
    out["prev_swing_low"] = _previous_event_value(sl_vals)
    out.attrs["k"] = k
    return out


def _previous_event_value(vals: pd.Series) -> pd.Series:
    """Her olay anında bir önceki olayın değerini taşır, sonra ileri doldurur."""
    events = vals.dropna()
    prev = events.shift()
    return prev.reindex(vals.index).ffill()


def market_structure(df: pd.DataFrame, k: int = 3, atr_series: pd.Series | None = None,
                     equal_tol_atr: float = 0.15) -> pd.DataFrame:
    sw = swing_points(df, k)
    close = df["close"]
    a = atr_series if atr_series is not None else ind.atr(df, 14)
    out = sw.copy()

    out["higher_high"] = out["last_swing_high"] > out["prev_swing_high"]
    out["lower_high"] = out["last_swing_high"] < out["prev_swing_high"]
    out["higher_low"] = out["last_swing_low"] > out["prev_swing_low"]
    out["lower_low"] = out["last_swing_low"] < out["prev_swing_low"]
    bull = out["higher_high"] & out["higher_low"]
    bear = out["lower_high"] & out["lower_low"]
    out["structure_trend"] = np.select([bull, bear], [1, -1], 0)

    prev_sh = out["last_swing_high"].shift()
    prev_sl = out["last_swing_low"].shift()
    up_break = (close > prev_sh) & (close.shift() <= prev_sh)
    dn_break = (close < prev_sl) & (close.shift() >= prev_sl)
    prev_trend = pd.Series(out["structure_trend"], index=df.index).shift()
    out["bos_bull"] = up_break & (prev_trend >= 0)
    out["bos_bear"] = dn_break & (prev_trend <= 0)
    out["choch_bull"] = up_break & (prev_trend == -1)
    out["choch_bear"] = dn_break & (prev_trend == 1)

    # Likidite süpürmesi: önceki swing seviyesinin ötesine fitil, kapanış geri içeride
    out["sweep_bull"] = (df["low"] < prev_sl) & (close > prev_sl)
    out["sweep_bear"] = (df["high"] > prev_sh) & (close < prev_sh)

    # Eşit tepeler / dipler (likidite bölgeleri)
    tol = a * equal_tol_atr
    out["equal_highs"] = (out["last_swing_high"] - out["prev_swing_high"]).abs() <= tol
    out["equal_lows"] = (out["last_swing_low"] - out["prev_swing_low"]).abs() <= tol
    out["liquidity_zone_high"] = out["last_swing_high"].where(out["equal_highs"])
    out["liquidity_zone_low"] = out["last_swing_low"].where(out["equal_lows"])

    sr = _support_resistance(df, sw)
    out["support"] = sr["support"]
    out["resistance"] = sr["resistance"]

    # Fair value gap / imbalance (3 mumluk desen t anında tamamlanır)
    out["fvg_bull"] = df["low"] > df["high"].shift(2)
    out["fvg_bear"] = df["high"] < df["low"].shift(2)
    out["fvg_bull_top"] = df["low"].where(out["fvg_bull"]).ffill()
    out["fvg_bull_bottom"] = df["high"].shift(2).where(out["fvg_bull"]).ffill()

    # Basit order block: yükseliş BOS'undan önceki son düşüş mumunun aralığı (sezgisel)
    bear_candle = close < df["open"]
    last_bear_high = df["high"].where(bear_candle).ffill().shift()
    last_bear_low = df["low"].where(bear_candle).ffill().shift()
    out["ob_bull_high"] = last_bear_high.where(out["bos_bull"]).ffill()
    out["ob_bull_low"] = last_bear_low.where(out["bos_bull"]).ffill()
    bull_candle = close > df["open"]
    last_bull_high = df["high"].where(bull_candle).ffill().shift()
    last_bull_low = df["low"].where(bull_candle).ffill().shift()
    out["ob_bear_high"] = last_bull_high.where(out["bos_bear"]).ffill()
    out["ob_bear_low"] = last_bull_low.where(out["bos_bear"]).ffill()

    out = out.join(previous_period_levels(df))

    # Aralık / sıkışma ve kırılım
    hh20 = df["high"].rolling(20, min_periods=20).max().shift()
    ll20 = df["low"].rolling(20, min_periods=20).min().shift()
    width_atr = (hh20 - ll20) / a.replace(0, np.nan)
    out["range_high"] = hh20
    out["range_low"] = ll20
    out["in_range"] = width_atr < 6
    out["consolidation"] = width_atr < 4
    out["breakout_up"] = close > hh20
    out["breakdown"] = close < ll20
    return out


def _support_resistance(df: pd.DataFrame, sw: pd.DataFrame, keep: int = 12) -> pd.DataFrame:
    """Fiyatın altındaki en yakın onaylı swing dip (destek) ve üstündeki en yakın swing tepe (direnç)."""
    close = df["close"].to_numpy(dtype=float)
    sh_conf = sw["swing_high_confirmed"].to_numpy()
    sl_conf = sw["swing_low_confirmed"].to_numpy()
    hi_k = df["high"].shift(_k_from(sw)).to_numpy(dtype=float)
    lo_k = df["low"].shift(_k_from(sw)).to_numpy(dtype=float)
    highs: list[float] = []
    lows: list[float] = []
    sup = np.full(len(df), np.nan)
    res = np.full(len(df), np.nan)
    for i in range(len(df)):
        if sh_conf[i] and not np.isnan(hi_k[i]):
            highs.append(hi_k[i])
            highs = highs[-keep:]
        if sl_conf[i] and not np.isnan(lo_k[i]):
            lows.append(lo_k[i])
            lows = lows[-keep:]
        c = close[i]
        below = [x for x in lows + highs if x < c]
        above = [x for x in highs + lows if x > c]
        if below:
            sup[i] = max(below)
        if above:
            res[i] = min(above)
    return pd.DataFrame({"support": sup, "resistance": res}, index=df.index)


def _k_from(sw: pd.DataFrame) -> int:
    return int(sw.attrs.get("k", 3))


def previous_period_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Önceki (tamamlanmış) gün ve hafta yüksek/düşük seviyeleri."""
    out = pd.DataFrame(index=df.index)
    if "open_time" not in df:
        for c in ("prev_day_high", "prev_day_low", "prev_week_high", "prev_week_low"):
            out[c] = np.nan
        return out
    t = pd.to_datetime(df["open_time"])
    for name, key in (("day", t.dt.floor("D")), ("week", t.dt.tz_localize(None).dt.to_period("W").dt.start_time)):
        grp = df.groupby(key.values)
        hi = grp["high"].max().shift()
        lo = grp["low"].min().shift()
        out[f"prev_{name}_high"] = pd.Series(key.values).map(hi).to_numpy()
        out[f"prev_{name}_low"] = pd.Series(key.values).map(lo).to_numpy()
    return out


# ---------------------------------------------------------------------- mum formasyonları
def candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Formasyonlar yalnızca uyum (confluence) özelliği olarak kullanılır, tek başına işlem açtırmaz."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    bull = c > o
    bear = c < o
    avg_body = body.rolling(14, min_periods=5).mean()
    prev_up_trend = c.shift() > c.shift(5)
    prev_down_trend = c.shift() < c.shift(5)

    p = pd.DataFrame(index=df.index)
    p["doji"] = body <= 0.1 * rng
    p["hammer"] = (lower >= 2 * body) & (upper <= 0.3 * body.clip(lower=1e-12) + 0.1 * rng) & prev_down_trend
    p["inverted_hammer"] = (upper >= 2 * body) & (lower <= 0.1 * rng) & prev_down_trend
    p["shooting_star"] = (upper >= 2 * body) & (lower <= 0.1 * rng) & prev_up_trend
    p["bullish_engulfing"] = bull & bear.shift(fill_value=False) & (c >= o.shift()) & (o <= c.shift()) & \
        (body > body.shift())
    p["bearish_engulfing"] = bear & bull.shift(fill_value=False) & (o >= c.shift()) & (c <= o.shift()) & \
        (body > body.shift())
    small_mid = body.shift() < 0.5 * avg_body
    p["morning_star"] = bear.shift(2, fill_value=False) & (body.shift(2) > avg_body) & small_mid & bull & \
        (c > (o.shift(2) + c.shift(2)) / 2)
    p["evening_star"] = bull.shift(2, fill_value=False) & (body.shift(2) > avg_body) & small_mid & bear & \
        (c < (o.shift(2) + c.shift(2)) / 2)
    p["pin_bar_bull"] = (lower >= 0.66 * rng) & (body <= 0.25 * rng)
    p["pin_bar_bear"] = (upper >= 0.66 * rng) & (body <= 0.25 * rng)
    p["inside_bar"] = (h < h.shift()) & (l > l.shift())
    p["outside_bar"] = (h > h.shift()) & (l < l.shift())
    p["three_white_soldiers"] = bull & bull.shift(fill_value=False) & bull.shift(2, fill_value=False) & \
        (c > c.shift()) & (c.shift() > c.shift(2)) & (body > 0.5 * avg_body)
    p["three_black_crows"] = bear & bear.shift(fill_value=False) & bear.shift(2, fill_value=False) & \
        (c < c.shift()) & (c.shift() < c.shift(2)) & (body > 0.5 * avg_body)
    p = p.fillna(False).astype(bool)
    bull_cols = ["hammer", "inverted_hammer", "bullish_engulfing", "morning_star", "pin_bar_bull",
                 "three_white_soldiers"]
    bear_cols = ["shooting_star", "bearish_engulfing", "evening_star", "pin_bar_bear", "three_black_crows"]
    p["pattern_bull_count"] = p[bull_cols].sum(axis=1)
    p["pattern_bear_count"] = p[bear_cols].sum(axis=1)
    return p
