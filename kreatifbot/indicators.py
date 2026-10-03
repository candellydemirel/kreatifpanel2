"""Teknik göstergeler.

Tüm fonksiyonlar yalnızca geçmiş veriyi kullanır (ileriye bakma yok), bu
yüzden backtest ve canlı işlemde aynı sonuçları üretir.
DataFrame girdileri 'open', 'high', 'low', 'close', 'volume' sütunlarını içerir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder yumuşatması (RSI, ATR ve ADX'te kullanılır)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    avg_gain = wilder(delta.clip(lower=0), period)
    avg_loss = wilder(-delta.clip(upper=0), period)
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """(macd çizgisi, sinyal çizgisi, histogram) döndürür."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    """(orta, üst, alt) bantları döndürür."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + mult * std, mid - mult * std


def bollinger_percent_b(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    _, upper, lower = bollinger(close, period, mult)
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14):
    """(adx, +DI, -DI) döndürür."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = wilder(true_range(df), period).replace(0, np.nan)
    plus_di = 100 * wilder(plus_dm, period) / tr
    minus_di = 100 * wilder(minus_dm, period) / tr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return wilder(dx, period), plus_di, minus_di


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """(%K, %D) döndürür."""
    lowest = df["low"].rolling(k_period, min_periods=k_period).min()
    highest = df["high"].rolling(k_period, min_periods=k_period).max()
    k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    return k, sma(k, d_period)


def donchian(df: pd.DataFrame, period: int = 20):
    """(üst, alt) kanal döndürür."""
    upper = df["high"].rolling(period, min_periods=period).max()
    lower = df["low"].rolling(period, min_periods=period).min()
    return upper, lower


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """(supertrend çizgisi, yön) döndürür. Yön: 1 yükseliş, -1 düşüş, 0 belirsiz."""
    atr_arr = atr(df, period).to_numpy()
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    hl2 = (high + low) / 2
    upper_basic = hl2 + mult * atr_arr
    lower_basic = hl2 - mult * atr_arr

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    direction = np.zeros(n)

    for i in range(n):
        if np.isnan(atr_arr[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper_basic[i]
            final_lower[i] = lower_basic[i]
            direction[i] = 1 if close[i] >= hl2[i] else -1
        else:
            if upper_basic[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper_basic[i]
            else:
                final_upper[i] = final_upper[i - 1]
            if lower_basic[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower_basic[i]
            else:
                final_lower[i] = final_lower[i - 1]
            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < final_lower[i] else 1
            else:
                direction[i] = 1 if close[i] > final_upper[i] else -1
        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def cross_above(a: pd.Series, b) -> pd.Series:
    """a, b'yi aşağıdan yukarı kestiği mumda True."""
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift() <= b_prev)


def cross_below(a: pd.Series, b) -> pd.Series:
    """a, b'yi yukarıdan aşağı kestiği mumda True."""
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift() >= b_prev)


# ---------------------------------------------------------------------------
# Genişletilmiş gösterge seti (zeka motoru). Hepsi nedenseldir: t anındaki değer
# yalnızca t ve öncesindeki mumları kullanır.
# ---------------------------------------------------------------------------

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """UTC gün başında sıfırlanan oturum VWAP'ı (gün içi kümülatif)."""
    tp = typical_price(df)
    day = pd.to_datetime(df["open_time"]).dt.floor("D") if "open_time" in df else pd.Series(0, index=df.index)
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vol


def vwap_rolling(df: pd.DataFrame, period: int = 50) -> pd.Series:
    tp = typical_price(df)
    pv = (tp * df["volume"]).rolling(period, min_periods=period).sum()
    vol = df["volume"].rolling(period, min_periods=period).sum().replace(0, np.nan)
    return pv / vol


def anchored_vwap(df: pd.DataFrame, anchor_mask: pd.Series) -> pd.Series:
    """Her True işaretinde yeniden başlayan VWAP (ör. son kırılım veya swing noktası)."""
    tp = typical_price(df)
    group = anchor_mask.fillna(False).astype(int).cumsum()
    pv = (tp * df["volume"]).groupby(group).cumsum()
    vol = df["volume"].groupby(group).cumsum().replace(0, np.nan)
    out = pv / vol
    return out.where(group > 0)


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=df.index)
    bull = high[1] >= high[0]
    af = step
    ep = high[0] if bull else low[0]
    sar[0] = low[0] if bull else high[0]
    for i in range(1, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if bull:
            cur = min(cur, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < cur:
                bull, cur, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            cur = max(cur, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > cur:
                bull, cur, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        sar[i] = cur
    return pd.Series(sar, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
    """(tenkan, kijun, span_a, span_b) — bulut değerleri t anında bilinen haliyle (26 mum geriden).

    Chikou span gelecekteki fiyatı geçmişe çizdiği için ileriye bakma oluşturur; bunun yerine
    'chikou onayı' olarak kapanışın 26 mum önceki kapanışla karşılaştırması kullanılmalıdır.
    """
    def mid(n):
        return (df["high"].rolling(n, min_periods=n).max() + df["low"].rolling(n, min_periods=n).min()) / 2
    t, k = mid(tenkan), mid(kijun)
    span_a = ((t + k) / 2).shift(kijun)
    span_b = mid(senkou).shift(kijun)
    return t, k, span_a, span_b


def stoch_rsi(close: pd.Series, period: int = 14, k: int = 3, d: int = 3):
    r = rsi(close, period)
    lo = r.rolling(period, min_periods=period).min()
    hi = r.rolling(period, min_periods=period).max()
    raw = (100 * (r - lo) / (hi - lo).replace(0, np.nan)).clip(0, 100)
    k_line = sma(raw, k)
    return k_line, sma(k_line, d)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return close - close.shift(period)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = typical_price(df)
    ma = tp.rolling(period, min_periods=period).mean()
    md = tp.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = typical_price(df)
    flow = tp * df["volume"]
    direction = tp.diff()
    pos = flow.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    neg = flow.where(direction < 0, 0.0).rolling(period, min_periods=period).sum()
    out = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return out.where(neg != 0, 100.0).where(pos.notna())


def bollinger_width(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    mid, upper, lower = bollinger(close, period, mult)
    return (upper - lower) / mid.replace(0, np.nan)


def keltner(df: pd.DataFrame, period: int = 20, mult: float = 2.0, atr_period: int = 10):
    mid = ema(df["close"], period)
    a = atr(df, atr_period)
    return mid, mid + mult * a, mid - mult * a


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift())


def historical_volatility(close: pd.Series, period: int = 30, bars_per_year: float = 365 * 24) -> pd.Series:
    """Yıllıklandırılmış standart sapma (%)."""
    return log_returns(close).rolling(period, min_periods=period).std() * np.sqrt(bars_per_year) * 100


def realized_volatility(close: pd.Series, period: int = 30) -> pd.Series:
    """Pencere içi gerçekleşen volatilite: sqrt(Σ r²) (%)."""
    r = log_returns(close)
    return np.sqrt((r ** 2).rolling(period, min_periods=period).sum()) * 100


def rolling_percentile(series: pd.Series, window: int = 250) -> pd.Series:
    """Son değerin pencere içindeki yüzdelik sırası (0..1), nedensel."""
    return series.rolling(window, min_periods=max(20, window // 5)).rank(pct=True)


def zscore(series: pd.Series, period: int = 50) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Hacim / önceki N mumun ortalaması (mevcut mum ortalamaya dahil değil)."""
    return volume / volume.shift().rolling(period, min_periods=period).mean().replace(0, np.nan)


def volume_delta(df: pd.DataFrame) -> pd.Series:
    """Taker alış hacmi - taker satış hacmi (Binance kline taker verisinden). Veri yoksa NaN."""
    if "taker_buy_base" not in df:
        return pd.Series(np.nan, index=df.index)
    return 2 * df["taker_buy_base"] - df["volume"]


def cvd(df: pd.DataFrame) -> pd.Series:
    d = volume_delta(df)
    return d.cumsum() if d.notna().any() else d
