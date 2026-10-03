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
