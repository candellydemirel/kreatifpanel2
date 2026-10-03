"""İşlem stratejileri.

Her strateji, her mum için 1 (AL), -1 (SAT) veya 0 (BEKLE) içeren bir sinyal
serisi üretir. Sinyal, ilgili mumun kapanışında bilinen veriyle hesaplanır.
Spot piyasada açığa satış olmadığı için SAT sinyali açık pozisyonu kapatır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind

BUY, SELL, HOLD = 1, -1, 0
SIGNAL_TEXT = {BUY: "AL", SELL: "SAT", HOLD: "BEKLE"}


@dataclass
class StrategyResult:
    signal: int
    reason: str
    score: float | None = None

    @property
    def text(self) -> str:
        return SIGNAL_TEXT[self.signal]


def _to_signals(index, buy: pd.Series, sell: pd.Series) -> pd.Series:
    buy = buy.fillna(False).astype(bool)
    sell = sell.fillna(False).astype(bool) & ~buy
    out = pd.Series(HOLD, index=index, dtype=int)
    out[buy] = BUY
    out[sell] = SELL
    return out


class Strategy:
    key = ""
    name = ""
    description = ""
    # parametre adı -> (varsayılan, min, max, etiket)
    param_specs: dict[str, tuple] = {}

    def __init__(self, **params):
        self.params = {k: spec[0] for k, spec in self.param_specs.items()}
        for k, v in params.items():
            if k in self.param_specs:
                default = self.param_specs[k][0]
                self.params[k] = int(v) if isinstance(default, int) else float(v)

    def p(self, name):
        return self.params[name]

    def min_bars(self) -> int:
        ints = [v for v in self.params.values() if isinstance(v, int)]
        return max(ints + [30]) + 5

    def signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def describe_state(self, df: pd.DataFrame) -> str:
        return ""

    def evaluate(self, df: pd.DataFrame) -> StrategyResult:
        if len(df) < self.min_bars():
            return StrategyResult(HOLD, f"Yetersiz veri (en az {self.min_bars()} mum gerekli)")
        sig = int(self.signals(df).iloc[-1])
        state = self.describe_state(df)
        return StrategyResult(sig, state)

    def __repr__(self) -> str:
        return f"{self.name} {self.params}"


class EmaCross(Strategy):
    key = "ema_cross"
    name = "EMA Kesişimi"
    description = "Hızlı EMA yavaş EMA'yı yukarı keserse AL, aşağı keserse SAT. İsteğe bağlı uzun vadeli trend filtresi."
    param_specs = {
        "fast": (9, 2, 100, "Hızlı EMA"),
        "slow": (21, 3, 300, "Yavaş EMA"),
        "trend": (200, 0, 400, "Trend filtresi EMA (0=kapalı)"),
    }

    def signals(self, df):
        close = df["close"]
        fast, slow = ind.ema(close, self.p("fast")), ind.ema(close, self.p("slow"))
        buy = ind.cross_above(fast, slow)
        if self.p("trend") > 0:
            buy &= close > ind.ema(close, self.p("trend"))
        return _to_signals(df.index, buy, ind.cross_below(fast, slow))

    def describe_state(self, df):
        close = df["close"]
        fast = ind.ema(close, self.p("fast")).iloc[-1]
        slow = ind.ema(close, self.p("slow")).iloc[-1]
        return f"EMA{self.p('fast')} {'>' if fast > slow else '<'} EMA{self.p('slow')}"


class RsiReversion(Strategy):
    key = "rsi"
    name = "RSI Dönüş"
    description = "RSI aşırı satım bölgesinden yukarı çıkınca AL, aşırı alım bölgesine girince SAT."
    param_specs = {
        "period": (14, 2, 50, "RSI periyodu"),
        "oversold": (30.0, 5.0, 50.0, "Aşırı satım"),
        "overbought": (70.0, 50.0, 95.0, "Aşırı alım"),
    }

    def signals(self, df):
        r = ind.rsi(df["close"], self.p("period"))
        buy = ind.cross_above(r, self.p("oversold"))
        sell = ind.cross_above(r, self.p("overbought"))
        return _to_signals(df.index, buy, sell)

    def describe_state(self, df):
        return f"RSI {ind.rsi(df['close'], self.p('period')).iloc[-1]:.1f}"


class MacdMomentum(Strategy):
    key = "macd"
    name = "MACD Momentum"
    description = "MACD çizgisi sinyal çizgisini yukarı keserse AL, aşağı keserse SAT."
    param_specs = {
        "fast": (12, 2, 50, "Hızlı"),
        "slow": (26, 5, 100, "Yavaş"),
        "signal": (9, 2, 50, "Sinyal"),
        "trend": (0, 0, 400, "Trend filtresi EMA (0=kapalı)"),
    }

    def signals(self, df):
        line, sig, _ = ind.macd(df["close"], self.p("fast"), self.p("slow"), self.p("signal"))
        buy = ind.cross_above(line, sig)
        if self.p("trend") > 0:
            buy &= df["close"] > ind.ema(df["close"], self.p("trend"))
        return _to_signals(df.index, buy, ind.cross_below(line, sig))

    def describe_state(self, df):
        _, _, hist = ind.macd(df["close"], self.p("fast"), self.p("slow"), self.p("signal"))
        h = hist.iloc[-1]
        return f"Histogram {'pozitif' if h > 0 else 'negatif'} ({h:.4g})"


class BollingerReversion(Strategy):
    key = "bollinger"
    name = "Bollinger Dönüş"
    description = "Fiyat alt banttan içeri dönünce AL, orta banda (veya üst banda) ulaşınca SAT."
    param_specs = {
        "period": (20, 5, 100, "Periyot"),
        "mult": (2.0, 1.0, 4.0, "Standart sapma çarpanı"),
        "exit_upper": (0, 0, 1, "Çıkış üst bantta (1) / orta bantta (0)"),
    }

    def signals(self, df):
        close = df["close"]
        mid, upper, lower = ind.bollinger(close, self.p("period"), self.p("mult"))
        buy = ind.cross_above(close, lower)
        target = upper if self.p("exit_upper") else mid
        return _to_signals(df.index, buy, ind.cross_above(close, target))

    def describe_state(self, df):
        pb = ind.bollinger_percent_b(df["close"], self.p("period"), self.p("mult")).iloc[-1]
        return f"%B {pb:.2f}"


class SupertrendStrategy(Strategy):
    key = "supertrend"
    name = "Supertrend"
    description = "Supertrend yönü yukarı dönünce AL, aşağı dönünce SAT. Güçlü trendlerde etkilidir."
    param_specs = {
        "period": (10, 3, 50, "ATR periyodu"),
        "mult": (3.0, 1.0, 6.0, "ATR çarpanı"),
    }

    def signals(self, df):
        _, direction = ind.supertrend(df, self.p("period"), self.p("mult"))
        prev = direction.shift()
        return _to_signals(df.index, (direction == 1) & (prev == -1), (direction == -1) & (prev == 1))

    def describe_state(self, df):
        _, direction = ind.supertrend(df, self.p("period"), self.p("mult"))
        return "Yön: yükseliş" if direction.iloc[-1] == 1 else "Yön: düşüş"


class DonchianBreakout(Strategy):
    key = "breakout"
    name = "Kanal Kırılımı (Donchian)"
    description = "Fiyat son N mumun en yükseğini kırarsa AL, son M mumun en düşüğünün altına inerse SAT (Turtle)."
    param_specs = {
        "entry": (20, 5, 200, "Giriş kanalı"),
        "exit": (10, 3, 100, "Çıkış kanalı"),
    }

    def signals(self, df):
        close = df["close"]
        upper, _ = ind.donchian(df, self.p("entry"))
        _, lower = ind.donchian(df, self.p("exit"))
        prev_upper, prev_lower = upper.shift(), lower.shift()
        above = close > prev_upper
        below = close < prev_lower
        buy = above & ~above.shift(fill_value=False)
        sell = below & ~below.shift(fill_value=False)
        return _to_signals(df.index, buy, sell)

    def describe_state(self, df):
        upper, _ = ind.donchian(df, self.p("entry"))
        return f"Kanal üstü {upper.iloc[-2]:.6g}" if len(df) > 1 else ""


class SmartEnsemble(Strategy):
    key = "ensemble"
    name = "Akıllı Kombine (Önerilen)"
    description = (
        "Piyasa rejimini ADX ile ölçer. Trend piyasasında trend göstergelerine (EMA, MACD, Supertrend), "
        "yatay piyasada dönüş göstergelerine (RSI, Bollinger) daha fazla ağırlık verir. "
        "Birleşik skor eşiği aşarsa AL, çıkış eşiğinin altına düşerse SAT."
    )
    param_specs = {
        "threshold": (0.5, 0.1, 1.0, "Giriş eşiği"),
        "exit_threshold": (-0.2, -1.0, 0.5, "Çıkış eşiği"),
        "adx_trend": (25.0, 10.0, 50.0, "Trend kabul ADX"),
    }

    def min_bars(self) -> int:
        return 60

    def score(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        t1 = np.sign(ind.ema(close, 9) - ind.ema(close, 21))
        _, _, hist = ind.macd(close)
        t2 = np.sign(hist)
        _, st_dir = ind.supertrend(df, 10, 3.0)
        t3 = st_dir.replace(0, np.nan)
        r = ind.rsi(close, 14)
        r1 = pd.Series(np.select([r < 35, r > 65], [1.0, -1.0], 0.0), index=df.index).where(r.notna())
        pb = ind.bollinger_percent_b(close, 20, 2.0)
        r2 = pd.Series(np.select([pb < 0.1, pb > 0.9], [1.0, -1.0], 0.0), index=df.index).where(pb.notna())
        adx_val, plus_di, minus_di = ind.adx(df, 14)

        trend_part = (t1 + t2 + t3) / 3
        revert_part = (r1 + r2) / 2
        trending = adx_val >= self.p("adx_trend")
        score = pd.Series(
            np.where(trending, 0.8 * trend_part + 0.2 * revert_part, 0.35 * trend_part + 0.65 * revert_part),
            index=df.index,
        )
        # Güçlü düşüş trendinde dönüş alımlarını engelle.
        strong_down = trending & (minus_di > plus_di) & (trend_part < 0)
        score = score.where(~strong_down, np.minimum(score, 0))
        valid = trend_part.notna() & revert_part.notna() & adx_val.notna()
        return score.where(valid)

    def signals(self, df):
        s = self.score(df)
        buy = ind.cross_above(s, self.p("threshold"))
        sell = ind.cross_below(s, self.p("exit_threshold"))
        return _to_signals(df.index, buy, sell)

    def evaluate(self, df):
        result = super().evaluate(df)
        if len(df) >= self.min_bars():
            result.score = float(self.score(df).iloc[-1])
        return result

    def describe_state(self, df):
        s = self.score(df).iloc[-1]
        adx_val = ind.adx(df, 14)[0].iloc[-1]
        regime = "trend" if adx_val >= self.p("adx_trend") else "yatay"
        return f"Skor {s:+.2f}, rejim: {regime} (ADX {adx_val:.0f})"


STRATEGIES: dict[str, type[Strategy]] = {
    cls.key: cls
    for cls in (SmartEnsemble, EmaCross, RsiReversion, MacdMomentum, BollingerReversion,
                SupertrendStrategy, DonchianBreakout)
}


def create_strategy(key: str, params: dict | None = None) -> Strategy:
    if key not in STRATEGIES:
        raise ValueError(f"Bilinmeyen strateji: {key}")
    return STRATEGIES[key](**(params or {}))
