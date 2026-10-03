"""Geçmiş veride strateji testi (backtest).

Sinyal bir mumun kapanışında oluşur, emir bir sonraki mumun açılışında
gerçekleşir. Stop-loss / kâr al mum içi en düşük / en yüksek ile kontrol edilir;
aynı mumda ikisi de tetiklenirse temkinli davranılıp stop kabul edilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr
from .models import ClosedTrade
from .risk import RiskManager, RiskSettings
from .strategies import BUY, SELL, Strategy
from .utils import interval_seconds


@dataclass
class BacktestResult:
    strategy_name: str
    initial_capital: float
    final_equity: float
    equity: pd.Series
    trades: list[ClosedTrade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    buy_points: list[int] = field(default_factory=list)
    sell_points: list[int] = field(default_factory=list)


def _ts(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def run_backtest(df: pd.DataFrame, strategy: Strategy, risk: RiskSettings | None = None,
                 initial_capital: float = 1000.0, interval: str = "1h", symbol: str = "") -> BacktestResult:
    risk = risk or RiskSettings()
    rm = RiskManager(risk)
    fee = risk.fee_pct / 100
    slip = risk.slippage_pct / 100

    if len(df) < strategy.min_bars() + 2:
        raise ValueError(f"Backtest için en az {strategy.min_bars() + 2} mum gerekli.")

    sig = strategy.signals(df).to_numpy()
    atr_arr = atr(df, 14).to_numpy()
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    times = df["open_time"].to_numpy() if "open_time" in df else np.arange(len(df))

    cash = float(initial_capital)
    pos = None
    trades: list[ClosedTrade] = []
    equity_curve = np.empty(len(df))
    buys, sells = [], []
    bars_in_market = 0

    def close_position(i: int, raw_price: float, reason: str):
        nonlocal cash, pos
        exit_price = raw_price * (1 - slip)
        gross = pos["qty"] * exit_price
        proceeds = gross * (1 - fee)
        cash += proceeds
        pnl = proceeds - pos["cost"]
        trades.append(ClosedTrade(
            symbol=symbol, qty=pos["qty"], entry_price=pos["entry"], exit_price=exit_price,
            cost=pos["cost"], proceeds=proceeds, pnl=pnl, pnl_pct=pnl / pos["cost"] * 100,
            opened_at=_ts(times[pos["i"]]), closed_at=_ts(times[i]), reason=reason,
            strategy=strategy.name,
        ))
        sells.append(i)
        pos = None

    for i in range(len(df)):
        # 1) Önceki mumun kapanış sinyalini bu mumun açılışında uygula.
        if i > 0:
            s = sig[i - 1]
            if pos is None and s == BUY and math.isfinite(atr_arr[i - 1]):
                entry = o[i] * (1 + slip)
                qty = rm.position_size(cash, entry, atr_arr[i - 1])
                qty = min(qty, cash / (entry * (1 + fee)))
                if qty * entry > 1e-9:
                    cost = qty * entry * (1 + fee)
                    cash -= cost
                    stop, take = rm.stops(entry, atr_arr[i - 1])
                    pos = {"qty": qty, "entry": entry, "cost": cost, "stop": stop,
                           "take": take, "high": entry, "i": i}
                    buys.append(i)
            elif pos is not None and s == SELL:
                close_position(i, o[i], "Strateji sinyali")

        # 2) Mum içi stop-loss / kâr al.
        if pos is not None:
            if pos["stop"] > 0 and lo[i] <= pos["stop"]:
                close_position(i, min(o[i], pos["stop"]), "Stop-loss")
            elif pos["take"] > 0 and h[i] >= pos["take"]:
                close_position(i, max(o[i], pos["take"]), "Kâr al")
            else:
                pos["high"] = max(pos["high"], h[i])
                if math.isfinite(atr_arr[i]):
                    pos["stop"] = rm.trail(pos["stop"], pos["high"], atr_arr[i])

        if pos is not None:
            bars_in_market += 1
        equity_curve[i] = cash + (pos["qty"] * c[i] if pos is not None else 0.0)

    if pos is not None:
        close_position(len(df) - 1, c[-1], "Test sonu")
        equity_curve[-1] = cash

    index = pd.DatetimeIndex(df["open_time"]) if "open_time" in df else df.index
    equity = pd.Series(equity_curve, index=index)
    metrics = compute_metrics(equity, trades, initial_capital, c, interval, bars_in_market)
    return BacktestResult(
        strategy_name=strategy.name, initial_capital=initial_capital,
        final_equity=float(equity.iloc[-1]), equity=equity, trades=trades,
        metrics=metrics, buy_points=buys, sell_points=sells,
    )


def compute_metrics(equity: pd.Series, trades: list[ClosedTrade], initial: float,
                    close: np.ndarray, interval: str, bars_in_market: int) -> dict:
    final = float(equity.iloc[-1])
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    returns = equity.pct_change().dropna()
    bars_per_year = 365 * 24 * 3600 / interval_seconds(interval)
    std = float(returns.std()) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean()) / std * math.sqrt(bars_per_year) if std > 0 else 0.0

    return {
        "Toplam getiri (%)": (final / initial - 1) * 100,
        "Al-tut getirisi (%)": (close[-1] / close[0] - 1) * 100,
        "Son bakiye": final,
        "İşlem sayısı": len(trades),
        "Kazanma oranı (%)": len(wins) / len(trades) * 100 if trades else 0.0,
        "Kâr faktörü": profit_factor,
        "Maks. düşüş (%)": float(drawdown.min()),
        "Sharpe oranı": sharpe,
        "Ort. işlem (%)": float(np.mean([t.pnl_pct for t in trades])) if trades else 0.0,
        "En iyi işlem (%)": max((t.pnl_pct for t in trades), default=0.0),
        "En kötü işlem (%)": min((t.pnl_pct for t in trades), default=0.0),
        "Piyasada kalma (%)": bars_in_market / len(equity) * 100,
    }


def compare_strategies(df: pd.DataFrame, strategies: list[Strategy], risk: RiskSettings,
                       initial_capital: float, interval: str, symbol: str = "") -> list[BacktestResult]:
    results = []
    for strat in strategies:
        try:
            results.append(run_backtest(df, strat, risk, initial_capital, interval, symbol))
        except ValueError:
            continue
    results.sort(key=lambda r: r.metrics["Toplam getiri (%)"], reverse=True)
    return results
