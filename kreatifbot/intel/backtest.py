"""Zeka motoru backtest'i (olay tabanlı, maliyet duyarlı).

Kurallar:
- Karar mum KAPANIŞINDA verilir; emir bir sonraki mumun açılışında (gecikme) gerçekleşir.
- Ücret (maker/taker), spread, kayma, futures funding ve kısmi dolum varsayımı uygulanır.
- Mum içi SL/TP sırası temkinlidir (önce SL).
- Brüt ve net PnL ayrı raporlanır. Veri yoksa DATA_UNAVAILABLE; sahte sonuç üretilmez.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..utils import INTERVALS
from .config import IntelConfig
from .decision import DecisionEngine, Prepared
from .position_manager import evaluate_signal_decay, open_position, update_on_bar
from .risk_engine import PortfolioState
from .scoring import total_scores
from .types import ExitReason, Regime

FUNDING_HOURS = (0, 8, 16)


@dataclass
class IntelTrade:
    symbol: str
    direction: str
    strategy: str
    regime: str
    signal_id: str
    entry_time: str
    exit_time: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    r_multiple: float
    holding_bars: int
    holding_min: float
    exit_reason: str
    confidence: float
    ai_probability: float
    risk_amount: float
    notional: float
    partial_exits: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntelBacktestResult:
    symbol: str
    interval: str
    market: str
    trades: list
    equity: pd.Series
    metrics: dict
    by_regime: pd.DataFrame
    by_strategy: pd.DataFrame
    exit_reasons: pd.Series
    holding: dict
    signal_stats: dict
    decisions: pd.DataFrame
    assumptions: list
    status: str = "OK"

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])


def data_unavailable(symbol: str, interval: str, market: str, reason: str) -> IntelBacktestResult:
    return IntelBacktestResult(symbol, interval, market, [], pd.Series(dtype=float), {"status": "DATA_UNAVAILABLE",
                               "reason": reason}, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int), {}, {},
                               pd.DataFrame(), [], status="DATA_UNAVAILABLE")


def apply_exit_mode(cfg: IntelConfig, mode: str | None) -> IntelConfig:
    """Çıkış optimizasyonu için yapılandırma varyantları."""
    if not mode or mode == "default":
        return cfg
    c = copy.deepcopy(cfg)
    r = c.risk
    inf = 1e9
    if mode == "fixed_tp_sl":
        r.tp_levels_r, r.tp_fractions = [2.0], [1.0]
        r.breakeven_at_r = r.profit_lock_at_r = r.trailing_at_r = inf
    elif mode == "atr_tp_sl":
        r.tp_method, r.tp_levels_r, r.tp_fractions = "atr", [1.0], [1.0]
        r.breakeven_at_r = r.profit_lock_at_r = r.trailing_at_r = inf
    elif mode == "trailing_only":
        r.tp_levels_r, r.tp_fractions = [50.0], [1.0]
        r.breakeven_at_r, r.profit_lock_at_r, r.trailing_at_r = inf, inf, 0.0
    elif mode == "breakeven":
        r.tp_levels_r, r.tp_fractions = [2.0], [1.0]
        r.profit_lock_at_r = r.trailing_at_r = inf
    elif mode == "time_only":
        r.tp_levels_r, r.tp_fractions = [50.0], [1.0]
        r.breakeven_at_r = r.profit_lock_at_r = r.trailing_at_r = inf
    elif mode in ("signal_reversal", "regime_exit"):
        r.tp_levels_r, r.tp_fractions = [50.0], [1.0]
        r.breakeven_at_r = r.profit_lock_at_r = r.trailing_at_r = inf
    return c


def run_intel_backtest(engine: DecisionEngine, prep: Prepared, start: int | None = None, end: int | None = None,
                       initial_equity: float | None = None, fee_mult: float = 1.0, slippage_mult: float = 1.0,
                       entry_mode: str | None = None, exit_mode: str | None = None,
                       funding_events: pd.DataFrame | None = None, disabled_groups: set | None = None,
                       record_decisions: bool = True, use_decay_exits: bool = True) -> IntelBacktestResult:
    cfg = apply_exit_mode(engine.cfg, exit_mode)
    base_engine_cfg = engine.cfg
    engine.cfg = cfg
    try:
        return _run(engine, prep, cfg, start, end, initial_equity, fee_mult, slippage_mult, entry_mode, exit_mode,
                    funding_events, disabled_groups, record_decisions, use_decay_exits)
    finally:
        engine.cfg = base_engine_cfg


def _run(engine, prep, cfg, start, end, initial_equity, fee_mult, slippage_mult, entry_mode, exit_mode,
         funding_events, disabled_groups, record_decisions, use_decay_exits):
    f = prep.f
    n = len(f)
    if n < 100:
        return data_unavailable(prep.symbol, prep.interval, prep.market, f"Yetersiz veri ({n} mum)")
    if disabled_groups:
        prep = copy.copy(prep)
        prep.totals = total_scores(prep.gs, cfg.scoring, disabled_groups)
    start = max(start or 0, 1)
    end = min(end or n, n)
    sec = INTERVALS[prep.interval]
    eq0 = float(initial_equity or cfg.backtest.initial_equity)
    entry_mode = entry_mode or cfg.backtest.entry_mode
    market = prep.market
    c = cfg.costs
    taker = (c.spot_taker_fee_pct if market == "SPOT" else c.futures_taker_fee_pct) / 100 * fee_mult
    maker = (c.spot_maker_fee_pct if market == "SPOT" else c.futures_maker_fee_pct) / 100 * fee_mult
    slip = c.slippage_pct / 100 * slippage_mult
    half_spread = c.default_spread_pct / 100 / 2
    assumptions = [f"Spread varsayımı %{c.default_spread_pct} (geçmiş order book verisi yok)",
                   f"Kayma %{c.slippage_pct * slippage_mult:.3f}/işlem, ücret x{fee_mult}",
                   "Giriş: sinyal mumundan sonraki mumun açılışı (gecikme)",
                   "Aynı mumda SL ve TP: temkinli olarak SL varsayıldı"]
    o = f["open"].to_numpy(dtype=float)
    cl = f["close"].to_numpy(dtype=float)
    lo = f["low"].to_numpy(dtype=float)
    hi = f["high"].to_numpy(dtype=float)
    times = pd.to_datetime(f["open_time"], utc=True)
    regimes = f["regime"].astype(str).to_numpy()
    funding_rate_col = f["funding_rate"].to_numpy(dtype=float) if "funding_rate" in f else np.full(n, np.nan)
    funding_assumed = market != "SPOT" and not np.isfinite(funding_rate_col).any()
    if funding_assumed:
        assumptions.append(f"Funding verisi yok: %{c.assumed_funding_rate_8h * 100:.3f}/8s varsayıldı")

    equity, cash_pnl = eq0, 0.0
    peak, day_start, cur_day = eq0, eq0, None
    consecutive_losses = 0
    pos, pending, pos_meta = None, None, {}
    trades: list[IntelTrade] = []
    eq_curve = np.full(n, np.nan)
    decisions = []
    stats = {"decisions": 0, "candidates": 0, "confirmed": 0, "executed": 0, "expired": 0, "cancelled": 0,
             "no_trade": {}}

    def exit_fill(price, s):
        return price * (1 - s * (slip + half_spread))

    def close_part(action_qty, raw_price, reason, i, note=""):
        nonlocal cash_pnl
        s = pos.sign
        px = exit_fill(raw_price, s)
        gross = (px - pos.entry_price) * s * action_qty
        fee = px * action_qty * taker
        pos_meta["gross"] += gross
        pos_meta["fees"] += fee
        pos_meta["exits"].append({"bar": i, "qty": action_qty, "price": px, "reason": str(reason), "note": note})
        cash_pnl += gross - fee

    def finalize(i, reason):
        nonlocal pos, consecutive_losses
        net = pos_meta["gross"] - pos_meta["fees"] - pos_meta["funding"]
        exits = pos_meta["exits"]
        tq = sum(e["qty"] for e in exits) or 1
        avg_exit = sum(e["qty"] * e["price"] for e in exits) / tq
        r_mult = net / pos_meta["risk_amount"] if pos_meta["risk_amount"] > 0 else 0.0
        hb = i - pos.opened_bar + 1
        trades.append(IntelTrade(
            symbol=prep.symbol, direction=pos.direction, strategy=pos.strategy, regime=pos.regime_at_entry,
            signal_id=pos.signal_id, entry_time=str(times.iloc[pos.opened_bar]), exit_time=str(times.iloc[i]),
            entry_bar=pos.opened_bar, exit_bar=i, entry_price=pos.entry_price, exit_price=avg_exit,
            qty=pos.initial_qty, gross_pnl=pos_meta["gross"], fees=pos_meta["fees"], funding=pos_meta["funding"],
            net_pnl=net, r_multiple=r_mult, holding_bars=hb, holding_min=hb * sec / 60, exit_reason=str(reason),
            confidence=pos.initial_confidence, ai_probability=pos_meta.get("ai", float("nan")),
            risk_amount=pos_meta["risk_amount"], notional=pos.entry_price * pos.initial_qty, partial_exits=exits))
        consecutive_losses = consecutive_losses + 1 if net <= 0 else 0
        pos = None

    for i in range(start, end):
        day = times.iloc[i].date()
        if day != cur_day:
            cur_day, day_start = day, equity
        # ---- 1) bekleyen sinyali gerçekleştir (bu mumun açılışı)
        if pending is not None and pos is None:
            d = pending["d"]
            s = 1 if d.direction == "LONG" else -1
            age = i - pending["bar"]
            fill_px, fee_rate, filled_ratio = None, taker, 1.0
            if age > d.expiry_bars:
                stats["expired"] += 1
                pending = None
            elif entry_mode == "next_open" or (entry_mode == "vwap_reclaim" and (cl[i - 1] - f["vwap"].iat[i - 1]) * s > 0):
                fill_px = o[i] * (1 + s * (slip + half_spread))
            elif entry_mode in ("limit_pullback", "retest"):
                atr_i = float(f["atr"].iat[pending["bar"]])
                limit = d.entry - s * (0.5 * atr_i if entry_mode == "limit_pullback" else 0.0)
                touched = lo[i] <= limit if s > 0 else hi[i] >= limit
                if touched:
                    fill_px = min(o[i], limit) if s > 0 else max(o[i], limit)
                    fill_px *= (1 + s * half_spread)
                    fee_rate, filled_ratio = maker, c.limit_fill_ratio
            if fill_px is not None and pending is not None:
                if (fill_px - d.stop_loss) * s <= 0:
                    stats["cancelled"] += 1
                else:
                    risk_amount = abs(d.entry - d.stop_loss) * d.position_size
                    qty = min(risk_amount / abs(fill_px - d.stop_loss), d.notional / fill_px * 1.05) * filled_ratio
                    pos = open_position(prep.symbol, market, d.direction, fill_px, qty, d.stop_loss,
                                        d.take_profit_levels, cfg.risk.tp_fractions[:len(d.take_profit_levels)],
                                        d.strategy, str(times.iloc[i]), i, d.max_hold_bars, d.expected_hold_bars,
                                        d.confidence, d.market_regime, d.signal_id, d.leverage, d.liquidation_price)
                    entry_fee = fill_px * qty * fee_rate
                    pos_meta = {"gross": 0.0, "fees": entry_fee, "funding": 0.0, "exits": [],
                                "risk_amount": abs(fill_px - d.stop_loss) * qty, "ai": d.ai_probability}
                    cash_pnl -= entry_fee
                    stats["executed"] += 1
                pending = None
            elif pending is not None and age >= d.expiry_bars:
                stats["expired"] += 1
                pending = None

        # ---- 2) açık pozisyonu yönet
        if pos is not None:
            row = f.iloc[i]
            # funding (futures): 00/08/16 UTC kesimleri bu mum içinde mi?
            if market != "SPOT":
                t0, t1 = times.iloc[i], times.iloc[i] + pd.Timedelta(seconds=sec)
                for hh in FUNDING_HOURS:
                    ft = t0.normalize() + pd.Timedelta(hours=hh)
                    if t0 < ft <= t1:
                        rate = funding_rate_col[i] if np.isfinite(funding_rate_col[i]) else c.assumed_funding_rate_8h
                        pay = cl[i] * pos.qty * rate * pos.sign
                        pos_meta["funding"] += pay
                        cash_pnl -= pay
            actions = update_on_bar(pos, row, cfg.risk, bar_open=o[i])
            for a in actions:
                close_part(a.qty, a.price, a.reason, i, a.note)
            if pos.qty <= 1e-12:
                finalize(i, actions[-1].reason if actions else ExitReason.SL_HIT)
            elif use_decay_exits and exit_mode not in ("fixed_tp_sl", "atr_tp_sl", "trailing_only", "time_only",
                                                       "breakeven"):
                d_s = pos.direction.lower()
                o_s = "short" if d_s == "long" else "long"
                conf = float(prep.totals[f"score_{d_s}"].iat[i])
                opp = float(prep.totals[f"score_{o_s}"].iat[i])
                opp_cand = any(prep.candidates[k][o_s][i] for k in prep.candidates)
                st = engine.by_key.get(pos.strategy)
                forbidden = st is not None and Regime(regimes[i]) in st.spec.forbidden_for(pos.direction)
                act = evaluate_signal_decay(pos, conf if exit_mode != "regime_exit" else None, cfg.risk,
                                            opp if opp_cand and exit_mode != "regime_exit" else None,
                                            forbidden if exit_mode != "signal_reversal" else False, cl[i])
                if act is not None:
                    close_part(pos.qty, cl[i], act.reason, i, act.note)
                    pos.qty = 0.0
                    finalize(i, act.reason)

        # ---- 3) mark-to-market
        unreal = 0.0
        if pos is not None:
            unreal = (cl[i] - pos.entry_price) * pos.sign * pos.qty
        equity = eq0 + cash_pnl + unreal
        peak = max(peak, equity)
        eq_curve[i] = equity

        # ---- 4) yeni karar (kapanışta)
        if pos is None and pending is None and i < end - 1:
            ps = PortfolioState(equity=equity, peak_equity=peak, day_start_equity=day_start,
                                consecutive_losses=consecutive_losses)
            d = engine.decide(prep, i, equity=equity, portfolio=ps)
            stats["decisions"] += 1
            if d.signal_status != "SCANNING":
                stats["candidates"] += 1
            for r in d.no_trade_reasons:
                stats["no_trade"][r] = stats["no_trade"].get(r, 0) + 1
            if record_decisions and d.signal_status != "SCANNING":
                decisions.append({"bar": i, "time": str(times.iloc[i]), "direction": d.direction,
                                  "status": d.signal_status, "strategy": d.strategy, "confidence": d.confidence,
                                  "long_score": d.long_score, "short_score": d.short_score,
                                  "regime": d.market_regime, "ai": d.ai_probability,
                                  "reasons": ";".join(d.no_trade_reasons), "signal_id": d.signal_id})
            if d.is_trade:
                stats["confirmed"] += 1
                pending = {"d": d, "bar": i}

    if pos is not None:
        close_part(pos.qty, cl[end - 1], ExitReason.END_OF_DATA, end - 1)
        pos.qty = 0.0
        finalize(end - 1, ExitReason.END_OF_DATA)
        eq_curve[end - 1] = eq0 + cash_pnl
    eq = pd.Series(eq_curve[start:end], index=pd.DatetimeIndex(times.iloc[start:end])).ffill().fillna(eq0)
    metrics = performance_metrics(trades, eq, eq0, prep.interval)
    return IntelBacktestResult(
        symbol=prep.symbol, interval=prep.interval, market=market, trades=trades, equity=eq, metrics=metrics,
        by_regime=breakdown(trades, "regime"), by_strategy=breakdown(trades, "strategy"),
        exit_reasons=pd.Series([t.exit_reason for t in trades]).value_counts() if trades else pd.Series(dtype=int),
        holding=holding_stats(trades), signal_stats=stats, decisions=pd.DataFrame(decisions),
        assumptions=assumptions)


# ---------------------------------------------------------------------- metrikler
def _streaks(wins: list[bool]) -> tuple[int, int]:
    best_w = best_l = cur_w = cur_l = 0
    for w in wins:
        cur_w, cur_l = (cur_w + 1, 0) if w else (0, cur_l + 1)
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def performance_metrics(trades: list[IntelTrade], equity: pd.Series, initial: float, interval: str) -> dict:
    n = len(trades)
    final = float(equity.iloc[-1]) if len(equity) else initial
    bpy = 365 * 24 * 3600 / INTERVALS[interval]
    rets = equity.pct_change().dropna()
    std = float(rets.std()) if len(rets) > 1 else 0.0
    down = rets[rets < 0]
    dstd = float(np.sqrt((down ** 2).mean())) if len(down) else 0.0
    years = len(equity) / bpy if len(equity) else 0
    dd = (equity / equity.cummax() - 1) * 100 if len(equity) else pd.Series([0.0])
    net = np.array([t.net_pnl for t in trades])
    wins = net[net > 0]
    losses = net[net <= 0]
    hold = np.array([t.holding_min for t in trades])
    turnover = sum(t.notional for t in trades) / (float(equity.mean()) if len(equity) else initial)
    max_w, max_l = _streaks([x > 0 for x in net])
    gp, gl = float(wins.sum()), float(-losses.sum())
    return {
        "status": "OK" if n else "NO_TRADES",
        "Toplam işlem": n,
        "Kazanma oranı (%)": float((net > 0).mean() * 100) if n else 0.0,
        "Kâr faktörü": (gp / gl) if gl > 0 else (math.inf if gp > 0 else 0.0),
        "Beklenti (USDT/işlem)": float(net.mean()) if n else 0.0,
        "Ortalama R": float(np.mean([t.r_multiple for t in trades])) if n else 0.0,
        "Sharpe": float(rets.mean() / std * math.sqrt(bpy)) if std > 0 else 0.0,
        "Sortino": float(rets.mean() / dstd * math.sqrt(bpy)) if dstd > 0 else 0.0,
        "Maks. düşüş (%)": float(dd.min()),
        "CAGR (%)": float(((final / initial) ** (1 / years) - 1) * 100) if years > 0 and final > 0 else 0.0,
        "Ort. kazanç": float(wins.mean()) if len(wins) else 0.0,
        "Ort. kayıp": float(losses.mean()) if len(losses) else 0.0,
        "Ort. süre (dk)": float(hold.mean()) if n else 0.0,
        "Medyan süre (dk)": float(np.median(hold)) if n else 0.0,
        "Ard arda kazanç": max_w,
        "Ard arda kayıp": max_l,
        "Devir (turnover)": float(turnover),
        "Ücretler": float(sum(t.fees for t in trades)),
        "Funding": float(sum(t.funding for t in trades)),
        "Brüt PnL": float(sum(t.gross_pnl for t in trades)),
        "Net PnL": float(net.sum()) if n else 0.0,
        "Net getiri (%)": (final / initial - 1) * 100,
        "Son bakiye": final,
    }


def breakdown(trades: list[IntelTrade], key: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.to_dict() for t in trades])
    g = df.groupby(key)
    out = pd.DataFrame({
        "işlem": g.size(),
        "kazanma_%": g["net_pnl"].apply(lambda x: (x > 0).mean() * 100),
        "net_pnl": g["net_pnl"].sum(),
        "brüt_pnl": g["gross_pnl"].sum(),
        "ort_R": g["r_multiple"].mean(),
        "kâr_faktörü": g["net_pnl"].apply(lambda x: x[x > 0].sum() / -x[x <= 0].sum() if (x <= 0).any() and
                                          x[x <= 0].sum() < 0 else np.inf),
        "medyan_süre_dk": g["holding_min"].median(),
    })
    return out.sort_values("net_pnl", ascending=False)


def holding_stats(trades: list[IntelTrade]) -> dict:
    if not trades:
        return {}
    df = pd.DataFrame([t.to_dict() for t in trades])

    def pct(x):
        return {"n": int(len(x)), "ortalama": float(x.mean()), "medyan": float(x.median()),
                "p25": float(x.quantile(0.25)), "p75": float(x.quantile(0.75)), "p90": float(x.quantile(0.90)),
                "maks": float(x.max())}

    out = {"tümü": pct(df["holding_min"]), "kazananlar": pct(df.loc[df["net_pnl"] > 0, "holding_min"])
           if (df["net_pnl"] > 0).any() else {}}
    for k, g in df.groupby("strategy"):
        out[f"strateji:{k}"] = pct(g["holding_min"])
    return out


def strategy_stats_from_trades(trades: list[IntelTrade], interval: str) -> dict:
    """Karar motoruna verilecek nedensel strateji istatistikleri (yalnızca geçmiş işlemlerden)."""
    sec = INTERVALS[interval]
    out = {}
    if not trades:
        return out
    df = pd.DataFrame([t.to_dict() for t in trades])
    for key, g in df.groupby("strategy"):
        win = g[g["net_pnl"] > 0]
        loss = g[g["net_pnl"] <= 0]
        notional = g["notional"].replace(0, np.nan)
        out[key] = {
            "n": int(len(g)), "p_win": float((g["net_pnl"] > 0).mean()),
            "avg_win_pct": float((win["net_pnl"] / win["notional"] * 100).mean()) if len(win) else 0.0,
            "avg_loss_pct": float((-loss["net_pnl"] / loss["notional"] * 100).mean()) if len(loss) else 0.0,
            "median_hold_bars": int(max(1, round(g["holding_min"].median() * 60 / sec))),
            "p90_hold_bars": int(max(1, round(win["holding_min"].quantile(0.9) * 60 / sec))) if len(win) >= 10 else 0,
            "avg_r": float(g["r_multiple"].mean()),
        }
        _ = notional
    return out
