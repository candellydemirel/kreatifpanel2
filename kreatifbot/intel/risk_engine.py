"""Deterministik risk motoru.

Pozisyon boyutu, maksimum risk, kaldıraç, SL/TP, maruziyet, devre kesici ve emir
izni burada belirlenir. AI/LLM katmanı bu değerleri DEĞİŞTİREMEZ; güven skoru
riski yalnızca azaltabilir, hiçbir zaman limitlerin üstüne çıkaramaz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CircuitBreakerConfig, CostConfig, RiskConfig
from .types import Regime


def _sign(direction: str) -> int:
    return 1 if direction == "LONG" else -1


def _ok(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------- stop-loss
def atr_multiplier(cfg: RiskConfig, regime: Regime | None, high_vol: bool, base: float | None = None) -> float:
    m = base if base is not None else cfg.stop_atr_mult
    if high_vol or regime in (Regime.HIGH_VOLATILITY, Regime.PANIC):
        return max(m, cfg.stop_atr_mult_high_vol)
    if regime == Regime.LOW_VOLATILITY:
        return min(m, cfg.stop_atr_mult_low_vol)
    return m


def compute_stop(direction: str, entry: float, row: pd.Series, cfg: RiskConfig, method: str | None = None,
                 mult: float | None = None, regime: Regime | None = None) -> tuple[float, str]:
    """(stop fiyatı, kullanılan yöntem). Swing/yapı verisi yoksa ATR'ye düşer."""
    s = _sign(direction)
    method = method or cfg.stop_method
    a = float(row.get("atr", float("nan")))
    m = atr_multiplier(cfg, regime, bool(row.get("high_vol", False)), mult)
    atr_stop = entry - s * m * a if _ok(a) else float("nan")
    stop, used = float("nan"), method
    if method == "fixed_pct":
        stop = entry * (1 - s * cfg.stop_fixed_pct / 100)
    elif method in ("swing", "structure"):
        lvl = row.get("last_swing_low" if s > 0 else "last_swing_high")
        if method == "structure":
            lvl = row.get("support" if s > 0 else "resistance", lvl)
        if _ok(lvl) and (entry - float(lvl)) * s > 0:
            buffer = 0.2 * a if _ok(a) else 0.0
            stop = float(lvl) - s * buffer
            # Çok uzak/çok yakın swing stoplarını ATR ile sınırla
            if _ok(a):
                dist = abs(entry - stop)
                if dist > 4 * a or dist < 0.5 * a:
                    stop, used = atr_stop, "atr(swing sınır dışı)"
    elif method == "chandelier":
        ext = row.get("high") if s > 0 else row.get("low")
        hh = row.get("range_high" if s > 0 else "range_low", ext)
        if _ok(hh) and _ok(a):
            stop = float(hh) - s * 3 * a
            if (entry - stop) * s <= 0:
                stop, used = atr_stop, "atr(chandelier geçersiz)"
    elif method == "supertrend":
        st = row.get("supertrend")
        if _ok(st) and (entry - float(st)) * s > 0:
            stop = float(st)
    elif method == "volatility":
        rv = row.get("rv")
        if _ok(rv):
            stop = entry * (1 - s * max(float(rv) / 100 * 1.5, 0.002))
    if not _ok(stop):
        stop, used = atr_stop, "atr" if method == "atr" else f"atr({method} verisi yok)"
    return float(stop), used


# ---------------------------------------------------------------------- take-profit
def compute_targets(direction: str, entry: float, stop: float, row: pd.Series, cfg: RiskConfig,
                    method: str | None = None, rr: float | None = None,
                    ai_expected_return_pct: float | None = None) -> tuple[list[float], list[str]]:
    """Çoklu TP seviyeleri (R katları veya yönteme göre) ve uyarılar."""
    s = _sign(direction)
    risk = abs(entry - stop)
    warnings: list[str] = []
    method = method or cfg.tp_method
    levels_r = list(cfg.tp_levels_r)
    if rr is not None and rr > 0 and method == "rr":
        scale = rr / levels_r[1] if len(levels_r) > 1 and levels_r[1] else 1.0
        levels_r = [r * scale for r in levels_r]
    a = float(row.get("atr", float("nan")))
    if method == "fixed_pct":
        base = entry * cfg.tp_fixed_pct / 100
        targets = [entry + s * base * r / levels_r[-1] for r in levels_r]
    elif method == "atr" and _ok(a):
        base = cfg.tp_atr_mult * a
        targets = [entry + s * base * r / levels_r[-1] for r in levels_r]
    elif method == "ai" and _ok(ai_expected_return_pct) and ai_expected_return_pct * s > 0:
        final = entry * (1 + ai_expected_return_pct / 100)
        targets = [entry + (final - entry) * r / levels_r[-1] for r in levels_r]
    elif method == "fibonacci" and _ok(row.get("range_high")) and _ok(row.get("range_low")):
        swing = float(row["range_high"]) - float(row["range_low"])
        fibs = [0.618, 1.0, 1.272, 1.618][:len(levels_r)]
        targets = [entry + s * swing * fx for fx in fibs]
    else:
        targets = [entry + s * risk * r for r in levels_r]

    if method == "structure":
        lvl = row.get("resistance" if s > 0 else "support")
        if _ok(lvl) and (float(lvl) - entry) * s > 0:
            targets[0] = min(targets[0], float(lvl)) if s > 0 else max(targets[0], float(lvl))
    # Yakın direnç / destek uyarısı
    opp = row.get("resistance" if s > 0 else "support")
    if _ok(opp) and (float(opp) - entry) * s > 0 and abs(float(opp) - entry) < abs(targets[0] - entry):
        warnings.append(f"{'Direnç' if s > 0 else 'Destek'} yakın: {float(opp):.6g} (TP1'den önce)")
    for name in (("prev_day_high", "Önceki gün yükseği") if s > 0 else ("prev_day_low", "Önceki gün düşüğü"),):
        lvl = row.get(name[0])
        if _ok(lvl) and (float(lvl) - entry) * s > 0 and abs(float(lvl) - entry) < abs(targets[0] - entry):
            warnings.append(f"{name[1]} TP1'den önce: {float(lvl):.6g}")
    return [float(t) for t in targets], warnings


def reward_risk(entry: float, stop: float, targets: list[float], fractions: list[float]) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = sum(abs(t - entry) * w for t, w in zip(targets, fractions))
    return reward / risk


# ---------------------------------------------------------------------- trailing
def trailing_stop(direction: str, current_stop: float, row: pd.Series, extreme: float, cfg: RiskConfig,
                  method: str | None = None, aggressive: bool = False) -> float:
    """Yeni trailing stop. LONG stop asla aşağı, SHORT stop asla yukarı gitmez."""
    s = _sign(direction)
    method = method or cfg.trailing_method
    a = float(row.get("atr", float("nan")))
    mult = cfg.aggressive_trailing_atr_mult if aggressive else cfg.trailing_atr_mult
    cand = float("nan")
    if method == "percent":
        cand = extreme * (1 - s * cfg.trailing_pct / 100)
    elif method in ("atr", "volatility") and _ok(a):
        if method == "volatility" and _ok(row.get("vol_percentile")):
            mult *= 0.75 + 0.5 * float(row["vol_percentile"])
        cand = extreme - s * mult * a
    elif method == "chandelier" and _ok(a):
        cand = extreme - s * 3 * a
    elif method == "ema":
        e = row.get(f"ema{cfg.trailing_ema}")
        cand = float(e) if _ok(e) else float("nan")
    elif method == "supertrend":
        st, d = row.get("supertrend"), row.get("st_dir")
        if _ok(st) and _ok(d) and int(d) == s:
            cand = float(st)
    elif method == "swing":
        lvl = row.get("last_swing_low" if s > 0 else "last_swing_high")
        cand = float(lvl) if _ok(lvl) else float("nan")
    if not _ok(cand):
        return current_stop
    return max(current_stop, cand) if s > 0 else min(current_stop, cand)


# ---------------------------------------------------------------------- maliyet
@dataclass
class CostEstimate:
    fee_pct: float
    spread_pct: float
    slippage_pct: float
    funding_pct: float
    latency_pct: float
    total_pct: float
    assumptions: list = field(default_factory=list)


def estimate_costs(market: str, costs: CostConfig, row: pd.Series | None = None, spread_pct: float | None = None,
                   slippage_pct: float | None = None, funding_rate_8h: float | None = None,
                   hold_minutes: float = 0.0, direction: str = "LONG", maker_entry: bool = False,
                   interval_seconds: int = 300) -> CostEstimate:
    """Gidiş-dönüş (giriş+çıkış) maliyet yüzdesi."""
    assumptions = []
    if market == "SPOT":
        entry_fee = costs.spot_maker_fee_pct if maker_entry else costs.spot_taker_fee_pct
        exit_fee = costs.spot_taker_fee_pct
    else:
        entry_fee = costs.futures_maker_fee_pct if maker_entry else costs.futures_taker_fee_pct
        exit_fee = costs.futures_taker_fee_pct
    fee = entry_fee + exit_fee
    if spread_pct is None:
        spread_pct = costs.default_spread_pct
        assumptions.append(f"Spread verisi yok, varsayılan %{spread_pct} kullanıldı")
    slip = costs.slippage_pct * 2 if slippage_pct is None else slippage_pct * 2
    funding = 0.0
    if market != "SPOT" and hold_minutes > 0:
        rate = funding_rate_8h
        if rate is None or not _ok(rate):
            rate = costs.assumed_funding_rate_8h
            assumptions.append(f"Funding verisi yok, %{rate * 100:.3f}/8s varsayıldı")
        periods = hold_minutes / 480
        # LONG pozitif funding öder, SHORT alır (temkinli: yalnızca ödenen tarafı maliyet say)
        paid = rate * _sign(direction)
        funding = max(paid, 0.0) * periods * 100
    lat = 0.0
    if row is not None and _ok(row.get("atr_pct")):
        lat = float(row["atr_pct"]) * math.sqrt(costs.latency_ms / 1000 / max(interval_seconds, 1))
    total = fee + spread_pct + slip + funding + lat + costs.safety_margin_pct
    return CostEstimate(fee, spread_pct, slip, funding, lat, total, assumptions)


def expectancy(p_win: float, avg_win_pct: float, avg_loss_pct: float, cost_pct: float) -> float:
    """Beklenen değer (%): P(kazanç)×ort.kazanç − P(kayıp)×ort.kayıp − maliyet."""
    return p_win * avg_win_pct - (1 - p_win) * avg_loss_pct - cost_pct


# ---------------------------------------------------------------------- futures tasfiye
def liquidation_price(direction: str, entry: float, qty: float, leverage: float, mmr: float,
                      maint_amount: float = 0.0, wallet_balance: float | None = None) -> float:
    """Binance USDⓈ-M, tek yönlü mod, izole marj tasfiye fiyatı.

    LP = (WB + cumB − Side×Q×EP) / (Q×MMR − Side×Q)
    WB: izole marj (varsayılan Q×EP/kaldıraç), cumB: bakım tutarı (bracket), MMR: bakım marjı oranı.
    """
    side = _sign(direction)
    q = abs(qty)
    if q <= 0 or entry <= 0 or leverage <= 0:
        return float("nan")
    wb = wallet_balance if wallet_balance is not None else q * entry / leverage
    denom = q * mmr - side * q
    if denom == 0:
        return float("nan")
    lp = (wb + maint_amount - side * q * entry) / denom
    return max(lp, 0.0)


def bracket_for(notional: float, brackets: list | None, default_mmr: float) -> tuple[float, float, bool]:
    """(MMR, bakım tutarı, tahmin mi) — Binance leverageBracket verisinden."""
    for b in brackets or []:
        floor = float(b.get("notionalFloor", 0))
        cap = float(b.get("notionalCap", float("inf")))
        if floor <= notional < cap:
            return float(b["maintMarginRatio"]), float(b.get("cum", 0)), False
    return default_mmr, 0.0, True


# ---------------------------------------------------------------------- boyut ve portföy
@dataclass
class OpenExposure:
    symbol: str
    direction: str
    notional: float
    risk_amount: float


@dataclass
class SizingResult:
    ok: bool
    qty: float = 0.0
    notional: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    leverage: int = 1
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def position_size(equity: float, entry: float, stop: float, cfg: RiskConfig, confidence: float,
                  strategy_risk_mult: float = 1.0, high_vol: bool = False, market: str = "SPOT",
                  available_balance: float | None = None, open_exposure: list[OpenExposure] | None = None,
                  stage_risk_cap: float = 1.0, min_notional: float = 0.0) -> SizingResult:
    """Risk tabanlı boyut: riskTutarı = sermaye × risk%; miktar = riskTutarı / stop mesafesi."""
    res = SizingResult(ok=False, leverage=cfg.leverage if market != "SPOT" else 1)
    dist = abs(entry - stop)
    if equity <= 0 or entry <= 0 or dist <= 0 or not _ok(dist):
        res.reasons.append("Geçersiz sermaye/giriş/stop mesafesi")
        return res
    conf_mult = cfg.confidence_size_floor + (1 - cfg.confidence_size_floor) * max(0.0, min(1.0, (confidence - 50) / 40))
    conf_mult = min(conf_mult, 1.0)  # Güven asla temel riskin üstüne çıkaramaz
    vol_mult = cfg.high_vol_size_mult if high_vol else 1.0
    risk_pct = cfg.risk_per_trade_pct * min(strategy_risk_mult, 1.0) * conf_mult * vol_mult * min(stage_risk_cap, 1.0)
    risk_amount = equity * risk_pct / 100
    qty = risk_amount / dist
    lev = res.leverage
    max_notional = equity * lev
    open_exposure = open_exposure or []
    used = sum(e.notional for e in open_exposure)
    cap_total = equity * cfg.max_total_exposure_pct / 100 * (lev if market != "SPOT" else 1)
    room = cap_total - used
    if room <= 0:
        res.reasons.append("Toplam maruziyet limiti dolu")
        return res
    notional = qty * entry
    caps = [max_notional, room]
    if available_balance is not None:
        caps.append(available_balance * (lev if market != "SPOT" else 1) * 0.98)
    cap = min(caps)
    if notional > cap:
        res.warnings.append(f"Pozisyon {notional:.2f} → {cap:.2f} ile sınırlandı (bakiye/maruziyet/kaldıraç)")
        notional = cap
        qty = notional / entry
        risk_amount = qty * dist
        risk_pct = risk_amount / equity * 100
    if min_notional > 0 and 0 < notional < min_notional * 1.1:
        # Küçük hesap: Binance en küçük emir tutarına yükselt (yuvarlama payı %10), risk sınırı aşılmamalı
        target = min_notional * 1.1
        new_qty = target / entry
        new_risk_pct = new_qty * dist / equity * 100
        if target > cap:
            res.reasons.append(f"Bakiye en küçük emir tutarına ({min_notional:.2f}) yetmiyor")
            return res
        if new_risk_pct > cfg.small_account_max_risk_pct:
            res.reasons.append(f"En küçük emir tutarı için risk %{new_risk_pct:.2f} > küçük hesap sınırı "
                               f"%{cfg.small_account_max_risk_pct:.2f} (stop çok uzak)")
            return res
        res.warnings.append(f"Küçük hesap: pozisyon {notional:.2f} → {target:.2f} (Binance en küçük emir), "
                            f"risk %{new_risk_pct:.2f}")
        qty, notional, risk_amount, risk_pct = new_qty, target, new_qty * dist, new_risk_pct
    res.ok = qty > 0
    res.qty, res.notional, res.risk_amount, res.risk_pct = qty, notional, risk_amount, risk_pct
    return res


def correlated_exposure_check(symbol: str, direction: str, notional: float, open_exposure: list[OpenExposure],
                              corr: pd.DataFrame | None, cfg: RiskConfig, equity: float) -> tuple[bool, str, float]:
    """Aynı yöndeki yüksek korelasyonlu pozisyonların toplam maruziyetini sınırlar."""
    total = notional
    names = []
    for e in open_exposure:
        if e.direction != direction or e.symbol == symbol:
            continue
        c = 1.0 if corr is None else float(corr.get(symbol, {}).get(e.symbol, np.nan)) \
            if isinstance(corr, dict) else (float(corr.loc[symbol, e.symbol])
                                            if symbol in corr.index and e.symbol in corr.columns else np.nan)
        if np.isnan(c):
            c = 1.0  # Bilinmiyorsa temkinli: tam korelasyonlu say
        if c >= cfg.correlation_threshold:
            total += e.notional * c
            names.append(f"{e.symbol}({c:.2f})")
    limit = equity * cfg.max_correlated_exposure_pct / 100 * (cfg.leverage or 1)
    if names and total > limit:
        return False, f"Korelasyonlu maruziyet {total:.0f} > limit {limit:.0f}: {', '.join(names)}", total
    return True, (f"Korelasyonlu: {', '.join(names)}" if names else ""), total


def return_correlation(closes: dict[str, pd.Series], window: int = 200) -> pd.DataFrame:
    df = pd.DataFrame({k: v.pct_change() for k, v in closes.items()}).tail(window)
    return df.corr()


# ---------------------------------------------------------------------- devre kesici
@dataclass
class PortfolioState:
    equity: float
    peak_equity: float
    day_start_equity: float
    consecutive_losses: int = 0
    api_errors_recent: int = 0


def circuit_breaker(f_row: pd.Series | None, cb: CircuitBreakerConfig, risk: RiskConfig,
                    portfolio: PortfolioState | None = None, book_stats: dict | None = None,
                    liq_spike_z: float | None = None) -> list[str]:
    """Aktif devre kesici nedenleri (boşsa yeni girişlere izin var)."""
    if not cb.enabled:
        return []
    reasons = []
    if f_row is not None:
        o, c = f_row.get("open"), f_row.get("close")
        if _ok(o) and _ok(c) and o > 0 and abs(c / o - 1) * 100 >= cb.flash_move_pct:
            reasons.append(f"Flaş hareket: tek mumda %{(c / o - 1) * 100:+.1f}")
        vp = f_row.get("vol_percentile")
        if _ok(vp) and float(vp) >= cb.extreme_vol_percentile:
            reasons.append(f"Aşırı volatilite (yüzdelik {float(vp):.2f})")
    if book_stats and book_stats.get("available"):
        if book_stats.get("spread_pct", 0) >= cb.spread_explosion_pct:
            reasons.append(f"Spread patlaması %{book_stats['spread_pct']:.3f}")
        depth = book_stats.get("bid_depth_quote", 0) + book_stats.get("ask_depth_quote", 0)
        if depth < cb.min_book_depth_quote:
            reasons.append(f"Likidite çöküşü: ±%0.5 derinlik {depth:.0f}")
    if liq_spike_z is not None and _ok(liq_spike_z) and liq_spike_z >= cb.liquidation_spike_z:
        reasons.append(f"Tasfiye kaskadı (z={liq_spike_z:.1f})")
    if portfolio is not None:
        if portfolio.day_start_equity > 0:
            dd_day = (portfolio.equity / portfolio.day_start_equity - 1) * 100
            if dd_day <= -risk.max_daily_loss_pct:
                reasons.append(f"Günlük zarar limiti: %{dd_day:.2f}")
        if portfolio.peak_equity > 0:
            dd = (portfolio.equity / portfolio.peak_equity - 1) * 100
            if dd <= -risk.max_drawdown_pct:
                reasons.append(f"Maksimum düşüş limiti: %{dd:.2f}")
        if portfolio.consecutive_losses >= risk.max_consecutive_losses:
            reasons.append(f"Art arda {portfolio.consecutive_losses} kayıp")
        if portfolio.api_errors_recent >= cb.api_error_limit:
            reasons.append(f"API kararsızlığı ({portfolio.api_errors_recent} hata)")
    return reasons
