"""Karar motoru: tüm katmanları birleştirip nihai karar nesnesini üretir.

Akış: veri kalitesi → özellikler (+MTF, +türevler) → rejim → yönlendirici →
aday sinyaller → LONG/SHORT skorları → MTF çatışması → meta model → seviyeler
(SL/TP/RR) → maliyet & beklenen değer → boyut, portföy, korelasyon, tasfiye,
devre kesici → karar (LONG / SHORT / NO TRADE) + açıklama + log.

Sistem işlem açmamakta özgürdür: herhangi bir kontrol başarısızsa NO TRADE.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..i18n import tr
from ..utils import INTERVALS
from . import ENGINE_VERSION
from .config import IntelConfig
from .data_quality import QualityReport
from .features import compute_features
from .ml import MetaModel, prepare_meta_frame
from .mtf import add_mtf, bias_label, mtf_score
from .orderflow import OI_QUADRANT_NOTES, classify_funding, cvd_divergence, oi_quadrant
from .regime import classify_regime, regime_at
from .risk_engine import (
    OpenExposure, PortfolioState, bracket_for, circuit_breaker, compute_stop, compute_targets,
    correlated_exposure_check, estimate_costs, expectancy, liquidation_price, position_size, reward_risk,
)
from .router import route
from .scoring import GROUPS, group_points, group_scores, total_scores
from .strategies import IntelStrategy, build_strategies
from .types import Direction, LifecycleStage, NoTradeReason, Regime, SignalStatus

STAGE_RISK_CAP = {LifecycleStage.LIMITED_LIVE: 0.5}


@dataclass
class DecisionObject:
    signal_id: str
    symbol: str
    market: str
    timeframe: str
    direction: str = Direction.NONE.value
    signal_status: str = SignalStatus.SCANNING.value
    strategy: str = ""
    strategies_agreeing: list = field(default_factory=list)
    confidence: float = 0.0
    confidence_band: str = "NO_TRADE"
    market_regime: str = Regime.UNKNOWN.value
    entry: float = float("nan")
    stop_loss: float = float("nan")
    stop_method: str = ""
    take_profit: float = float("nan")
    take_profit_levels: list = field(default_factory=list)
    risk_reward: float = float("nan")
    risk_percent: float = 0.0
    position_size: float = 0.0
    notional: float = 0.0
    leverage: int = 1
    liquidation_price: float = float("nan")
    expected_return: float = float("nan")
    expected_value_pct: float = float("nan")
    expected_duration_min: float = float("nan")
    maximum_duration_min: float = float("nan")
    created_at: str = ""
    signal_expiry: str = ""
    ai_probability: float = float("nan")
    long_score: float = float("nan")
    short_score: float = float("nan")
    coverage: float = float("nan")
    scores: dict = field(default_factory=dict)
    mtf: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    invalidations: list = field(default_factory=list)
    no_trade_reasons: list = field(default_factory=list)
    data_sources: dict = field(default_factory=dict)
    log_lines: list = field(default_factory=list)
    explanation: str = ""
    strategy_version: str = ENGINE_VERSION
    model_version: str = "none"
    bar_index: int = -1
    max_hold_bars: int = 0
    expected_hold_bars: int = 0
    expiry_bars: int = 1

    @property
    def is_trade(self) -> bool:
        return self.direction in (Direction.LONG.value, Direction.SHORT.value) and \
            self.signal_status == SignalStatus.CONFIRMED.value

    def to_dict(self) -> dict:
        return asdict(self)

    def score_of(self, group: str) -> float | None:
        v = self.scores.get(group)
        return None if v is None else v[0]


@dataclass
class Prepared:
    f: pd.DataFrame
    regime: pd.DataFrame
    gs: pd.DataFrame
    totals: pd.DataFrame
    candidates: dict
    meta_x: pd.DataFrame
    interval: str
    symbol: str
    market: str
    missing: dict


def _fmt(x, nd=6):
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return "-"


class DecisionEngine:
    def __init__(self, cfg: IntelConfig, strategies: list[IntelStrategy] | None = None,
                 meta_model: MetaModel | None = None, strategy_stats: dict | None = None,
                 health: dict | None = None):
        self.cfg = cfg
        self.strategies = strategies if strategies is not None else build_strategies(cfg)
        self.by_key = {s.key: s for s in self.strategies}
        self.meta = meta_model
        self.stats = strategy_stats or {}
        self.health = health or {}

    # ------------------------------------------------------------------ hazırlık (vektörel)
    def prepare(self, entry_df: pd.DataFrame, symbol: str, interval: str | None = None,
                htf_frames: dict[str, pd.DataFrame] | None = None, derivatives: dict | None = None,
                btc_trend_features: pd.DataFrame | None = None, ml_probs: pd.DataFrame | None = None,
                features: pd.DataFrame | None = None) -> Prepared:
        cfg = self.cfg
        interval = interval or cfg.timeframes.entry
        f = features if features is not None else compute_features(entry_df, interval, cfg.market, derivatives)
        avail = dict(f.attrs.get("availability", {}))
        htf_feats = {}
        for role, df in (htf_frames or {}).items():
            if df is None or len(df) < 60:
                avail[f"mtf_{role}"] = "UNAVAILABLE"
                continue
            htf_feats[role] = compute_features(df, getattr(cfg.timeframes, role), cfg.market)
            avail[f"mtf_{role}"] = "OK"
        f = add_mtf(f, htf_feats)
        if btc_trend_features is not None and not btc_trend_features.empty:
            from .mtf import align_htf, htf_summary
            f["btc_bias"] = align_htf(f, htf_summary(btc_trend_features), "btc")["btc_bias"].to_numpy()
            avail["btc_context"] = "OK"
        if ml_probs is not None:
            for c in ("ml_prob_long", "ml_prob_short"):
                if c in ml_probs:
                    f[c] = ml_probs[c].to_numpy()
        reg = classify_regime(f)
        f["regime"] = reg["regime"]
        f["high_vol"] = reg["high_vol"]
        f.attrs["availability"] = avail
        gs = group_scores(f)
        totals = total_scores(gs, cfg.scoring)
        cands, missing = {}, {}
        for st in self.strategies:
            c = st.candidates(f)
            cands[st.key] = {
                "long": c["long"].to_numpy(), "short": c["short"].to_numpy(),
                "long_strength": c["long_strength"].to_numpy(), "short_strength": c["short_strength"].to_numpy(),
                "long_invalid": c["long_invalid"].to_numpy(), "short_invalid": c["short_invalid"].to_numpy(),
            }
            if c.attrs.get("missing"):
                missing[st.key] = c.attrs["missing"]
        return Prepared(f=f, regime=reg, gs=gs, totals=totals, candidates=cands, meta_x=prepare_meta_frame(f),
                        interval=interval, symbol=symbol, market=cfg.market, missing=missing)

    # ------------------------------------------------------------------ yardımcılar
    def _hold(self, st: IntelStrategy, interval: str) -> tuple[int, int, int]:
        sec = INTERVALS[interval]
        exp_m, _min_m, max_m, expiry_m = st.spec.holding_minutes()
        sc = self.cfg.strategy(st.key)
        stats = self.stats.get(st.key, {})
        to_bars = lambda m: max(1, int(round(m * 60 / sec)))  # noqa: E731
        exp_b = sc.expected_hold_bars or stats.get("median_hold_bars") or to_bars(exp_m)
        max_b = sc.max_hold_bars or stats.get("p90_hold_bars") or to_bars(max_m)
        expiry_b = sc.expiry_bars or to_bars(expiry_m)
        return int(exp_b), int(max(max_b, exp_b)), int(expiry_b)

    def stage_of(self, key: str) -> LifecycleStage:
        return LifecycleStage(self.cfg.strategy(key).stage)

    # ------------------------------------------------------------------ karar
    def decide(self, prep: Prepared, i: int = -1, equity: float = 10000.0,
               open_exposure: list[OpenExposure] | None = None, portfolio: PortfolioState | None = None,
               rules=None, book_stats: dict | None = None, live: bool = False,
               dq: QualityReport | None = None, has_position: bool = False, available_balance: float | None = None,
               correlations: pd.DataFrame | None = None, brackets: list | None = None,
               funding_now: float | None = None, news_ctx=None) -> DecisionObject:
        cfg = self.cfg
        f = prep.f
        n = len(f)
        i = i if i >= 0 else n + i
        row = f.iloc[i]
        created = pd.to_datetime(row["close_time"], utc=True) if "close_time" in f else datetime.now(timezone.utc)
        d = DecisionObject(signal_id=uuid.uuid4().hex[:16], symbol=prep.symbol, market=prep.market,
                           timeframe=prep.interval, created_at=created.isoformat(), bar_index=i,
                           entry=float(row["close"]), data_sources=dict(f.attrs.get("availability", {})))
        if self.meta is not None and self.meta.ready:
            d.model_version = self.meta.version
        log = d.log_lines
        log.append(f"{created:%Y-%m-%d %H:%M:%S} {prep.symbol} {prep.market} {prep.interval}")

        def no_trade(reason: NoTradeReason, msg: str, status: SignalStatus = SignalStatus.NO_TRADE):
            d.no_trade_reasons.append(reason.value)
            d.reasons.append(msg)
            d.signal_status = status.value
            d.direction = Direction.NONE.value
            log.append(f"Karar: İŞLEM YOK ({tr(reason.value)}) — {msg}")
            d.explanation = self.explain(d)
            return d

        # 1) Veri kalitesi
        if dq is not None and not dq.ok:
            d.warnings += dq.warnings
            return no_trade(NoTradeReason.DATA_QUALITY_FAILURE, "; ".join(dq.critical))
        if dq is not None:
            d.warnings += dq.warnings
        if rules is None and live:
            return no_trade(NoTradeReason.SYMBOL_FILTER_UNKNOWN, "Binance sembol filtreleri bilinmiyor")
        if rules is not None and not rules.tradable:
            return no_trade(NoTradeReason.DATA_QUALITY_FAILURE, f"Sembol durumu {rules.status}")

        # 2) Rejim
        fr_rate = row.get("funding_rate")
        if funding_now is not None:
            fr_rate = funding_now
        f_class = classify_funding(fr_rate if fr_rate is not None and pd.notna(fr_rate) else None, cfg.funding)
        btc_b = row.get("btc_bias") if "btc_bias" in f else None
        rr = regime_at(f, prep.regime, i, btc_bias=btc_b, funding_class=f_class,
                       oi_change_pct=row.get("oi_change_pct") if pd.notna(row.get("oi_change_pct")) else None)
        regime = rr.regime
        d.market_regime = regime.value
        d.reasons += [f"Rejim {regime.value}: " + "; ".join(rr.reasons)]
        log.append(f"Rejim: {tr(regime.value)} (ADX {rr.adx:.1f}, vol%ile {rr.vol_percentile:.2f})")

        # 3) Yönlendirici
        rt = route(regime, self.strategies, cfg, prep.market, self.health, live=live)
        if rt.blocked_reason:
            return no_trade(NoTradeReason.REGIME_INCOMPATIBLE if regime == Regime.PANIC else
                            NoTradeReason.INSUFFICIENT_DATA, rt.blocked_reason)

        # 4) Aday sinyaller
        cands = {"LONG": [], "SHORT": []}
        for direction in ("LONG", "SHORT"):
            key = direction.lower()
            for sk in rt.active[direction]:
                c = prep.candidates[sk]
                if c[key][i]:
                    cands[direction].append((sk, float(c[f"{key}_strength"][i])))
        if not cands["LONG"] and not cands["SHORT"]:
            d.signal_status = SignalStatus.SCANNING.value
            d.reasons.append(f"Aktif stratejiler: LONG {len(rt.active['LONG'])}, SHORT {len(rt.active['SHORT'])}; "
                             "bu mumda aday sinyal yok")
            d.no_trade_reasons.append(NoTradeReason.NO_CANDIDATE.value)
            log.append("Karar: NO TRADE (aday yok)")
            d.explanation = self.explain(d)
            return d

        # 5) Skorlar
        tot = prep.totals.iloc[i]
        long_s, short_s = float(tot["score_long"]), float(tot["score_short"])
        d.long_score, d.short_score = long_s, short_s
        if cands["LONG"] and cands["SHORT"]:
            direction = "LONG" if long_s >= short_s else "SHORT"
        else:
            direction = "LONG" if cands["LONG"] else "SHORT"
        my, other = (long_s, short_s) if direction == "LONG" else (short_s, long_s)
        coverage = float(tot[f"coverage_{direction.lower()}"])
        d.coverage = coverage
        d.scores = group_points(prep.gs.iloc[i], direction, cfg.scoring)
        primary_key, primary_strength = max(cands[direction], key=lambda t: t[1])
        primary = self.by_key[primary_key]
        d.strategy = primary_key
        d.strategies_agreeing = [k for k, _ in cands[direction]]
        d.strategy_version = f"{ENGINE_VERSION}/{primary.spec.version}"

        # Uyum bonusu: farklı bilgi kaynaklarından gelen adaylar (aynı kaynak tek sayılır)
        groups = {self.by_key[k].spec.info_group for k, _ in cands[direction]} - {primary.spec.info_group}
        bonus = min(cfg.scoring.confluence_bonus_max, cfg.scoring.confluence_bonus_per_family * len(groups))
        confidence = float(np.clip((my if np.isfinite(my) else 0) + bonus, 0, 100))
        d.confidence = round(confidence, 1)
        d.confidence_band = cfg.scoring.band(confidence)
        d.direction = direction
        d.signal_status = SignalStatus.CANDIDATE.value
        log.append(f"{direction} adayı: {primary.name} (+{len(cands[direction]) - 1} uyumlu, "
                   f"bağımsız kaynak bonusu +{bonus:.0f})")
        for g in GROUPS:
            pts, w = d.scores[g]
            if w > 0:
                log.append(f"  {tr(g):13s}: {('Veri yok' if pts is None else f'{pts:.1f}/{w:.0f}')}")
        log.append(f"  Toplam: {confidence:.1f}/100 (LONG {long_s:.1f} / SHORT {short_s:.1f}, kapsam {coverage:.0%})")

        # MTF
        mscore, mdetail = mtf_score(row, direction, cfg.scoring.mtf_weights)
        d.mtf = {role: bias_label(v) for role, v in mdetail.items()}
        d.mtf["entry"] = bias_label(row.get("entry_bias"))
        d.mtf["score"] = None if mscore is None else round(mscore, 3)

        # Haber bağlamı (haber tek başına işlem açtırmaz; yalnızca risk filtresi)
        if news_ctx is not None:
            if news_ctx.items:
                d.reasons.append(f"Haber: son {len(news_ctx.items)} haber, ortalama duyarlılık "
                                 f"{news_ctx.sentiment:+.2f}")
            if (direction == "LONG" and news_ctx.block_long) or (direction == "SHORT" and news_ctx.block_short):
                return no_trade(NoTradeReason.NEWS_RISK, "; ".join(news_ctx.reasons[:3]))

        # Çatışma / eşik kontrolleri
        threshold = cfg.scoring.min_trade_score + rt.threshold_bonus
        if coverage < cfg.scoring.min_coverage:
            return no_trade(NoTradeReason.DATA_UNAVAILABLE,
                            f"Veri kapsamı %{coverage * 100:.0f} < %{cfg.scoring.min_coverage * 100:.0f}")
        if cands["LONG"] and cands["SHORT"] and min(long_s, short_s) >= threshold:
            return no_trade(NoTradeReason.CONFLICT, f"Her iki yön de güçlü (LONG {long_s:.0f}, SHORT {short_s:.0f})")
        if my - other < cfg.scoring.conflict_margin:
            return no_trade(NoTradeReason.CONFLICT,
                            f"Yönler arası fark yetersiz ({my:.0f} vs {other:.0f}, gerekli fark "
                            f"{cfg.scoring.conflict_margin:.0f})")
        if mscore is not None and mscore <= cfg.scoring.mtf_conflict_threshold:
            return no_trade(NoTradeReason.CONFLICTING_TIMEFRAMES,
                            f"Üst zaman dilimleri ters yönde (MTF {mscore:+.2f})")
        if confidence < threshold:
            return no_trade(NoTradeReason.LOW_CONFIDENCE, f"Güven {confidence:.0f} < eşik {threshold:.0f}")
        if direction == "SHORT" and prep.market == "SPOT" and not cfg.backtest.allow_short_on_spot:
            return no_trade(NoTradeReason.SHORT_NOT_SUPPORTED, "Spot piyasada SHORT açılamaz")
        if has_position:
            return no_trade(NoTradeReason.POSITION_EXISTS, "Bu sembolde zaten açık pozisyon var")

        # 6) Meta model
        sign = 1 if direction == "LONG" else -1
        p_ai = float("nan")
        if self.meta is not None and self.meta.ready and cfg.ml.enabled:
            p_ai = float(self.meta.predict(prep.meta_x, np.array([i]), np.array([sign]))[0])
            d.ai_probability = round(p_ai, 4)
            log.append(f"Meta model: P(başarı) = {p_ai:.2f}")
            if abs(p_ai - 0.5) < cfg.ml.uncertainty_band:
                return no_trade(NoTradeReason.AI_UNCERTAIN, f"Model belirsiz (P={p_ai:.2f})")
            if p_ai < cfg.ml.meta_threshold:
                return no_trade(NoTradeReason.META_MODEL_REJECT,
                                f"Meta model olasılığı {p_ai:.2f} < {cfg.ml.meta_threshold:.2f}")
        elif live and cfg.ml.enabled and self.meta is not None and not self.meta.ready:
            d.warnings.append("AI modeli hazır değil: deterministik moda geçildi")

        # 7) Seviyeler
        entry = float(row["close"])
        stop, stop_used = compute_stop(direction, entry, row, cfg.risk, primary.spec.stop_method,
                                       primary.spec.stop_mult if primary.spec.stop_method == "atr" else None, regime)
        if not math.isfinite(stop) or (entry - stop) * sign <= 0:
            return no_trade(NoTradeReason.POSITION_SIZE_FAILURE, "Geçerli stop hesaplanamadı (ATR yok)")
        exp_ret_ai = self.meta.expected_return(p_ai) if (self.meta is not None and self.meta.ready) else None
        targets, tp_warn = compute_targets(direction, entry, stop, row, cfg.risk, None, primary.spec.tp_rr,
                                           exp_ret_ai)
        d.warnings += tp_warn
        d.entry, d.stop_loss, d.stop_method = entry, stop, stop_used
        d.take_profit_levels = [round(t, 10) for t in targets]
        d.take_profit = targets[-1]
        rrv = reward_risk(entry, stop, targets, cfg.risk.tp_fractions)
        d.risk_reward = round(rrv, 2)
        exp_b, max_b, expiry_b = self._hold(primary, prep.interval)
        sec = INTERVALS[prep.interval]
        d.expected_hold_bars, d.max_hold_bars, d.expiry_bars = exp_b, max_b, expiry_b
        if self.meta is not None and self.meta.ready and math.isfinite(self.meta.bars_win_median):
            d.expected_duration_min = self.meta.bars_win_median * sec / 60
        else:
            d.expected_duration_min = exp_b * sec / 60
        d.maximum_duration_min = max_b * sec / 60
        d.signal_expiry = (created + timedelta(seconds=expiry_b * sec)).isoformat()
        d.invalidations = [primary.spec.invalidation or "Ters yönde kurulum",
                           f"Stop {_fmt(stop)} ({stop_used})",
                           f"Sinyal {expiry_b} mum içinde gerçekleşmezse SIGNAL_EXPIRED"]
        if rrv < cfg.risk.min_rr:
            return no_trade(NoTradeReason.POOR_RR, f"R:R {rrv:.2f} < minimum {cfg.risk.min_rr}")

        # 8) Maliyet ve beklenen değer
        spread = book_stats.get("spread_pct") if book_stats and book_stats.get("available") else None
        fr_for_cost = funding_now if funding_now is not None else (fr_rate if pd.notna(fr_rate) else None)
        cost = estimate_costs(prep.market, cfg.costs, row, spread, None, fr_for_cost, d.expected_duration_min,
                              direction, interval_seconds=sec)
        d.costs = asdict(cost)
        d.warnings += cost.assumptions
        risk_pct_move = abs(entry - stop) / entry * 100
        reward_pct_move = rrv * risk_pct_move
        p_win, src = None, ""
        if math.isfinite(p_ai):
            p_win, src = p_ai, "meta model"
        elif self.stats.get(primary_key, {}).get("n", 0) >= 30:
            st = self.stats[primary_key]
            p_win, src = st["p_win"], f"strateji geçmişi (n={st['n']})"
            if st.get("avg_win_pct") and st.get("avg_loss_pct"):
                reward_pct_move, risk_pct_move = st["avg_win_pct"], st["avg_loss_pct"]
        if p_win is not None:
            ev = expectancy(p_win, reward_pct_move, risk_pct_move, cost.total_pct)
            d.expected_value_pct = round(ev, 4)
            d.expected_return = round(p_win * reward_pct_move - (1 - p_win) * risk_pct_move, 4)
            log.append(f"EV ({src}): {ev:+.3f}% (P={p_win:.2f}, ödül %{reward_pct_move:.2f}, "
                       f"risk %{risk_pct_move:.2f}, maliyet %{cost.total_pct:.3f})")
            if ev <= 0:
                return no_trade(NoTradeReason.NEGATIVE_EXPECTANCY, f"Beklenen değer {ev:+.3f}% ≤ 0 (maliyetler sonrası)")
        else:
            d.warnings.append("Başarı olasılığı tahmini yok (model eğitilmemiş / yetersiz geçmiş): EV hesaplanamadı")
            if live and cfg.risk.require_probability_for_live:
                return no_trade(NoTradeReason.AI_UNCERTAIN, "Canlı işlemde olasılık tahmini zorunlu (ayar)")
            if reward_pct_move * cfg.risk.tp_fractions[0] <= cost.total_pct * 2:
                return no_trade(NoTradeReason.EDGE_BELOW_COSTS,
                                f"İlk hedef (%{reward_pct_move * cfg.risk.tp_fractions[0]:.3f}) maliyetlerin "
                                f"(%{cost.total_pct:.3f}) 2 katını karşılamıyor")
        log.append(f"R:R {rrv:.2f}: PASS; maliyet %{cost.total_pct:.3f}: PASS")

        # 9) Devre kesici ve portföy
        cb = circuit_breaker(row, cfg.circuit_breaker, cfg.risk, portfolio, book_stats)
        if cb:
            return no_trade(NoTradeReason.CIRCUIT_BREAKER, "; ".join(cb))
        open_exposure = open_exposure or []
        if len(open_exposure) >= cfg.risk.max_open_positions:
            return no_trade(NoTradeReason.EXCESSIVE_PORTFOLIO_RISK,
                            f"Maksimum açık pozisyon ({cfg.risk.max_open_positions}) dolu")
        stage = self.stage_of(primary_key)
        if live and stage not in (LifecycleStage.LIMITED_LIVE, LifecycleStage.FULL_LIVE):
            return no_trade(NoTradeReason.STRATEGY_NOT_LIVE, f"{primary.name} aşaması {stage.value}")
        sz = position_size(equity, entry, stop, cfg.risk, confidence, primary.spec.risk_multiplier *
                           cfg.strategy(primary_key).risk_multiplier, rr.high_vol or rt.size_mult < 1,
                           prep.market, available_balance, open_exposure, STAGE_RISK_CAP.get(stage, 1.0))
        d.warnings += sz.warnings
        if not sz.ok:
            return no_trade(NoTradeReason.POSITION_SIZE_FAILURE, "; ".join(sz.reasons) or "Boyut hesaplanamadı")
        qty = sz.qty
        if rules is not None:
            chk = rules.check_order(qty, entry, market_order=True)
            if not chk.ok:
                return no_trade(NoTradeReason.POSITION_SIZE_FAILURE, "Binance filtreleri: " + "; ".join(chk.errors))
            qty = float(chk.quantity)
        ok_corr, corr_msg, _ = correlated_exposure_check(prep.symbol, direction, qty * entry, open_exposure,
                                                         correlations, cfg.risk, equity)
        if not ok_corr:
            return no_trade(NoTradeReason.CORRELATED_EXPOSURE, corr_msg)
        if corr_msg:
            d.warnings.append(corr_msg)
        d.position_size, d.notional = qty, qty * entry
        d.risk_percent = round(abs(entry - stop) * qty / equity * 100, 4)
        d.leverage = sz.leverage

        # Futures tasfiye: stop'tan önce tasfiye olmamalı
        if prep.market != "SPOT":
            mmr, cum, estimated = bracket_for(qty * entry, brackets, cfg.risk.default_maintenance_margin_rate)
            lp = liquidation_price(direction, entry, qty, sz.leverage, mmr, cum)
            d.liquidation_price = lp
            if estimated:
                d.warnings.append(f"Bakım marjı oranı tahmini (%{mmr * 100:.2f}); Binance bracket verisi alınamadı")
            buf = cfg.risk.liquidation_buffer_atr * float(row.get("atr", 0) or 0)
            if math.isfinite(lp) and ((lp >= stop - buf) if sign > 0 else (lp <= stop + buf)):
                return no_trade(NoTradeReason.LIQUIDATION_RISK,
                                f"Tasfiye fiyatı {_fmt(lp)} stop'a ({_fmt(stop)}) çok yakın/önde; kaldıracı düşürün")
        log.append(f"Risk kontrolü: GEÇTİ (risk %{d.risk_percent:.3f}, boyut {qty:.6g}, nominal {d.notional:.2f})")

        d.signal_status = SignalStatus.CONFIRMED.value
        d.reasons.append(f"{primary.name}: {primary.spec.entry}")
        self._add_factor_reasons(d, row, direction)
        log.append(f"Karar: {tr(direction)}")
        d.explanation = self.explain(d)
        return d

    # ------------------------------------------------------------------ açıklama
    def _add_factor_reasons(self, d: DecisionObject, row: pd.Series, direction: str):
        s = 1 if direction == "LONG" else -1
        if pd.notna(row.get("vwap")):
            d.reasons.append(f"VWAP: fiyat VWAP'ın {'üstünde' if row['close'] > row['vwap'] else 'altında'}")
        if pd.notna(row.get("cvd_slope")):
            d.reasons.append(f"CVD: {'pozitif' if row['cvd_slope'] > 0 else 'negatif'} eğim — "
                             f"{tr(cvd_divergence(row.get('roc', np.nan), row['cvd_slope']))}")
        if pd.notna(row.get("oi_change_pct")):
            q = oi_quadrant(row.get("roc", np.nan), row["oi_change_pct"])
            d.reasons.append(f"OI: %{row['oi_change_pct']:+.2f} → {tr(q)} ({OI_QUADRANT_NOTES.get(q, '')})")
        fc = classify_funding(row.get("funding_rate") if pd.notna(row.get("funding_rate")) else None, self.cfg.funding)
        d.reasons.append(f"Fonlama oranı: {tr(fc)}")
        if row.get("pattern_bull_count", 0) and s > 0:
            d.reasons.append("Mum formasyonu: yükseliş formasyonu (yalnızca uyum faktörü)")
        if row.get("pattern_bear_count", 0) and s < 0:
            d.reasons.append("Mum formasyonu: düşüş formasyonu (yalnızca uyum faktörü)")

    def explain(self, d: DecisionObject) -> str:
        lines = [f"{d.symbol} ({tr(d.market)}, {d.timeframe})",
                 f"Karar: {tr(d.direction) if d.is_trade else 'İŞLEM YOK'}  [{tr(d.signal_status)}]"]
        if d.no_trade_reasons:
            lines.append("Neden: " + ", ".join(tr(r) for r in d.no_trade_reasons))
        if d.strategy:
            lines.append(f"Birincil strateji: {self.by_key[d.strategy].name if d.strategy in self.by_key else d.strategy}"
                         f"  | Uyumlu: {', '.join(d.strategies_agreeing)}")
        if d.confidence:
            lines.append(f"Güven: {d.confidence:.0f}/100 ({d.confidence_band})  LONG {d.long_score:.0f} / "
                         f"SHORT {d.short_score:.0f}  kapsam %{(d.coverage or 0) * 100:.0f}")
        lines.append(f"Piyasa rejimi: {tr(d.market_regime)}")
        if d.scores:
            lines.append("")
            lines.append("Grup puanları:")
            for g, (pts, w) in d.scores.items():
                if w > 0:
                    lines.append(f"  {tr(g):13s} {'Veri yok' if pts is None else f'{pts:5.1f} / {w:.0f}'}")
        if d.mtf:
            lines.append("")
            lines.append("Zaman dilimleri: " + ", ".join(f"{k}: {tr(v)}" for k, v in d.mtf.items() if k != "score"))
        if math.isfinite(d.ai_probability):
            lines.append(f"AI: başarı olasılığı %{d.ai_probability * 100:.0f} (model {d.model_version})")
        if d.is_trade:
            lines += ["", f"Giriş: {_fmt(d.entry)}", f"SL: {_fmt(d.stop_loss)} ({d.stop_method})"]
            for k, t in enumerate(d.take_profit_levels):
                lines.append(f"TP{k + 1}: {_fmt(t)} (%{self.cfg.risk.tp_fractions[k] * 100:.0f})")
            lines += [f"R:R 1:{d.risk_reward:.2f}", f"Risk: %{d.risk_percent:.3f} sermaye, boyut {d.position_size:.6g}"
                      f" (nominal {d.notional:.2f}, kaldıraç {d.leverage}x)",
                      f"Beklenen süre: ~{d.expected_duration_min:.0f} dk, maksimum {d.maximum_duration_min:.0f} dk",
                      f"Sinyal geçerlilik sonu: {d.signal_expiry}"]
            if math.isfinite(d.expected_value_pct):
                lines.append(f"Beklenen değer (maliyet sonrası): %{d.expected_value_pct:+.3f}")
            if math.isfinite(d.liquidation_price):
                lines.append(f"Tahmini tasfiye fiyatı: {_fmt(d.liquidation_price)}")
        if d.reasons:
            lines += ["", "Gerekçeler:"] + [f"  • {r}" for r in d.reasons]
        if d.invalidations:
            lines += ["Geçersizleşme:"] + [f"  • {r}" for r in d.invalidations]
        if d.warnings:
            lines += ["Uyarılar:"] + [f"  ⚠ {w}" for w in d.warnings]
        unavailable = [k for k, v in d.data_sources.items() if str(v).startswith("UNAVAILABLE")]
        if unavailable:
            lines.append("Eksik veri: " + ", ".join(unavailable))
        lines.append("Not: Bu bir olasılık değerlendirmesidir; kâr garantisi yoktur.")
        return "\n".join(lines)
