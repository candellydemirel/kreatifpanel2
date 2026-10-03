"""Pozisyon yöneticisi: triple barrier, çoklu TP, breakeven, kâr kilidi, trailing,
sinyal zayıflaması, rejim değişimi ve izleme metrikleri.

Mum içi sıra (temkinli): önce SL, sonra TP seviyeleri; aynı mumda ikisi de
tetiklenirse SL kabul edilir. Dikey (zaman) bariyeri mum kapanışında kontrol edilir.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import RiskConfig
from .risk_engine import trailing_stop
from .types import ExitReason


@dataclass
class TPLevel:
    price: float
    fraction: float
    hit: bool = False


@dataclass
class ExitAction:
    reason: ExitReason
    qty: float
    price: float
    full: bool
    note: str = ""


@dataclass
class ManagedPosition:
    symbol: str
    market: str
    direction: str
    entry_price: float
    qty: float                   # açık miktar
    initial_qty: float
    stop_loss: float
    initial_stop: float
    tp_levels: list
    strategy: str
    opened_at: str
    opened_bar: int = 0
    max_hold_bars: int = 0
    expected_hold_bars: int = 0
    initial_confidence: float = 0.0
    confidence: float = 0.0
    regime_at_entry: str = ""
    signal_id: str = ""
    leverage: int = 1
    liquidation_price: float = float("nan")
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    realized_pnl: float = 0.0          # kısmi çıkışlardan (ücret hariç brüt)
    stage: str = "INITIAL"             # INITIAL → BREAKEVEN → PROFIT_LOCK → TRAILING → AGGRESSIVE
    extreme: float = 0.0               # LONG için en yüksek, SHORT için en düşük
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    confidence_history: list = field(default_factory=list)
    status: str = "ACTIVE"
    position_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # GUI/Telegram uyumluluğu (klasik Position ile aynı alan adları)
    @property
    def take_profit(self) -> float:
        open_tps = [t.price for t in self.tp_levels if not t.hit]
        return open_tps[-1] if open_tps else 0.0

    @property
    def cost(self) -> float:
        return self.entry_price * self.qty / max(self.leverage, 1) if self.market != "SPOT" else self.entry_price * self.qty

    @property
    def highest(self) -> float:
        return self.extreme

    @property
    def sign(self) -> int:
        return 1 if self.direction == "LONG" else -1

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def r_multiple(self, price: float) -> float:
        rpu = self.risk_per_unit
        return (price - self.entry_price) * self.sign / rpu if rpu > 0 else 0.0

    def unrealized(self, price: float) -> tuple[float, float]:
        pnl = (price - self.entry_price) * self.sign * self.qty
        base = self.entry_price * self.qty / (self.leverage if self.market != "SPOT" else 1)
        return pnl, (pnl / base * 100 if base else 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tp_levels"] = [asdict(t) for t in self.tp_levels]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ManagedPosition":
        d = dict(d)
        d["tp_levels"] = [TPLevel(**t) for t in d.get("tp_levels", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def open_position(symbol: str, market: str, direction: str, entry: float, qty: float, stop: float,
                  targets: list[float], fractions: list[float], strategy: str, opened_at: str, bar: int,
                  max_hold_bars: int, expected_hold_bars: int, confidence: float, regime: str, signal_id: str = "",
                  leverage: int = 1, liquidation_price: float = float("nan")) -> ManagedPosition:
    return ManagedPosition(
        symbol=symbol, market=market, direction=direction, entry_price=entry, qty=qty, initial_qty=qty,
        stop_loss=stop, initial_stop=stop, tp_levels=[TPLevel(t, w) for t, w in zip(targets, fractions)],
        strategy=strategy, opened_at=opened_at, opened_bar=bar, max_hold_bars=max_hold_bars,
        expected_hold_bars=expected_hold_bars, initial_confidence=confidence, confidence=confidence,
        regime_at_entry=regime, signal_id=signal_id, leverage=leverage, liquidation_price=liquidation_price,
        extreme=entry,
    )


def _round_down_qty(q: float, step: float | None) -> float:
    if not step or step <= 0:
        return q
    return float(np.floor(q / step + 1e-9) * step)


def update_on_bar(pos: ManagedPosition, row: pd.Series, cfg: RiskConfig, bar_open: float | None = None,
                  qty_step: float | None = None, trailing_method: str | None = None,
                  count_bar: bool = True) -> list[ExitAction]:
    """Bir mumun yüksek/düşük/kapanış verisiyle pozisyonu günceller ve çıkış eylemlerini döndürür.

    count_bar=False: canlıda anlık fiyat kontrolü (süre sayacı ve zaman bariyeri ilerlemez).
    """
    actions: list[ExitAction] = []
    s = pos.sign
    hi, lo, close = float(row["high"]), float(row["low"]), float(row["close"])
    op = float(bar_open if bar_open is not None else row.get("open", close))
    adverse = lo if s > 0 else hi
    favorable = hi if s > 0 else lo

    # 0) Futures tasfiye (stop'tan önce gelirse)
    if pos.market != "SPOT" and np.isfinite(pos.liquidation_price):
        liq_hit = adverse <= pos.liquidation_price if s > 0 else adverse >= pos.liquidation_price
        stop_first = (pos.stop_loss >= pos.liquidation_price) if s > 0 else (pos.stop_loss <= pos.liquidation_price)
        if liq_hit and not stop_first:
            actions.append(ExitAction(ExitReason.EMERGENCY_EXIT, pos.qty, pos.liquidation_price, True,
                                      "Tasfiye fiyatına ulaşıldı"))
            pos.qty = 0.0
            return actions

    # 1) Stop-loss / trailing stop (temkinli: önce)
    hit_sl = adverse <= pos.stop_loss if s > 0 else adverse >= pos.stop_loss
    if hit_sl:
        px = min(op, pos.stop_loss) if s > 0 else max(op, pos.stop_loss)
        reason = ExitReason.TRAILING_STOP if pos.stage in ("TRAILING", "AGGRESSIVE") else ExitReason.SL_HIT
        note = "Breakeven stop" if pos.stage in ("BREAKEVEN", "PROFIT_LOCK") else ""
        actions.append(ExitAction(reason, pos.qty, px, True, note))
        pos.qty = 0.0
        return actions

    # 1b) Kesin kâr hedefi: pozisyon yüzde X kâra ulaşınca koşulsuz tamamı satılır
    hard = float(getattr(cfg, "hard_take_profit_pct", 0) or 0)
    if hard > 0:
        lev = max(1, int(pos.leverage or 1)) if pos.market != "SPOT" else 1
        target = pos.entry_price * (1 + s * hard / 100 / lev)
        if (favorable >= target) if s > 0 else (favorable <= target):
            px = max(op, target) if s > 0 else min(op, target)
            actions.append(ExitAction(ExitReason.TP_HIT, pos.qty, px, True, f"%{hard:g} kâr hedefi"))
            pos.qty = 0.0
            return actions

    # 2) Kâr al seviyeleri (kısmi)
    for k, tp in enumerate(pos.tp_levels):
        if tp.hit:
            continue
        reached = favorable >= tp.price if s > 0 else favorable <= tp.price
        if not reached:
            break
        tp.hit = True
        last = all(t.hit for t in pos.tp_levels)
        q = pos.qty if last else min(pos.qty, _round_down_qty(pos.initial_qty * tp.fraction, qty_step))
        if q <= 0:
            continue
        px = max(op, tp.price) if s > 0 else min(op, tp.price)
        actions.append(ExitAction(ExitReason.TP_HIT, q, px, last or q >= pos.qty - 1e-12, f"TP{k + 1}"))
        pos.qty -= q
        if pos.qty <= 1e-12:
            pos.qty = 0.0
            return actions

    # 3) Aşama yönetimi (breakeven → kâr kilidi → trailing → agresif trailing)
    pos.extreme = max(pos.extreme, hi) if s > 0 else min(pos.extreme, lo)
    r_best = pos.r_multiple(pos.extreme)
    pos.mfe_r = max(pos.mfe_r, r_best)
    pos.mae_r = min(pos.mae_r, pos.r_multiple(adverse))
    hits = sum(t.hit for t in pos.tp_levels)
    new_stop = pos.stop_loss
    if r_best >= cfg.breakeven_at_r or hits >= 1:
        be = pos.entry_price
        new_stop = max(new_stop, be) if s > 0 else min(new_stop, be)
        pos.stage = "BREAKEVEN" if pos.stage == "INITIAL" else pos.stage
    if r_best >= cfg.profit_lock_at_r:
        lock = pos.entry_price + s * cfg.profit_lock_r * pos.risk_per_unit
        new_stop = max(new_stop, lock) if s > 0 else min(new_stop, lock)
        if pos.stage in ("INITIAL", "BREAKEVEN"):
            pos.stage = "PROFIT_LOCK"
    if r_best >= cfg.trailing_at_r or hits >= 2:
        aggressive = hits >= 3
        new_stop = trailing_stop(pos.direction, new_stop, row, pos.extreme, cfg, trailing_method, aggressive)
        pos.stage = "AGGRESSIVE" if aggressive else "TRAILING"
    # Stop asla geriye gitmez
    pos.stop_loss = max(pos.stop_loss, new_stop) if s > 0 else min(pos.stop_loss, new_stop)

    # 4) Dikey bariyer (süre)
    if not count_bar:
        return actions
    pos.bars_held += 1
    if pos.max_hold_bars and pos.bars_held >= pos.max_hold_bars:
        actions.append(ExitAction(ExitReason.TIME_EXPIRY, pos.qty, close, True,
                                  f"Maksimum süre {pos.max_hold_bars} mum"))
        pos.qty = 0.0
    return actions


def evaluate_signal_decay(pos: ManagedPosition, new_confidence: float | None, cfg: RiskConfig,
                          reversal_confidence: float | None = None, regime_forbidden: bool = False,
                          price: float | None = None) -> ExitAction | None:
    """Pozisyon açıkken orijinal sinyalin yeniden değerlendirilmesi."""
    if pos.qty <= 0:
        return None
    if new_confidence is not None and np.isfinite(new_confidence):
        pos.confidence = float(new_confidence)
        pos.confidence_history.append(round(float(new_confidence), 1))
        pos.confidence_history = pos.confidence_history[-50:]
        drop = pos.initial_confidence - new_confidence
        if new_confidence < cfg.decay_exit_confidence:
            pos.status = "WEAKENING"
            return ExitAction(ExitReason.SIGNAL_DECAY, pos.qty, price or pos.entry_price, True,
                              f"Güven {pos.initial_confidence:.0f} → {new_confidence:.0f}")
        if drop >= cfg.decay_drop_points:
            pos.status = "WEAKENING"
    if reversal_confidence is not None and reversal_confidence >= 75:
        return ExitAction(ExitReason.SIGNAL_REVERSAL, pos.qty, price or pos.entry_price, True,
                          f"Ters yönde güçlü sinyal ({reversal_confidence:.0f})")
    if regime_forbidden:
        return ExitAction(ExitReason.REGIME_CHANGE, pos.qty, price or pos.entry_price, True,
                          "Rejim strateji için yasaklı hale geldi")
    return None


def monitor(pos: ManagedPosition, price: float, row: pd.Series | None = None, interval_seconds: int = 300,
            funding_rate: float | None = None, oi_change: float | None = None, regime: str = "") -> dict:
    pnl, roi = pos.unrealized(price)
    tp = pos.take_profit
    out = {
        "symbol": pos.symbol, "direction": pos.direction, "current_price": price, "pnl": pnl, "roi_pct": roi,
        "r_multiple": pos.r_multiple(price),
        "distance_to_sl_pct": abs(price - pos.stop_loss) / price * 100 if price else float("nan"),
        "distance_to_tp_pct": abs(tp - price) / price * 100 if tp and price else float("nan"),
        "trailing_stop": pos.stop_loss, "stage": pos.stage, "confidence": pos.confidence,
        "time_in_trade_min": pos.bars_held * interval_seconds / 60,
        "expected_remaining_min": max(0, pos.expected_hold_bars - pos.bars_held) * interval_seconds / 60,
        "max_remaining_min": max(0, pos.max_hold_bars - pos.bars_held) * interval_seconds / 60,
        "funding_rate": funding_rate, "oi_change_pct": oi_change, "regime": regime,
        "liquidation_price": pos.liquidation_price,
    }
    if row is not None:
        out["atr"] = float(row.get("atr", float("nan")))
        out["volatility_pct"] = float(row.get("atr_pct", float("nan")))
    return out
