"""Sinyal veritabanı (SQLite, standart kütüphane).

Her karar (işlem ve NO TRADE) kaydedilir; sinyal durumu ve sonuç (outcome,
çıkış nedeni, PnL, R, süre) güncellenir. Veritabanı yazılamıyorsa canlı motor
yeni emir göndermez (DatabaseError).
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    symbol TEXT, market_type TEXT, timeframe TEXT, strategy TEXT, direction TEXT, status TEXT,
    created_at TEXT, confirmed_at TEXT, entry REAL, sl REAL, tp1 REAL, tp2 REAL, tp3 REAL, tp4 REAL,
    confidence REAL, trend_score REAL, momentum_score REAL, volume_score REAL, volatility_score REAL,
    structure_score REAL, mtf_score REAL, orderflow_score REAL, oi_score REAL, funding_score REAL, ai_score REAL,
    market_regime TEXT, expected_return REAL, expected_value REAL, expected_duration REAL, maximum_duration REAL,
    signal_expiry TEXT, ai_probability REAL, risk_reward REAL, risk_percent REAL, position_size REAL,
    outcome TEXT, exit_reason TEXT, pnl REAL, pnl_gross REAL, fees REAL, r_multiple REAL, holding_time_min REAL,
    no_trade_reasons TEXT, warnings TEXT, reasons TEXT, explanation TEXT,
    strategy_version TEXT, model_version TEXT, source TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_symbol ON signals(symbol, created_at);
CREATE INDEX IF NOT EXISTS ix_signals_strategy ON signals(strategy, status);
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT, ts TEXT, line TEXT
);
"""


class DatabaseError(Exception):
    pass


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class SignalStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else data_dir() / "signals.sqlite3"
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.executescript(SCHEMA)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Sinyal veritabanı açılamadı: {exc}") from exc

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def healthy(self) -> bool:
        try:
            with self._lock, self._conn() as c:
                c.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def save_decision(self, d, source: str = "live", log: bool = True) -> None:
        tps = list(d.take_profit_levels) + [None] * 4

        def sc(g):
            v = d.scores.get(g) if d.scores else None
            return _num(v[0]) if v else None

        row = {
            "signal_id": d.signal_id, "symbol": d.symbol, "market_type": d.market, "timeframe": d.timeframe,
            "strategy": d.strategy, "direction": d.direction, "status": d.signal_status, "created_at": d.created_at,
            "confirmed_at": d.created_at if d.signal_status == "CONFIRMED" else None,
            "entry": _num(d.entry), "sl": _num(d.stop_loss), "tp1": _num(tps[0]), "tp2": _num(tps[1]),
            "tp3": _num(tps[2]), "tp4": _num(tps[3]), "confidence": _num(d.confidence),
            "trend_score": sc("trend"), "momentum_score": sc("momentum"), "volume_score": sc("volume"),
            "volatility_score": sc("volatility"), "structure_score": sc("structure"), "mtf_score": sc("mtf"),
            "orderflow_score": sc("orderflow"), "oi_score": sc("oi_funding"), "funding_score": sc("oi_funding"),
            "ai_score": sc("ai"), "market_regime": d.market_regime, "expected_return": _num(d.expected_return),
            "expected_value": _num(d.expected_value_pct), "expected_duration": _num(d.expected_duration_min),
            "maximum_duration": _num(d.maximum_duration_min), "signal_expiry": d.signal_expiry,
            "ai_probability": _num(d.ai_probability), "risk_reward": _num(d.risk_reward),
            "risk_percent": _num(d.risk_percent), "position_size": _num(d.position_size),
            "no_trade_reasons": json.dumps(d.no_trade_reasons, ensure_ascii=False),
            "warnings": json.dumps(d.warnings, ensure_ascii=False),
            "reasons": json.dumps(d.reasons, ensure_ascii=False), "explanation": d.explanation,
            "strategy_version": d.strategy_version, "model_version": d.model_version, "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        try:
            with self._lock, self._conn() as c:
                c.execute(f"INSERT OR REPLACE INTO signals ({cols}) VALUES ({qs})", list(row.values()))
                if log and d.log_lines:
                    ts = datetime.now(timezone.utc).isoformat()
                    c.executemany("INSERT INTO decision_log (signal_id, ts, line) VALUES (?, ?, ?)",
                                  [(d.signal_id, ts, line) for line in d.log_lines])
        except sqlite3.Error as exc:
            raise DatabaseError(f"Sinyal kaydedilemedi: {exc}") from exc

    def update_status(self, signal_id: str, status: str) -> None:
        self._update(signal_id, {"status": status})

    def record_outcome(self, signal_id: str, outcome: str, exit_reason: str, pnl: float, pnl_gross: float,
                       fees: float, r_multiple: float, holding_min: float) -> None:
        self._update(signal_id, {"status": "CLOSED", "outcome": outcome, "exit_reason": exit_reason,
                                 "pnl": _num(pnl), "pnl_gross": _num(pnl_gross), "fees": _num(fees),
                                 "r_multiple": _num(r_multiple), "holding_time_min": _num(holding_min)})

    def _update(self, signal_id: str, fields: dict) -> None:
        fields = {**fields, "updated_at": datetime.now(timezone.utc).isoformat()}
        sets = ", ".join(f"{k} = ?" for k in fields)
        try:
            with self._lock, self._conn() as c:
                c.execute(f"UPDATE signals SET {sets} WHERE signal_id = ?", [*fields.values(), signal_id])
        except sqlite3.Error as exc:
            raise DatabaseError(f"Sinyal güncellenemedi: {exc}") from exc

    def recent(self, limit: int = 200, symbol: str | None = None, only_trades: bool = False) -> pd.DataFrame:
        q = "SELECT * FROM signals"
        cond, args = [], []
        if symbol:
            cond.append("symbol = ?")
            args.append(symbol)
        if only_trades:
            cond.append("status IN ('CONFIRMED','EXECUTED','ACTIVE','CLOSED','EXPIRED','CANCELLED')")
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock, self._conn() as c:
            return pd.read_sql_query(q, c, params=args)

    def log_for(self, signal_id: str) -> list[str]:
        with self._lock, self._conn() as c:
            return [r[0] for r in c.execute("SELECT line FROM decision_log WHERE signal_id = ? ORDER BY id",
                                            (signal_id,))]

    def strategy_stats(self, min_trades: int = 1) -> dict:
        """Kapanmış canlı/kağıt işlemlerden strateji istatistikleri (gerçek sonuçlar)."""
        with self._lock, self._conn() as c:
            df = pd.read_sql_query("SELECT strategy, pnl, r_multiple, holding_time_min, exit_reason FROM signals "
                                   "WHERE status = 'CLOSED' AND pnl IS NOT NULL", c)
        out = {}
        for key, g in df.groupby("strategy"):
            if len(g) < min_trades:
                continue
            out[key] = {"n": len(g), "p_win": float((g["pnl"] > 0).mean()), "avg_r": float(g["r_multiple"].mean()),
                        "median_hold_min": float(g["holding_time_min"].median()),
                        "top_exit": g["exit_reason"].mode().iat[0] if not g["exit_reason"].mode().empty else ""}
        return out
