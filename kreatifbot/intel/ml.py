"""AI/ML katmanı: meta-labeling, kalibrasyon, özellik önemi ve model kayması (drift).

AI doğrudan AL/SAT üretmez. Birincil strateji bir aday sinyal ürettiğinde meta
model bu sinyalin başarılı olma olasılığını (TP bariyerine SL'den önce ulaşma)
tahmin eder; ayrıca beklenen getiri ve süre tarihsel benzer işlemlerden çıkarılır.

Dahili model: numpy ile L2-düzenlemeli lojistik regresyon (bağımlılık gerektirmez).
scikit-learn kuruluysa Random Forest seçilebilir. LSTM/GRU/Transformer/XGBoost/
LightGBM bu sürümde YOKTUR (arayüz `BaseModel` ile eklenebilir).
Yüksek doğruluk ≠ kârlı strateji: model katkısı ablasyon ve walk-forward ile ölçülür.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

META_FEATURES = [
    "rsi", "adx", "atr_pct", "vol_percentile", "bb_pctb", "bb_width_pct", "ema50_slope", "ema20_slope", "rvol",
    "roc", "zscore", "structure_trend", "st_dir", "di_spread", "macd_hist_atr", "cci", "mfi", "willr",
    "major_bias", "trend_bias", "confirmation_bias", "entry_bias",
    "taker_buy_ratio", "cvd_slope_norm", "oi_change_pct", "funding_rate",
    "pattern_bull_count", "pattern_bear_count", "dist_support_atr", "dist_resistance_atr",
]
DIRECTIONAL = {"ema50_slope", "ema20_slope", "roc", "zscore", "structure_trend", "st_dir", "di_spread",
               "macd_hist_atr", "major_bias", "trend_bias", "confirmation_bias", "entry_bias", "cvd_slope_norm",
               "oi_change_pct", "funding_rate"}


def prepare_meta_frame(f: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=f.index)
    a = f["atr"].replace(0, np.nan)
    derived = {
        "di_spread": f["di_plus"] - f["di_minus"],
        "macd_hist_atr": f["macd_hist"] / a,
        "cvd_slope_norm": f["cvd_slope"] / f["volume"].rolling(20, min_periods=5).sum().replace(0, np.nan),
        "dist_support_atr": (f["close"] - f["support"]) / a,
        "dist_resistance_atr": (f["resistance"] - f["close"]) / a,
    }
    for c in META_FEATURES:
        if c in derived:
            x[c] = derived[c]
        elif c in f:
            x[c] = pd.to_numeric(f[c], errors="coerce")
        else:
            x[c] = np.nan
    return x


def directional_matrix(x: pd.DataFrame, idx: np.ndarray, signs: np.ndarray, columns: list[str]) -> np.ndarray:
    m = x.iloc[idx][columns].to_numpy(dtype=float).copy()
    for j, c in enumerate(columns):
        if c in DIRECTIONAL:
            m[:, j] *= signs
        elif c in ("rsi", "mfi", "bb_pctb", "willr"):
            center = 0.5 if c == "bb_pctb" else (-50 if c == "willr" else 50)
            m[:, j] = center + (m[:, j] - center) * signs
    m = np.column_stack([m, signs])  # yön bayrağı
    return m


# ---------------------------------------------------------------------- triple barrier
def triple_barrier(f: pd.DataFrame, idx: np.ndarray, signs: np.ndarray, stop_mult: float, tp_r: float,
                   max_bars: int, fee_pct: float = 0.0) -> pd.DataFrame:
    """Etiketler: giriş i+1 açılışı, TP = giriş ± tp_r×R, SL = giriş ∓ stop_mult×ATR, dikey bariyer max_bars.

    Bu fonksiyon geleceği kullanır; YALNIZCA eğitim etiketleri için kullanılır ve eğitimde
    sadece çözülme anı eğitim penceresi içinde kalan örnekler alınır (purging).
    """
    o, h, l, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = f["atr"].to_numpy(dtype=float)
    n = len(f)
    rows = []
    for i, s in zip(idx, signs):
        e = i + 1
        if e >= n or not np.isfinite(atr[i]):
            continue
        entry = o[e]
        risk = stop_mult * atr[i]
        sl, tp = entry - s * risk, entry + s * tp_r * risk
        outcome, exit_px, end = 0, None, min(n - 1, e + max_bars)
        for j in range(e, end + 1):
            hit_sl = (l[j] <= sl) if s > 0 else (h[j] >= sl)
            hit_tp = (h[j] >= tp) if s > 0 else (l[j] <= tp)
            if hit_sl:  # aynı mumda ikisi: temkinli → SL
                outcome, exit_px, end = -1, sl, j
                break
            if hit_tp:
                outcome, exit_px, end = 1, tp, j
                break
        if exit_px is None:
            exit_px = c[end]
        ret = (exit_px / entry - 1) * 100 * s - fee_pct
        rows.append((i, s, outcome, ret, end - e + 1, end, ret / (risk / entry * 100) if risk > 0 else 0.0))
    return pd.DataFrame(rows, columns=["idx", "sign", "outcome", "ret_pct", "bars", "end_idx", "r_multiple"])


# ---------------------------------------------------------------------- modeller
class BaseModel:
    name = "base"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def importance(self) -> np.ndarray | None:
        return None


class LogisticModel(BaseModel):
    name = "logistic"

    def __init__(self, l2: float = 1.0, iters: int = 300):
        self.l2, self.iters = l2, iters
        self.mu = self.sd = self.w = None
        self.b = 0.0

    def _prep(self, x):
        x = np.where(np.isfinite(x), x, np.nan)
        if self.mu is None:
            self.mu = np.nanmean(x, axis=0)
            self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
            self.sd = np.nanstd(x, axis=0)
            self.sd = np.where((self.sd > 1e-12) & np.isfinite(self.sd), self.sd, 1.0)
        z = (x - self.mu) / self.sd
        return np.clip(np.nan_to_num(z, nan=0.0), -6, 6)

    def fit(self, x, y):
        self.mu = None
        z = self._prep(x)
        n, d = z.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(self.iters):  # Newton-Raphson (IRLS)
            p = 1 / (1 + np.exp(-(z @ w + b)))
            g_w = z.T @ (p - y) / n + self.l2 * w / n
            g_b = float(np.mean(p - y))
            s = p * (1 - p)
            hess = (z.T * s) @ z / n + self.l2 * np.eye(d) / n
            hb = float(np.mean(s)) + 1e-9
            try:
                step = np.linalg.solve(hess + 1e-9 * np.eye(d), g_w)
            except np.linalg.LinAlgError:
                step = g_w
            w -= step
            b -= g_b / hb
            if np.abs(step).max() < 1e-7:
                break
        self.w, self.b = w, b
        return self

    def predict_proba(self, x):
        z = self._prep(x)
        return 1 / (1 + np.exp(-(z @ self.w + self.b)))

    def importance(self):
        return np.abs(self.w) if self.w is not None else None


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, **kw):
        from sklearn.ensemble import RandomForestClassifier  # isteğe bağlı bağımlılık
        self.m = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_depth=6, random_state=7, **kw)

    def fit(self, x, y):
        self.m.fit(np.nan_to_num(x), y)
        return self

    def predict_proba(self, x):
        return self.m.predict_proba(np.nan_to_num(x))[:, 1]

    def importance(self):
        return self.m.feature_importances_


def make_model(name: str, l2: float = 1.0) -> BaseModel:
    if name == "random_forest":
        try:
            return RandomForestModel()
        except ImportError:
            pass
    return LogisticModel(l2=l2)


# ---------------------------------------------------------------------- değerlendirme
def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "actual": float(y[m].mean())})
    return pd.DataFrame(rows)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    t = calibration_table(p, y, bins)
    if t.empty:
        return float("nan")
    return float((t["n"] * (t["predicted"] - t["actual"]).abs()).sum() / t["n"].sum())


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def permutation_importance(model: BaseModel, x: np.ndarray, y: np.ndarray, names: list[str],
                           repeats: int = 5, seed: int = 7) -> pd.DataFrame:
    """Örneklem dışı veride bir özelliği karıştırınca AUC'deki düşüş."""
    rng = np.random.default_rng(seed)
    base = auc(model.predict_proba(x), y)
    rows = []
    for j, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            xp = x.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            drops.append(base - auc(model.predict_proba(xp), y))
        rows.append({"feature": name, "auc_drop": float(np.nanmean(drops)), "std": float(np.nanstd(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: özellik/tahmin dağılımı kayması."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(np.clip(actual, qs[0], qs[-1]), qs)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------------- meta etiketleyici
@dataclass
class MetaModel:
    model: BaseModel | None = None
    columns: list = field(default_factory=list)
    trained_samples: int = 0
    version: str = "untrained"
    base_rate: float = float("nan")
    train_pred: np.ndarray | None = None
    train_features: np.ndarray | None = None
    ret_by_bin: dict = field(default_factory=dict)
    bars_win_median: float = float("nan")
    bars_all_median: float = float("nan")
    metrics: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def fit(self, x_frame: pd.DataFrame, labels: pd.DataFrame, model_name: str = "logistic", l2: float = 1.0,
            min_samples: int = 100, version: str = "") -> "MetaModel":
        lab = labels[labels["outcome"] != 0] if (labels["outcome"] != 0).sum() >= min_samples else labels
        if len(lab) < min_samples:
            self.model = None
            self.metrics = {"status": f"Yetersiz örnek ({len(lab)} < {min_samples})"}
            return self
        cols = [c for c in META_FEATURES if x_frame[c].notna().mean() > 0.5]
        self.columns = cols
        x = directional_matrix(x_frame, lab["idx"].to_numpy(), lab["sign"].to_numpy(), cols)
        y = (lab["outcome"].to_numpy() == 1).astype(float)
        self.model = make_model(model_name, l2).fit(x, y)
        self.trained_samples = len(lab)
        self.base_rate = float(y.mean())
        p = self.model.predict_proba(x)
        self.train_pred, self.train_features = p, x
        bins = np.clip((p * 10).astype(int), 0, 9)
        self.ret_by_bin = {int(b): float(lab["ret_pct"].to_numpy()[bins == b].mean()) for b in np.unique(bins)}
        wins = lab[lab["outcome"] == 1]
        self.bars_win_median = float(wins["bars"].median()) if len(wins) else float("nan")
        self.bars_all_median = float(lab["bars"].median())
        self.version = version or f"{self.model.name}-{len(lab)}"
        self.metrics = {"status": "OK", "train_auc": auc(p, y), "train_brier": brier(p, y), "base_rate": self.base_rate}
        return self

    def predict(self, x_frame: pd.DataFrame, idx: np.ndarray, signs: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.full(len(idx), np.nan)
        return self.model.predict_proba(directional_matrix(x_frame, idx, signs, self.columns))

    def expected_return(self, p: float) -> float:
        if not self.ret_by_bin or not np.isfinite(p):
            return float("nan")
        b = int(min(9, max(0, p * 10)))
        if b in self.ret_by_bin:
            return self.ret_by_bin[b]
        nearest = min(self.ret_by_bin, key=lambda k: abs(k - b))
        return self.ret_by_bin[nearest]

    def drift(self, x_recent: np.ndarray) -> dict:
        """Eğitim ve son dönem arasında özellik ve tahmin dağılımı kayması."""
        if not self.ready or self.train_features is None or len(x_recent) < 20:
            return {"status": "UNAVAILABLE"}
        feats = {}
        for j, c in enumerate(self.columns):
            feats[c] = psi(self.train_features[:, j], x_recent[:, j])
        pred_psi = psi(self.train_pred, self.model.predict_proba(x_recent))
        vals = [v for v in feats.values() if np.isfinite(v)]
        return {"status": "OK", "prediction_psi": pred_psi, "max_feature_psi": max(vals) if vals else float("nan"),
                "feature_psi": feats}


def drift_state(drift: dict, degraded: float, paused: float) -> str:
    if drift.get("status") != "OK":
        return "ACTIVE"
    worst = np.nanmax([drift.get("prediction_psi", 0) or 0, drift.get("max_feature_psi", 0) or 0])
    if not math.isfinite(worst):
        return "ACTIVE"
    if worst >= paused:
        return "PAUSED"
    if worst >= degraded:
        return "DEGRADED"
    return "ACTIVE"
