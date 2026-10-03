import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_ohlcv(n: int = 600, seed: int = 7, start: float = 100.0, interval_s: int = 3600) -> pd.DataFrame:
    """Trend ve yatay dönemler içeren sentetik mum verisi."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([
        np.full(n // 3, 0.002), np.full(n // 3, -0.0015), np.zeros(n - 2 * (n // 3)),
    ])
    rets = drift + rng.normal(0, 0.01, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    spread = np.abs(rng.normal(0, 0.006, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(100, 1000, n)
    open_time = pd.to_datetime(
        pd.Timestamp("2024-01-01", tz="UTC").value // 10**6 + np.arange(n) * interval_s * 1000, unit="ms", utc=True
    )
    return pd.DataFrame({
        "open_time": open_time, "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "close_time": open_time + pd.Timedelta(seconds=interval_s - 0.001),
        "quote_volume": volume * close,
    })


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KREATIFBOT_HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Testler gerçek ağa çıkmaz (CI'da internet olsa bile sonuçlar tekrarlanabilir kalır)."""
    def blocked(self, method, url, *a, **kw):
        raise requests.ConnectionError(f"test ortamında ağ kapalı: {url}")
    monkeypatch.setattr(requests.Session, "request", blocked)
