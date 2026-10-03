import pandas as pd
import pytest

from kreatifbot.broker import PaperBroker
from kreatifbot.engine import BotEngine
from kreatifbot.risk import RiskSettings
from kreatifbot.strategies import BUY, HOLD, SELL, Strategy

from .conftest import make_ohlcv


class ScriptedStrategy(Strategy):
    """Son mumda önceden belirlenen sinyali veren test stratejisi."""
    key = "scripted"
    name = "Test"

    def __init__(self):
        super().__init__()
        self.next_signal = HOLD

    def min_bars(self):
        return 30

    def signals(self, df):
        s = pd.Series(HOLD, index=df.index)
        s.iloc[-1] = self.next_signal
        return s


class FakeClient:
    def __init__(self, df):
        self.df = df
        self.n = 200

    def klines(self, symbol, interval, limit=300, **kw):
        return self.df.iloc[max(0, self.n - limit):self.n].reset_index(drop=True)

    def price(self, symbol):
        return float(self.df["close"].iloc[self.n - 1])


def make_engine(tmp_path, risk=None):
    df = make_ohlcv(400)
    client = FakeClient(df)
    strat = ScriptedStrategy()
    events = []
    engine = BotEngine(client, PaperBroker(balance=1000, slippage_pct=0), strat, risk or RiskSettings(),
                       ["BTCUSDT"], "1h", state_path=tmp_path / "state.json",
                       on_event=lambda k, p: events.append((k, p)))
    return engine, client, strat, events


def test_engine_open_and_close(tmp_path):
    engine, client, strat, events = make_engine(tmp_path)
    strat.next_signal = BUY
    engine.tick()
    assert "BTCUSDT" in engine.positions
    pos = engine.positions["BTCUSDT"]
    assert pos.stop_loss < pos.entry_price < pos.take_profit
    assert engine.broker.balance < 1000

    # Aynı mumda sinyal tekrar işlenmez
    engine.tick()
    assert len(engine.positions) == 1

    client.n += 1
    strat.next_signal = SELL
    engine.tick()
    assert not engine.positions
    assert len(engine.trades) == 1 and engine.trades[0].reason == "Strateji sinyali"
    assert any(k == "trade" for k, _ in events)
    assert any(k == "equity" for k, _ in events)


def test_engine_stop_loss(tmp_path):
    engine, client, strat, _ = make_engine(tmp_path)
    strat.next_signal = BUY
    engine.tick()
    pos = engine.positions["BTCUSDT"]
    pos.stop_loss = pos.entry_price * 10  # fiyat hemen stop altında kalsın
    strat.next_signal = HOLD
    engine.tick()
    assert not engine.positions
    assert engine.trades[-1].reason == "Stop-loss"


def test_engine_state_persists(tmp_path):
    engine, client, strat, _ = make_engine(tmp_path)
    strat.next_signal = BUY
    engine.tick()
    balance = engine.broker.balance

    engine2, _, _, _ = make_engine(tmp_path)
    assert "BTCUSDT" in engine2.positions
    assert engine2.broker.balance == pytest.approx(balance)


def test_engine_max_positions_and_daily_halt(tmp_path):
    engine, client, strat, _ = make_engine(tmp_path, RiskSettings(max_open_positions=0))
    strat.next_signal = BUY
    engine.tick()
    assert not engine.positions

    engine, client, strat, _ = make_engine(tmp_path / "b")
    engine.tick()
    engine.halted_today = True
    client.n += 1
    strat.next_signal = BUY
    engine.tick()
    # yeni gün olmadığı sürece yeni işlem açılmaz
    assert not engine.positions


def test_engine_manual_close_and_validation(tmp_path):
    engine, client, strat, _ = make_engine(tmp_path)
    strat.next_signal = BUY
    engine.tick()
    engine.close_position("BTCUSDT")
    assert engine.trades[-1].reason == "Elle kapatıldı"
    with pytest.raises(ValueError):
        BotEngine(client, PaperBroker(), strat, RiskSettings(), ["BTCEUR"], "1h")


def test_engine_thread_start_stop(tmp_path):
    engine, *_ = make_engine(tmp_path)
    engine.start()
    assert engine.running
    engine.stop(wait=True)
    assert not engine.running
