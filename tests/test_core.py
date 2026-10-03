import math

import numpy as np
import pandas as pd
import pytest

from kreatifbot import indicators as ind
from kreatifbot.analyzer import analyze
from kreatifbot.backtest import compare_strategies, run_backtest
from kreatifbot.binance_client import BinanceAPIError, BinanceClient, floor_to_step, sign_query
from kreatifbot.broker import BrokerError, PaperBroker
from kreatifbot.config import Settings, load_settings, protect, save_settings, unprotect
from kreatifbot.risk import RiskManager, RiskSettings
from kreatifbot.strategies import BUY, HOLD, SELL, STRATEGIES, create_strategy

from .conftest import make_ohlcv


# ---------------------------------------------------------------- göstergeler
def test_rsi_bounds_and_extremes(ohlcv):
    r = ind.rsi(ohlcv["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    rising = pd.Series(np.arange(1, 50, dtype=float))
    assert ind.rsi(rising).iloc[-1] == pytest.approx(100)
    flat = pd.Series(np.ones(50))
    assert ind.rsi(flat).iloc[-1] == pytest.approx(50)


def test_ema_matches_manual():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    e = ind.ema(s, 3)
    alpha = 2 / 4
    manual = 1.0
    for v in s.iloc[1:]:
        manual = alpha * v + (1 - alpha) * manual
    assert e.iloc[-1] == pytest.approx(manual)
    assert math.isnan(e.iloc[1])


def test_atr_adx_supertrend(ohlcv):
    a = ind.atr(ohlcv).dropna()
    assert (a > 0).all()
    adx_val, pdi, mdi = ind.adx(ohlcv)
    assert adx_val.dropna().between(0, 100).all()
    line, direction = ind.supertrend(ohlcv)
    assert set(direction.unique()) <= {-1.0, 0.0, 1.0}
    up = direction == 1
    assert (line[up] <= ohlcv["high"][up] + 1e-9).all()


def test_cross_helpers():
    a = pd.Series([1.0, 2, 3, 2, 1])
    b = pd.Series([2.0, 2, 2, 2, 2])
    assert list(ind.cross_above(a, b)) == [False, False, True, False, False]
    assert list(ind.cross_below(a, b)) == [False, False, False, False, True]


# ---------------------------------------------------------------- stratejiler
@pytest.mark.parametrize("key", list(STRATEGIES))
def test_strategy_has_no_lookahead(key, ohlcv):
    """Sinyaller sonraki verilerden etkilenmemeli."""
    strat = create_strategy(key)
    full = strat.signals(ohlcv)
    for cut in (250, 400, 520):
        partial = strat.signals(ohlcv.iloc[:cut].reset_index(drop=True))
        assert (partial.to_numpy() == full.iloc[:cut].to_numpy()).all(), f"{key} kesim {cut}"
    assert set(full.unique()) <= {BUY, SELL, HOLD}


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_strategy_evaluate(key, ohlcv):
    res = create_strategy(key).evaluate(ohlcv)
    assert res.signal in (BUY, SELL, HOLD)
    assert isinstance(res.reason, str)


def test_strategies_produce_trades(ohlcv):
    total = sum(int((create_strategy(k).signals(ohlcv) == BUY).sum()) for k in STRATEGIES)
    assert total > 10


def test_strategy_params_cast():
    s = create_strategy("ema_cross", {"fast": 5.0, "slow": 30})
    assert s.params["fast"] == 5 and isinstance(s.params["fast"], int)


# ---------------------------------------------------------------- analiz
def test_analyze(ohlcv):
    a = analyze(ohlcv, "TESTUSDT", "1h")
    assert -100 <= a.score <= 100
    assert a.recommendation in {"GÜÇLÜ AL", "AL", "NÖTR", "SAT", "GÜÇLÜ SAT"}
    assert a.support <= a.resistance
    assert "GENEL SKOR" in a.summary_text()


def test_analyze_uptrend_is_positive():
    df = make_ohlcv(600, seed=1)
    up = df.iloc[:200].reset_index(drop=True)  # ilk bölüm yükseliş trendi
    assert analyze(up).score > 0


def test_analyze_requires_data(ohlcv):
    with pytest.raises(ValueError):
        analyze(ohlcv.iloc[:20])


# ---------------------------------------------------------------- risk
def test_position_size_and_stops():
    rm = RiskManager(RiskSettings(risk_per_trade_pct=1, stop_atr_mult=2, take_profit_rr=2, max_position_pct=50))
    qty = rm.position_size(1000, 100, 2)  # risk 10 / stop mesafesi 4 = 2.5
    assert qty == pytest.approx(2.5)
    assert rm.position_size(1000, 100, 0.01) == pytest.approx(5.0)  # %50 sınırı
    sl, tp = rm.stops(100, 2)
    assert sl == pytest.approx(96) and tp == pytest.approx(108)
    assert rm.trail(96, 110, 2) == 96  # iz süren stop kapalı
    rm2 = RiskManager(RiskSettings(trailing_atr_mult=1.5))
    assert rm2.trail(96, 110, 2) == pytest.approx(107)
    assert rm.daily_loss_hit(1000, 949)
    assert not rm.daily_loss_hit(1000, 960)


def test_risk_settings_roundtrip():
    rs = RiskSettings(max_open_positions=5, fee_pct=0.075)
    back = RiskSettings.from_dict(rs.to_dict())
    assert back == rs and isinstance(back.max_open_positions, int)


# ---------------------------------------------------------------- backtest
@pytest.mark.parametrize("key", list(STRATEGIES))
def test_backtest_accounting(key, ohlcv):
    res = run_backtest(ohlcv, create_strategy(key), RiskSettings(), 1000, "1h", "TEST")
    pnl_sum = sum(t.pnl for t in res.trades)
    assert res.final_equity == pytest.approx(1000 + pnl_sum, rel=1e-9)
    assert res.metrics["Maks. düşüş (%)"] <= 0
    assert len(res.equity) == len(ohlcv)
    for t in res.trades:
        assert t.reason in {"Strateji sinyali", "Stop-loss", "Kâr al", "Test sonu"}


def test_backtest_no_fee_flat_market():
    df = make_ohlcv(300)
    for col in ("open", "high", "low", "close"):
        df[col] = 100.0
    res = run_backtest(df, create_strategy("ema_cross"), RiskSettings(fee_pct=0, slippage_pct=0), 1000, "1h")
    assert res.final_equity == pytest.approx(1000)


def test_compare_strategies(ohlcv):
    results = compare_strategies(ohlcv, [create_strategy(k) for k in STRATEGIES], RiskSettings(), 1000, "1h")
    returns = [r.metrics["Toplam getiri (%)"] for r in results]
    assert returns == sorted(returns, reverse=True)


# ---------------------------------------------------------------- aracı
def test_paper_broker():
    b = PaperBroker(balance=1000, fee_pct=0.1, slippage_pct=0)
    fill = b.market_buy("BTCUSDT", 500, 100)
    assert b.balance == pytest.approx(500)
    assert fill.qty == pytest.approx(4.995)
    sell = b.market_sell("BTCUSDT", fill.qty, 110)
    assert sell.quote == pytest.approx(4.995 * 110 * 0.999)
    with pytest.raises(BrokerError):
        b.market_buy("BTCUSDT", 10_000, 100)


# ---------------------------------------------------------------- binance istemcisi
def test_signature_matches_binance_docs():
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    query = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
             "&recvWindow=5000&timestamp=1499827319559")
    assert sign_query(secret, query) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_floor_to_step():
    assert floor_to_step(0.123456, "0.00100000") == "0.123"
    assert floor_to_step(5.9, "1.00000000") == "5"
    assert floor_to_step(12.3456, "0.01") == "12.34"
    assert floor_to_step(1e-9, "0.001") == "0"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, timeout=None):
        self.calls.append((method, url, headers))
        return self.responses.pop(0)


def test_client_klines_and_errors():
    now_ms = int(pd.Timestamp.now(tz="UTC").value // 10**6)
    rows = [[now_ms - 7_200_000, "1", "2", "0.5", "1.5", "10", now_ms - 3_600_001, "15", 3, "1", "1", "0"],
            [now_ms - 3_600_000, "1.5", "2", "1", "1.8", "11", now_ms + 1000, "16", 3, "1", "1", "0"]]
    session = _Session([_Resp(200, rows), _Resp(200, rows),
                        _Resp(400, {"code": -1121, "msg": "Invalid symbol."})])
    client = BinanceClient(testnet=True, session=session)
    df = client.klines("BTCUSDT", "1h")
    assert len(df) == 2 and df["close"].iloc[-1] == 1.8
    assert len(client.klines("BTCUSDT", "1h", closed_only=True)) == 1
    with pytest.raises(BinanceAPIError) as err:
        client.klines("XXX", "1h")
    assert err.value.code == -1121
    with pytest.raises(BinanceAPIError):
        client.account()  # anahtar yok


def test_signed_request_has_signature():
    session = _Session([_Resp(200, {"balances": [{"asset": "USDT", "free": "12.5", "locked": "0"},
                                                  {"asset": "BTC", "free": "0", "locked": "0"}]})])
    client = BinanceClient("key", "secret", testnet=True, session=session)
    assert client.balances() == {"USDT": {"free": 12.5, "locked": 0.0}}
    method, url, headers = session.calls[0]
    assert "signature=" in url and "timestamp=" in url
    assert headers["X-MBX-APIKEY"] == "key"
    assert url.startswith("https://testnet.binance.vision/api/v3/account")


# ---------------------------------------------------------------- ayarlar
def test_settings_roundtrip():
    s = Settings(api_key="abc", api_secret="s3cr3t", symbols=["SOLUSDT"], live_mode=True)
    save_settings(s)
    loaded = load_settings()
    assert loaded.api_secret == "s3cr3t" and loaded.symbols == ["SOLUSDT"] and loaded.live_mode
    assert unprotect(protect("xyz")) == "xyz"
