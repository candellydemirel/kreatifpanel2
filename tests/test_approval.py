"""Manuel işlem onayı testleri."""

from kreatifbot.intel.approval import ApprovalBook
from kreatifbot.intel.execution import PaperVenue
from kreatifbot.intel.signal_store import SignalStore
from kreatifbot.telegram import TelegramNotifier

from .conftest import make_ohlcv
from .test_intel import FakeMarket, ScriptedDecisions, cfg_1h, with_taker


def _engine(tmp_path, book):
    from kreatifbot.intel.live_engine import IntelligentBotEngine
    cfg = cfg_1h()
    cfg.btc_context = cfg.use_futures_context = False
    cfg.data_quality.min_history_bars = 200
    events = []
    eng = IntelligentBotEngine(cfg, ["BTCUSDT"], FakeMarket(with_taker(make_ohlcv(700, seed=9))),
                               PaperVenue("SPOT", 10000, cfg), SignalStore(tmp_path / "s.db"),
                               decision_engine=ScriptedDecisions(cfg), state_path=tmp_path / "st.json",
                               on_event=lambda k, p: events.append((k, p)), kline_limit=600, approvals=book)
    return eng, events


def test_manual_approval_waits_then_opens(tmp_path):
    eng, events = _engine(tmp_path, ApprovalBook(10))
    eng.tick()
    assert "BTCUSDT" not in eng.positions and "BTCUSDT" in eng.pending   # onay bekleniyor, emir yok
    reqs = [p for k, p in events if k == "approval_request"]
    assert len(reqs) == 1 and reqs[0].symbol == "BTCUSDT"
    eng.tick()
    assert len([1 for k, _ in events if k == "approval_request"]) == 1  # aynı sinyal için tek istek
    msg = eng.resolve_approval(reqs[0].request_id, True, "test")
    assert "Onaylandı" in msg
    assert "zaten" in eng.resolve_approval(reqs[0].request_id, False, "test")
    eng.tick()
    assert "BTCUSDT" in eng.positions


def test_manual_rejection_cancels(tmp_path):
    eng, events = _engine(tmp_path, ApprovalBook(10))
    eng.tick()
    req = [p for k, p in events if k == "approval_request"][0]
    eng.resolve_approval(req.request_id, False, "test")
    eng.tick()
    assert "BTCUSDT" not in eng.positions and "BTCUSDT" not in eng.pending


def test_auto_mode_opens_without_request(tmp_path):
    eng, events = _engine(tmp_path, None)
    eng.tick()
    assert "BTCUSDT" in eng.positions and not any(k == "approval_request" for k, _ in events)


def test_expired_request(tmp_path):
    book = ApprovalBook(10)
    state, req, new = book.gate("k", lambda: dict(symbol="X", direction="LONG", strategy="s", price=1, qty=1,
                                                  notional=1, stop=0.9))
    assert state == "PENDING" and new
    req.expires_at = 0
    state, req2, new = book.gate("k", lambda: {})
    assert state == "EXPIRED" and not new
    assert not book.resolve(req.request_id, True)[0]


class FakeTG:
    def __init__(self):
        self.sent, self.answers, self.removed = [], [], []

    def send_message(self, chat_id, text, html_mode=True, reply_markup=None):
        self.sent.append((text, reply_markup))

    def answer_callback(self, cid, text=""):
        self.answers.append(text)

    def remove_buttons(self, chat_id, mid):
        self.removed.append(mid)


def test_telegram_buttons(tmp_path):
    eng, events = _engine(tmp_path, ApprovalBook(10))
    tg = FakeTG()
    n = TelegramNotifier(tg, "42")
    n.attach(eng)
    eng.tick()
    req = [p for k, p in events if k == "approval_request"][0]
    n.handle_event("approval_request", req)
    item = n._queue.get_nowait()
    n._deliver(item)
    text, markup = tg.sent[-1]
    assert "ONAYI GEREKLİ" in text and markup["inline_keyboard"][0][0]["callback_data"] == f"ap:{req.request_id}:1"
    # yetkisiz sohbetten gelen düğme yok sayılır
    n._handle_callback({"id": "c0", "data": f"ap:{req.request_id}:1", "message": {"chat": {"id": 99}}})
    assert req.state == "PENDING"
    n._handle_callback({"id": "c1", "data": f"ap:{req.request_id}:1",
                        "message": {"chat": {"id": 42}, "message_id": 7}})
    assert req.state == "APPROVED" and tg.removed == [7] and "Onaylandı" in tg.answers[-1]
    eng.tick()
    assert "BTCUSDT" in eng.positions


def test_hard_take_profit_sells_everything_at_15pct():
    import pandas as pd

    from kreatifbot.intel.config import RiskConfig
    from kreatifbot.intel.position_manager import open_position, update_on_bar
    cfg = RiskConfig()
    assert cfg.hard_take_profit_pct == 15.0
    pos = open_position("XUSDT", "SPOT", "LONG", 100.0, 1.0, 90.0, [130.0, 140.0], [0.5, 0.5], "s",
                        "2026-01-01T00:00:00+00:00", 0, 0, 0, 70, "BULL", "id")
    acts = update_on_bar(pos, pd.Series({"open": 110, "high": 114.9, "low": 109, "close": 114, "atr": 1.0}), cfg)
    assert not acts and pos.qty == 1.0
    acts = update_on_bar(pos, pd.Series({"open": 114, "high": 116, "low": 113, "close": 115.5, "atr": 1.0}), cfg)
    assert len(acts) == 1 and acts[0].full and acts[0].qty == 1.0 and abs(acts[0].price - 115.0) < 1e-9
    assert "%15" in acts[0].note
    cfg.hard_take_profit_pct = 0                     # kapalı
    pos2 = open_position("XUSDT", "SPOT", "LONG", 100.0, 1.0, 90.0, [130.0], [1.0], "s",
                         "2026-01-01T00:00:00+00:00", 0, 0, 0, 70, "BULL", "id2")
    assert not update_on_bar(pos2, pd.Series({"open": 114, "high": 116, "low": 113, "close": 115, "atr": 1.0}), cfg)
