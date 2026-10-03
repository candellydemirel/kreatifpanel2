import logging
import time

import pytest

from kreatifbot.broker import PaperBroker
from kreatifbot.config import Settings, load_settings, save_settings
from kreatifbot.engine import BotEngine
from kreatifbot.models import ClosedTrade, Position
from kreatifbot.risk import RiskSettings
from kreatifbot.strategies import BUY, SELL
from kreatifbot.telegram import TelegramClient, TelegramError, TelegramNotifier, esc

from .test_engine import FakeClient, ScriptedStrategy, make_ohlcv


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class FakeTelegram:
    """Telegram Bot API'sini taklit eden oturum."""

    def __init__(self, chat_id="42"):
        self.chat_id = chat_id
        self.sent: list[dict] = []
        self.updates: list[dict] = []
        self.fail_next = 0

    def post(self, url, json=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        if "/botBAD/" in url:
            return _Resp(401, {"ok": False, "description": "Unauthorized"})
        if method == "getMe":
            return _Resp(200, {"ok": True, "result": {"username": "kreatif_test_bot"}})
        if method == "sendMessage":
            if self.fail_next:
                self.fail_next -= 1
                return _Resp(429, {"ok": False, "description": "Too Many Requests"})
            self.sent.append(json)
            return _Resp(200, {"ok": True, "result": {"message_id": len(self.sent)}})
        if method == "getUpdates":
            offset = (json or {}).get("offset")
            ups = [u for u in self.updates if offset is None or u["update_id"] >= offset]
            if (json or {}).get("timeout"):
                time.sleep(0.05)
            return _Resp(200, {"ok": True, "result": ups})
        return _Resp(404, {"ok": False, "description": "Not Found"})

    def texts(self):
        return [m["text"] for m in self.sent]


def wait_for(cond, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_client_errors_and_chat_discovery():
    fake = FakeTelegram()
    with pytest.raises(TelegramError, match="Token geçersiz"):
        TelegramClient("BAD", session=fake).get_me()
    with pytest.raises(TelegramError):
        TelegramClient("", session=fake).get_me()
    client = TelegramClient("TOKEN", session=fake)
    with pytest.raises(TelegramError, match="/start"):
        client.find_chat_id()
    fake.updates.append({"update_id": 5, "message": {"text": "/start",
                                                     "chat": {"id": 42, "first_name": "Can", "last_name": "D"}}})
    assert client.find_chat_id() == ("42", "Can D")


def test_long_message_truncated():
    fake = FakeTelegram()
    TelegramClient("TOKEN", session=fake).send_message(1, "x" * 10000)
    assert len(fake.sent[0]["text"]) <= 4000 and fake.sent[0]["parse_mode"] == "HTML"


def test_escape():
    assert esc("<b>&") == "&lt;b&gt;&amp;"


def _engine_with_notifier(tmp_path, notify=None, commands=False):
    fake = FakeTelegram()
    notifier = TelegramNotifier(TelegramClient("TOKEN", session=fake), "42", notify, commands=commands,
                                summary_hour=23)
    strat = ScriptedStrategy()
    client = FakeClient(make_ohlcv(400))
    engine = BotEngine(client, PaperBroker(balance=1000, slippage_pct=0), strat, RiskSettings(),
                       ["BTCUSDT"], "1h", state_path=tmp_path / "s.json", on_event=notifier.handle_event)
    notifier.attach(engine)
    notifier.start()
    return engine, notifier, fake, strat, client


def test_notifier_trade_flow(tmp_path):
    engine, notifier, fake, strat, client = _engine_with_notifier(tmp_path)
    strat.next_signal = BUY
    engine.tick()
    client.n += 1
    strat.next_signal = SELL
    engine.tick()
    assert wait_for(lambda: len(fake.sent) >= 4)
    texts = "\n".join(fake.texts())
    assert "AL sinyali — BTCUSDT" in texts
    assert "ALIM — BTCUSDT" in texts and "Stop-loss" in texts
    assert "SAT sinyali — BTCUSDT" in texts
    assert "SATIŞ — BTCUSDT" in texts and "K/Z" in texts
    assert all(m["chat_id"] == "42" for m in fake.sent)
    notifier.stop()


def test_notifier_respects_switches(tmp_path):
    engine, notifier, fake, strat, client = _engine_with_notifier(
        tmp_path, notify={"signals": False, "trades": True})
    strat.next_signal = BUY
    engine.tick()
    assert wait_for(lambda: len(fake.sent) >= 1)
    time.sleep(0.2)
    assert not any("sinyali" in t for t in fake.texts())
    assert any("ALIM" in t for t in fake.texts())
    notifier.stop()


def test_notifier_status_alert_and_halt():
    fake = FakeTelegram()
    n = TelegramNotifier(TelegramClient("T", session=fake), "42", commands=False)
    assert n.format_event("signal", {"symbol": "X", "signal": "BEKLE"}) is None
    assert n.format_event("log", "abc") is None
    assert "⚠️" in n.format_event("alert", {"level": logging.WARNING, "message": "<x>"})
    assert "&lt;x&gt;" in n.format_event("alert", {"level": logging.WARNING, "message": "<x>"})
    assert "Günlük zarar" in n.format_event("halt", "detay")
    assert n.format_event("equity", {"equity": 1, "quote": 1, "day_pct": 0}) is None
    assert "durduruldu" in n.format_event("status", "stopped")


def test_notifier_retries_failed_send():
    fake = FakeTelegram()
    fake.fail_next = 1
    n = TelegramNotifier(TelegramClient("T", session=fake), "42", commands=False)
    n._stop.wait = lambda t: False  # bekleme olmadan tekrar dene
    assert n._deliver("merhaba")
    assert fake.texts() == ["merhaba"]


def test_commands(tmp_path):
    engine, notifier, fake, strat, client = _engine_with_notifier(tmp_path, commands=False)
    strat.next_signal = BUY
    engine.tick()
    assert "Açık pozisyon: 1" in notifier.handle_command("/durum")
    assert "BTCUSDT" in notifier.handle_command("/pozisyonlar")
    assert "Henüz" in notifier.handle_command("/islemler")
    assert "Günlük özet" in notifier.handle_command("/ozet")
    assert "/durum" in notifier.handle_command("/yardim@kreatif_test_bot")
    assert "Bilinmeyen" in notifier.handle_command("/abc")
    engine.start()
    assert wait_for(lambda: engine.running)
    assert "durduruluyor" in notifier.handle_command("/durdur")
    assert wait_for(lambda: not engine.running, 10)
    notifier.stop()


def test_poller_only_accepts_authorized_chat(tmp_path):
    fake = FakeTelegram()
    fake.updates.append({"update_id": 1, "message": {"text": "/durum", "chat": {"id": 42}}})  # eski komut
    n = TelegramNotifier(TelegramClient("T", session=fake), "42", commands=True)
    n.start()
    time.sleep(0.2)
    fake.updates.append({"update_id": 2, "message": {"text": "/durum", "chat": {"id": 999}}})  # yabancı
    fake.updates.append({"update_id": 3, "message": {"text": "/yardim", "chat": {"id": 42}}})
    assert wait_for(lambda: len(fake.sent) >= 1)
    time.sleep(0.3)
    texts = fake.texts()
    assert len(texts) == 1 and "komutları" in texts[0]
    n.stop()


def test_daily_summary(tmp_path):
    fake = FakeTelegram()
    n = TelegramNotifier(TelegramClient("T", session=fake), "42", commands=False, summary_hour=0)
    n._summary_sent_for = None

    class E:
        positions = {"BTCUSDT": Position("BTCUSDT", 1, 1, 1, 0, 0, 1, "")}
        from datetime import datetime as _d
        trades = [ClosedTrade("BTCUSDT", 1, 1, 2, 1, 2, 1.0, 100, "", _d.now().strftime("%Y-%m-%d %H:%M:%S"),
                              "Kâr al")]
    n.attach(E())
    n._check_summary()
    n._check_summary()  # aynı gün ikinci kez gönderilmez
    assert n._queue.qsize() == 1
    text = n._queue.get()
    assert "Kapanan işlem: 1" in text and "+1.00" in text


def test_settings_token_encrypted(tmp_path):
    save_settings(Settings(telegram_token="123:abc", telegram_chat_id="42", telegram_enabled=True))
    from kreatifbot.config import settings_path
    raw = settings_path().read_text(encoding="utf-8")
    assert "123:abc" not in raw
    s = load_settings()
    assert s.telegram_token == "123:abc" and s.telegram_enabled
