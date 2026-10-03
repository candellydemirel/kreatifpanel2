"""Arayüz duman testi: sahte Binance istemcisiyle tüm sekmeler uçtan uca çalıştırılır."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from kreatifbot.binance_client import SymbolRules  # noqa: E402

from .conftest import make_ohlcv  # noqa: E402


class FakeClient:
    base_url = "fake"
    testnet = True
    has_keys = False

    def __init__(self):
        self.df = make_ohlcv(1200)

    def klines(self, symbol, interval, limit=500, **kw):
        return self.df.tail(limit).reset_index(drop=True)

    def klines_history(self, symbol, interval, total):
        return self.df.tail(total).reset_index(drop=True)

    def ticker_24h(self, symbol=None):
        return [{"symbol": s, "quoteVolume": str(1e6 * (i + 1)), "priceChangePercent": "1.5"}
                for i, s in enumerate(["BTCUSDT", "ETHUSDT", "USDCUSDT", "BTCUPUSDT", "SOLUSDT"])]

    def sync_time(self):
        return 0

    def ping(self):
        return True

    def symbol_rules(self, symbol):
        return SymbolRules(symbol, symbol[:-4], "USDT", "0.0001", 0.0001, "0.01", 5.0, 8)

    def price(self, symbol):
        return float(self.df["close"].iloc[-1])


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, cond=lambda: True, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        QThreadPool.globalInstance().waitForDone(50)
        app.processEvents()
        if cond():
            return True
    return False


@pytest.fixture
def window(app, monkeypatch):
    from kreatifbot.gui import main_window
    errors = []
    monkeypatch.setattr(main_window.MainWindow, "show_error", lambda self, t, m: errors.append((t, m)))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w = main_window.MainWindow()
    fake = FakeClient()
    w.data_client = lambda: fake
    w.trade_client = lambda: fake
    w.errors = errors
    w.show()
    yield w
    w.bot.shutdown()
    w.close()


def test_analysis_tab(app, window):
    window.analysis.run()
    assert wait(app, lambda: window.analysis.button.isEnabled() and window.analysis.signals.rowCount() > 0)
    assert "GENEL SKOR" in window.analysis.summary.toPlainText()
    assert not window.errors


def test_scanner_tab(app, window):
    window.scanner.run()
    assert wait(app, lambda: window.scanner.table.rowCount() > 0)
    symbols = {window.scanner.table.item(r, 0).text() for r in range(window.scanner.table.rowCount())}
    assert symbols == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}  # stabil ve kaldıraçlı tokenlar elendi
    assert not window.errors


def test_backtest_and_compare(app, window):
    window.backtest.run()
    assert wait(app, lambda: window.backtest.metrics.rowCount() > 0 and window.backtest.run_btn.isEnabled())
    window.backtest.compare()
    assert wait(app, lambda: window.backtest.compare_table.rowCount() == 7)
    assert not window.errors


def test_bot_paper_run(app, window):
    bot = window.bot
    bot.engine_type.setCurrentIndex(bot.engine_type.findData("classic"))
    bot.paper.setChecked(True)
    bot.symbols.setText("BTCUSDT, ETHUSDT")
    bot.poll.setValue(5)
    bot.start()
    assert wait(app, lambda: bot.engine is not None and bot.engine.running)
    assert wait(app, lambda: "Bot başlatıldı" in bot.log.toPlainText())
    assert wait(app, lambda: bot.cards["Toplam değer"].text() != "-")
    assert not bot.start_btn.isEnabled()
    bot.stop()
    assert wait(app, lambda: not bot.engine.running, timeout=40)
    assert wait(app, lambda: bot.start_btn.isEnabled())
    assert not window.errors


def test_settings_save(app, window):
    window.settings_tab.api_key.setText("abc")
    window.settings_tab.testnet.setChecked(False)
    window.settings_tab.save()
    from kreatifbot.config import load_settings
    s = load_settings()
    assert s.api_key == "abc" and s.testnet is False
    assert "GERÇEK" in window.net_badge.text()


def test_api_dialog_saves_keys(app, window):
    from kreatifbot.config import load_settings
    from kreatifbot.gui.api_dialog import ApiKeyDialog
    dialog = ApiKeyDialog(window, first_run=True)
    dialog.api_key.setText("  KEY123 ")
    dialog.api_secret.setText("SECRET456")
    dialog.testnet.setChecked(True)
    dialog.save_and_close()
    s = load_settings()
    assert (s.api_key, s.api_secret, s.testnet) == ("KEY123", "SECRET456", True)
    assert window.settings_tab.api_key.text() == "KEY123"  # Ayarlar sekmesi de güncellendi
    assert "TESTNET" in window.net_badge.text()


def test_api_dialog_requires_both_keys(app, window):
    from kreatifbot.gui.api_dialog import ApiKeyDialog
    dialog = ApiKeyDialog(window)
    dialog.api_key.setText("only-key")
    dialog.api_secret.setText("")
    dialog.test()
    assert "doldurun" in dialog.result.text()


def test_guide_pdf_bundled():
    from kreatifbot.config import GUIDE_PDF, resource_path
    path = resource_path(GUIDE_PDF)
    assert path.exists() and path.read_bytes()[:4] == b"%PDF"


def test_bot_sends_telegram(app, window):
    from kreatifbot.telegram import TelegramClient, TelegramNotifier

    from .test_telegram import FakeTelegram
    fake = FakeTelegram()
    window.make_notifier = lambda: TelegramNotifier(TelegramClient("T", session=fake), "42", commands=False)
    bot = window.bot
    bot.engine_type.setCurrentIndex(bot.engine_type.findData("classic"))
    bot.paper.setChecked(True)
    bot.symbols.setText("BTCUSDT")
    bot.start()
    assert wait(app, lambda: bot.engine is not None and bot.engine.running)
    assert wait(app, lambda: any("başlatıldı" in m["text"] for m in fake.sent))
    bot.stop()
    assert wait(app, lambda: any("durduruldu" in m["text"] for m in fake.sent), timeout=40)
    assert wait(app, lambda: not bot.notifier.running, timeout=10)


def test_telegram_tab_save_and_scanner_send(app, window):
    tab = window.telegram_tab
    tab.enabled.setChecked(True)
    tab.token.setText("123:abc")
    tab.chat_id.setText("42")
    tab.notify["signals"].setChecked(False)
    tab.save()
    from kreatifbot.config import load_settings
    s = load_settings()
    assert s.telegram_enabled and s.telegram_chat_id == "42" and s.telegram_notify["signals"] is False
    assert window.telegram_ready()

    sent = []
    window.telegram_send = lambda text: sent.append(text)
    window.scanner.run()
    assert wait(app, lambda: window.scanner.tg_button.isEnabled())
    window.scanner.send_telegram()
    window.analysis.run()
    assert wait(app, lambda: window.analysis.tg_button.isEnabled())
    window.analysis.send_telegram()
    assert "Piyasa taraması" in sent[0] and "analizi" in sent[1]


# ---------------------------------------------------------------- Zeka Motoru / Araştırma sekmeleri
def _patch_intel_network(monkeypatch):
    """Ağ çağrılarını sentetik veriyle değiştirir (yalnızca test)."""
    from kreatifbot.intel import futures_client, market_data
    from kreatifbot.intel.types import FeatureSet

    from .test_intel import FakeMarket, resample, with_taker
    df = with_taker(make_ohlcv(1500, seed=4))

    def fake_history(cfg, symbol, bars, spot=None, futures=None, include_derivatives=True):
        e = df.tail(bars).reset_index(drop=True)
        return market_data.HistoricalBundle(symbol, cfg.market, e, {"trend": resample(e, "4h"),
                                            "major": resample(e, "4h"), "confirmation": e,
                                            "macro": resample(e, "1D")}, None, None, [])

    monkeypatch.setattr(market_data, "load_history", fake_history)
    monkeypatch.setattr(market_data, "live_snapshot", lambda *a, **k: (FeatureSet(), {"spot_book": None,
                                                                                     "futures_book": None}))
    fm = FakeMarket(df)

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def symbol_rules(self, symbol):
            return fm.symbol_rules(symbol)

    import kreatifbot.binance_client as bc
    monkeypatch.setattr(bc, "BinanceClient", FakeClient)

    class FakeFut(FakeClient):
        def funding_history(self, *a, **k):
            raise bc.BinanceAPIError(0, -2, "test: ağ yok")
        open_interest_hist = long_short_ratio = taker_volume = premium_index = funding_history

    monkeypatch.setattr(futures_client, "BinanceFuturesClient", FakeFut)
    return fm


def _intel_cfg_1h(tmp_path=None):
    from kreatifbot.intel.config import IntelConfig, save_intel_config
    cfg = IntelConfig()
    cfg.timeframes.entry, cfg.timeframes.confirmation, cfg.timeframes.trend = "1h", "1h", "4h"
    cfg.timeframes.major, cfg.timeframes.macro = "4h", "1d"
    cfg.btc_context = False
    cfg.data_quality.min_history_bars = 200
    save_intel_config(cfg)
    return cfg


def test_intel_tab_decision_and_config(app, window, monkeypatch):
    _patch_intel_network(monkeypatch)
    _intel_cfg_1h()
    tab = window.intel_tab
    tab.load_cfg()
    tab.bars.setValue(600)
    tab.run()
    assert wait(app, lambda: tab.run_btn.isEnabled() and tab.log.toPlainText() != "", timeout=60)
    text = tab.explain.toPlainText()
    assert "Karar:" in text and "Piyasa rejimi" in text
    assert tab.scores.rowCount() >= 9 and tab.strats.rowCount() == 33
    # Yönetici ayarları: geçersiz JSON reddedilir, geçerli kaydedilir
    tab.cfg_edit.setPlainText("{bozuk")
    tab.save_cfg()
    assert "JSON" in tab.cfg_msg.text()
    tab.load_cfg()
    tab.st_table.cellWidget(0, 0).setChecked(False)
    tab.save_strategies()
    from kreatifbot.intel.config import load_intel_config
    first_key = tab.st_table.item(0, 1).data(0x0100)
    assert load_intel_config().strategy(first_key).enabled is False
    assert not window.errors


def test_research_tab_backtest_and_questions(app, window, monkeypatch):
    _patch_intel_network(monkeypatch)
    _intel_cfg_1h()
    tab = window.research_tab
    tab.bars.setValue(1500)
    tab.run_backtest()
    assert wait(app, lambda: all(b.isEnabled() for b in tab.buttons) and tab.summary.rowCount() > 0, timeout=120)
    assert tab.rc.backtest is not None and tab.rc.backtest.status == "OK"
    tab.question.setCurrentIndex(list(__import__("kreatifbot.intel.research", fromlist=["x"]).QUESTIONS).index(
        "edge_after_fees"))
    tab.ask()
    assert "Brüt PnL" in tab.answer.toPlainText()


def test_bot_tab_starts_intel_engine_paper(app, window, monkeypatch):
    fm = _patch_intel_network(monkeypatch)
    cfg = _intel_cfg_1h()
    cfg.use_futures_context = False
    from kreatifbot.intel.config import save_intel_config
    save_intel_config(cfg)
    window.data_client = lambda: fm
    bot = window.bot
    bot.paper.setChecked(True)
    bot.engine_type.setCurrentIndex(bot.engine_type.findData("intel"))
    assert not bot.strategy.isVisible()
    bot.symbols.setText("BTCUSDT")
    bot.poll.setValue(5)
    bot.start()
    assert wait(app, lambda: bot.engine is not None and bot.engine.running, timeout=30)
    assert wait(app, lambda: window.intel_tab.live_table.rowCount() > 0, timeout=60)
    assert "Zeka Motoru" in bot.log.toPlainText()
    bot.stop()
    assert wait(app, lambda: not bot.engine.running, timeout=40)
    assert not window.errors


# ---------------------------------------------------------------- arka planda çalışma
def test_system_helpers_are_safe_off_windows():
    from kreatifbot import system
    if not system.is_windows():
        assert system.prevent_sleep(True) is False
        assert system.set_autostart(True) is False and system.autostart_enabled() is False
    cmd = system.launch_command()
    assert "--minimized" in cmd and "main.py" in cmd


def test_close_hides_to_tray_and_sleep_prevention(app, window, monkeypatch):
    from unittest.mock import MagicMock

    from PySide6.QtGui import QAction

    from kreatifbot.gui import main_window
    calls = []
    monkeypatch.setattr(main_window, "prevent_sleep", lambda on: calls.append(on))
    window.tray = MagicMock()
    window._tray_toggle = QAction("x", window)
    window.settings.minimize_to_tray = True
    window.close()
    assert not window.isVisible() and window.tray.showMessage.called  # gizlendi, kapanmadı
    window.show_from_tray()
    assert window.isVisible()
    window.set_bot_running(True)
    window.set_bot_running(False)
    assert calls == [True, False]
    assert window._tray_toggle.text() == "Botu başlat"
    window.settings.prevent_sleep = False
    window.set_bot_running(True)
    assert calls[-1] is False
    window.set_bot_running(False)
    window.tray = None


def test_unattended_start_rules(app, window, monkeypatch):
    bot = window.bot
    # Canlı modda onay verilmemişse otomatik başlatma yapılmaz
    bot.live.setChecked(True)
    window.settings.unattended_live_confirmed = False
    bot.start(unattended=True)
    assert bot.engine is None and "onay verilmediği" in bot.log.toPlainText()
    # Kağıt modda otomatik başlar, onay penceresi açılmaz
    monkeypatch.setattr(QMessageBox, "exec", lambda *a: (_ for _ in ()).throw(AssertionError("pencere açıldı")))
    bot.paper.setChecked(True)
    bot.engine_type.setCurrentIndex(bot.engine_type.findData("classic"))
    bot.symbols.setText("BTCUSDT")
    bot.start(unattended=True)
    assert wait(app, lambda: bot.engine is not None and bot.engine.running)
    bot.stop()
    assert wait(app, lambda: not bot.engine.running, timeout=40)


def test_background_settings_saved(app, window):
    t = window.settings_tab
    t.tray.setChecked(False)
    t.nosleep.setChecked(False)
    t.autobot.setChecked(True)
    t.save()
    from kreatifbot.config import load_settings
    s = load_settings()
    assert (s.minimize_to_tray, s.prevent_sleep, s.start_bot_on_launch) == (False, False, True)
    assert s.autostart is False  # Windows dışında ayarlanamaz


def test_news_tab_insights(app, window, monkeypatch):
    import kreatifbot.binance_client as bc
    from kreatifbot.intel import catalyst as cat_mod
    from kreatifbot.intel.news import NewsStore

    from .test_catalyst import FakeAPISession, item, uptrend_1h
    from .test_intel import resample
    abc = uptrend_1h()

    class FakeEx:
        def __init__(self, *a, **k):
            pass

        def exchange_info(self):
            return {"symbols": [{"symbol": "ABCUSDT", "status": "TRADING", "quoteAsset": "USDT", "baseAsset": "ABC"}]}

        def klines(self, symbol, interval, limit=500, **kw):
            return abc.tail(limit).reset_index(drop=True) if interval == "1h" else resample(abc, "4h")

    monkeypatch.setattr(bc, "BinanceClient", FakeEx)
    orig = cat_mod.FundamentalsProvider.__init__

    def init(self, *a, **k):
        k["session"] = FakeAPISession()
        orig(self, *a, **k)
    monkeypatch.setattr(cat_mod.FundamentalsProvider, "__init__", init)
    from kreatifbot.intel.config import load_intel_config, save_intel_config
    cfg = load_intel_config()
    cfg.catalyst.min_quote_volume_24h = 1000
    save_intel_config(cfg)
    NewsStore().add([item("ABC partners with Microsoft", "ABC", 2, iid="g1"),
                     item("ABC strikes partnership with Microsoft", "ABC", 3, "Decrypt", iid="g2")])
    tab = window.news_tab
    tab.run_insights()
    assert wait(app, lambda: tab.ins_btn.isEnabled() and tab.ins_table.rowCount() > 0, timeout=30)
    assert tab.ins_table.item(0, 0).text() == "ABCUSDT"
    assert "Katalizör" in tab.ins_detail.toPlainText()
    assert not window.errors


def test_autopilot_button_runs_everything(app, window, monkeypatch):
    fm = _patch_intel_network(monkeypatch)
    cfg = _intel_cfg_1h()
    cfg.use_futures_context = False
    cfg.autopilot.enabled = False  # testte ağır bakımı çalıştırma
    from kreatifbot.intel.config import save_intel_config
    save_intel_config(cfg)
    window.data_client = lambda: fm
    bot = window.bot
    bot.engine_type.setCurrentIndex(bot.engine_type.findData("classic"))
    bot.paper.setChecked(True)
    bot.symbols.setText("BTCUSDT")
    bot.toggle_autopilot()
    s = window.settings
    assert s.autopilot and s.start_bot_on_launch and bot.engine_type.currentData() == "intel"
    assert wait(app, lambda: bot.engine is not None and bot.engine.running, timeout=30)
    assert getattr(bot.engine, "news", None) is not None  # haber + listeleme + öngörü birlikte
    assert "AÇIK" in bot.autopilot_btn.text()
    bot.toggle_autopilot()
    assert wait(app, lambda: not bot.engine.running, timeout=40)
    assert not s.autopilot and not s.start_bot_on_launch and not window.settings_tab.autobot.isChecked()


def test_small_live_profile(app, window):
    from kreatifbot.intel.config import load_intel_config
    tab = window.intel_tab
    tab.set_all_stages("SHADOW")
    assert all(tab.st_table.cellWidget(r, 4).currentData() == "SHADOW" for r in range(tab.st_table.rowCount()))
    tab.apply_small_live_profile(confirm=False)
    cfg = load_intel_config()
    assert all(sc.stage == "FULL_LIVE" for sc in cfg.strategies.values())
    assert cfg.listing.stage == cfg.catalyst.stage == "FULL_LIVE"
    assert cfg.risk.max_open_positions == 3 and cfg.risk.require_probability_for_live
    assert window.news_tab.s_stage.currentData() == "FULL_LIVE"


def test_small_live_profile(app, window):
    from kreatifbot.intel.config import load_intel_config
    tab = window.intel_tab
    tab.set_all_stages("SHADOW")
    assert all(tab.st_table.cellWidget(r, 4).currentData() == "SHADOW" for r in range(tab.st_table.rowCount()))
    tab.apply_small_live_profile(confirm=False)
    cfg = load_intel_config()
    assert all(sc.stage == "FULL_LIVE" for sc in cfg.strategies.values())
    assert cfg.listing.stage == cfg.catalyst.stage == "FULL_LIVE"
    assert cfg.strategy("trend_following").risk_multiplier == 1.0
    assert cfg.strategy("mean_reversion").risk_multiplier == 0.6
    assert cfg.strategy("vwap_reclaim").risk_multiplier == 0.5          # scalp
    assert all(0.1 <= sc.risk_multiplier <= 1.0 for sc in cfg.strategies.values())
    assert cfg.risk.max_open_positions == 3 and cfg.risk.require_probability_for_live
    assert window.news_tab.s_stage.currentData() == "FULL_LIVE"


def test_approval_setting_and_dialog(app, window):
    st = window.settings_tab
    st.approval.setCurrentIndex(st.approval.findData("manual"))
    st.approval_timeout.setValue(5)
    st.save()
    assert window.settings.trade_approval == "manual" and window.settings.approval_timeout_min == 5
    from kreatifbot.intel.approval import ApprovalBook
    book = ApprovalBook(5)
    _, req, _ = book.gate("k", lambda: dict(symbol="SOLUSDT", direction="LONG", strategy="ema_trend", price=100,
                                            qty=0.055, notional=5.5, stop=98, targets=[102, 104]))
    window._route_event("approval_request", req)
    assert req.request_id in window._approval_boxes
    window._route_event("approval_resolved", req)
    assert req.request_id not in window._approval_boxes
