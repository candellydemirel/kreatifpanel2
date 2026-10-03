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
