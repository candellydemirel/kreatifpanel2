"""Ana pencere."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QStyleFactory, QTabWidget

from .. import __version__
from ..binance_client import BinanceClient
from ..config import Settings, data_dir, load_settings, save_settings
from .tabs import AnalysisTab, BacktestTab, BotTab, ScannerTab, SettingsTab
from .widgets import TaskRunner


class EngineBridge(QObject):
    """Bot iş parçacığından gelen olayları arayüz iş parçacığına taşır."""
    event = Signal(str, object)


def apply_dark_theme(app: QApplication):
    app.setStyle(QStyleFactory.create("Fusion"))
    pal = QPalette()
    base, alt, text = QColor("#0d1117"), QColor("#161b22"), QColor("#c9d1d9")
    pal.setColor(QPalette.ColorRole.Window, alt)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, base)
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#151a21"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, alt)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, QColor("#21262d"))
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#1f6feb"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#58a6ff"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6e7681"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6e7681"))
    app.setPalette(pal)
    app.setStyleSheet("""
        QGroupBox { border: 1px solid #30363d; border-radius: 6px; margin-top: 10px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8b949e; }
        QPushButton { padding: 6px 12px; border-radius: 4px; border: 1px solid #30363d; }
        QPushButton:hover { border-color: #58a6ff; }
        QTabBar::tab { padding: 8px 16px; }
        QTabBar::tab:selected { color: #58a6ff; }
        QHeaderView::section { background: #21262d; padding: 4px; border: none; }
    """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"KreatifBot {__version__} — Binance Trading Bot")
        self.resize(1440, 900)
        self.settings: Settings = load_settings()
        self.tasks = TaskRunner(self)
        self.bridge = EngineBridge()
        self._bot_running = False

        self.tabs = QTabWidget()
        self.analysis = AnalysisTab(self)
        self.scanner = ScannerTab(self)
        self.backtest = BacktestTab(self)
        self.bot = BotTab(self)
        self.settings_tab = SettingsTab(self)
        self.tabs.addTab(self.analysis, "📈  Piyasa Analizi")
        self.tabs.addTab(self.scanner, "🔎  Tarayıcı")
        self.tabs.addTab(self.backtest, "🧪  Backtest")
        self.tabs.addTab(self.bot, "🤖  Bot")
        self.tabs.addTab(self.settings_tab, "⚙  Ayarlar")
        self.setCentralWidget(self.tabs)
        self.bridge.event.connect(self.bot.on_event, Qt.ConnectionType.QueuedConnection)

        self.net_badge = QLabel()
        self.statusBar().addPermanentWidget(self.net_badge)
        self.settings_changed()
        self.status(f"Hazır. Veri klasörü: {data_dir()}")
        if not self.settings.api_key:
            self.tabs.setCurrentWidget(self.settings_tab)

    # ---------------------------------------------------------------- bağlam
    def data_client(self) -> BinanceClient:
        s = self.settings
        if s.mainnet_data:
            return BinanceClient(testnet=False)
        return BinanceClient(s.api_key, s.api_secret, testnet=s.testnet)

    def trade_client(self) -> BinanceClient:
        s = self.settings
        return BinanceClient(s.api_key, s.api_secret, testnet=s.testnet)

    def persist(self):
        try:
            save_settings(self.settings)
        except OSError as exc:
            self.show_error("Ayarlar kaydedilemedi", str(exc))

    def settings_changed(self):
        s = self.settings
        if s.testnet:
            self.net_badge.setText("  TESTNET  ")
            self.net_badge.setStyleSheet("background:#9a6700; color:white; border-radius:3px;")
        else:
            self.net_badge.setText("  GERÇEK HESAP  ")
            self.net_badge.setStyleSheet("background:#b62324; color:white; border-radius:3px;")
        self.bot.refresh_network_label()

    def status(self, message: str):
        self.statusBar().showMessage(message, 15000)

    def show_error(self, title: str, message: str):
        self.status(f"{title}: {message}")
        QMessageBox.warning(self, title, message)

    def open_analysis(self, symbol: str, interval: str | None = None):
        self.tabs.setCurrentWidget(self.analysis)
        self.analysis.open_symbol(symbol, interval)

    def set_bot_running(self, running: bool):
        self._bot_running = running
        self.settings_tab.setEnabled(not running)

    # ---------------------------------------------------------------- kapanış
    def closeEvent(self, event):
        if self._bot_running:
            answer = QMessageBox.question(
                self, "Çıkış", "Bot çalışıyor. Durdurup çıkmak istiyor musunuz?\n"
                "(Açık pozisyonlar kaydedilir ve bot yeniden başlatıldığında takip edilmeye devam eder.)")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.bot.shutdown()
        self.tasks.pool.waitForDone(3000)
        event.accept()
