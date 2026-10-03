"""Ana pencere."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QMenu, QMessageBox, QStyleFactory, QSystemTrayIcon, QTabWidget,
)

from .. import __version__
from ..binance_client import BinanceClient
from ..config import GUIDE_PDF, Settings, data_dir, load_settings, resource_path, save_settings
from ..system import prevent_sleep
from .api_dialog import ApiKeyDialog
from ..telegram import TelegramClient, TelegramNotifier
from .tabs import AnalysisTab, BacktestTab, BotTab, ScannerTab, SettingsTab
from .intel_tab import IntelTab
from .news_tab import NewsTab
from .research_tab import ResearchTab
from .telegram_tab import TelegramTab
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


def app_icon(running: bool = False) -> QIcon:
    """Basit uygulama simgesi (çalışırken yeşil, dururken mavi)."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#26a69a" if running else "#1f6feb"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    p.setPen(QColor("white"))
    f = QFont()
    f.setBold(True)
    f.setPixelSize(38)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "K")
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, prompt_api: bool = False, start_hidden: bool = False):
        super().__init__()
        self._force_quit = False
        self._tray_hint_shown = False
        self.setWindowIcon(app_icon())
        self.setWindowTitle(f"KreatifBot {__version__} — Binance Trading Bot")
        self.resize(1440, 900)
        self.settings: Settings = load_settings()
        self.tasks = TaskRunner(self)
        self.bridge = EngineBridge()
        self._bot_running = False
        # Zeka motoru oturum durumu (meta model, strateji istatistikleri ve sağlığı Araştırma'dan gelir)
        self.intel_meta = None
        self.intel_stats = None
        self.intel_health = None
        self.intel_last_decision = None

        self.tabs = QTabWidget()
        self.analysis = AnalysisTab(self)
        self.scanner = ScannerTab(self)
        self.backtest = BacktestTab(self)
        self.bot = BotTab(self)
        self.settings_tab = SettingsTab(self)
        self.telegram_tab = TelegramTab(self)
        self.intel_tab = IntelTab(self)
        self.research_tab = ResearchTab(self)
        self.news_tab = NewsTab(self)
        self.tabs.addTab(self.analysis, "📈  Piyasa Analizi")
        self.tabs.addTab(self.scanner, "🔎  Tarayıcı")
        self.tabs.addTab(self.backtest, "🧪  Backtest")
        self.tabs.addTab(self.intel_tab, "🧠  Zeka Motoru")
        self.tabs.addTab(self.research_tab, "🔬  Araştırma")
        self.tabs.addTab(self.news_tab, "📰  Haberler")
        self.tabs.addTab(self.bot, "🤖  Bot")
        self.tabs.addTab(self.telegram_tab, "📨  Telegram")
        self.tabs.addTab(self.settings_tab, "⚙  Ayarlar")
        self.setCentralWidget(self.tabs)
        self.bridge.event.connect(self.bot.on_event, Qt.ConnectionType.QueuedConnection)
        self.bridge.event.connect(self._route_event, Qt.ConnectionType.QueuedConnection)

        self.net_badge = QLabel()
        self.statusBar().addPermanentWidget(self.net_badge)
        self._build_menu()
        self._build_tray()
        self.settings_changed()
        self.status(f"Hazır. Veri klasörü: {data_dir()}")
        if prompt_api and not self.settings.api_key and not start_hidden:
            QTimer.singleShot(400, lambda: self.open_api_dialog(first_run=True))

    def _build_menu(self):
        bar = self.menuBar()
        api = QAction("🔑  Binance API Anahtarı", self)
        api.triggered.connect(lambda: self.open_api_dialog())
        bar.addAction(api)
        help_menu = bar.addMenu("Yardım")
        guide = QAction("📘  Strateji ve Kullanım Rehberi (PDF)", self)
        guide.triggered.connect(self.open_guide)
        folder = QAction("📂  Veri / kayıt klasörünü aç", self)
        folder.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(data_dir()))))
        about = QAction("Hakkında", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "KreatifBot", f"<b>KreatifBot {__version__}</b><br>Binance Spot analiz, backtest ve trading botu."
            "<br><br>Yatırım tavsiyesi değildir. Kripto işlemleri yüksek risk içerir."))
        for action in (guide, folder, about):
            help_menu.addAction(action)

    # ---------------------------------------------------------------- sistem tepsisi
    def _build_tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(app_icon(), self)
        menu = QMenu(self)
        self._tray_show = QAction("Pencereyi göster", self)
        self._tray_show.triggered.connect(self.show_from_tray)
        self._tray_toggle = QAction("Botu başlat", self)
        self._tray_toggle.triggered.connect(self._tray_toggle_bot)
        quit_action = QAction("Çıkış", self)
        quit_action.triggered.connect(self.quit_app)
        for a in (self._tray_show, self._tray_toggle):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.setToolTip("KreatifBot — bot durduruldu")
        tray.activated.connect(lambda reason: self.show_from_tray()
                               if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                                             QSystemTrayIcon.ActivationReason.DoubleClick) else None)
        tray.show()
        self.tray = tray

    def show_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_toggle_bot(self):
        if self._bot_running:
            self.bot.stop()
        else:
            self.show_from_tray()
            self.tabs.setCurrentWidget(self.bot)
            self.bot.start()

    def quit_app(self):
        if self._bot_running:
            self.show_from_tray()
            answer = QMessageBox.question(
                self, "Çıkış", "Bot çalışıyor. Durdurup uygulamadan tamamen çıkılsın mı?\n"
                "(Açık pozisyonlar kaydedilir; spot pozisyonların stop/hedefleri uygulama kapalıyken izlenmez.)")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._force_quit = True
        self.close()
        QApplication.instance().quit()

    def open_api_dialog(self, first_run: bool = False):
        if self._bot_running:
            self.show_error("Bot çalışıyor", "API anahtarını değiştirmek için önce botu durdurun.")
            return
        dialog = ApiKeyDialog(self, first_run)
        dialog.open()
        self._api_dialog = dialog

    def open_guide(self):
        path = resource_path(GUIDE_PDF)
        if not path.exists():
            self.show_error("Rehber bulunamadı", str(path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ---------------------------------------------------------------- bağlam
    def data_client(self) -> BinanceClient:
        s = self.settings
        if s.mainnet_data:
            return BinanceClient(testnet=False)
        return BinanceClient(s.api_key, s.api_secret, testnet=s.testnet)

    def trade_client(self) -> BinanceClient:
        s = self.settings
        return BinanceClient(s.api_key, s.api_secret, testnet=s.testnet)

    def _route_event(self, kind: str, payload):
        if kind == "decision":
            self.intel_tab.on_live_decision(payload)
        elif kind in ("news", "listing"):
            self.news_tab._load_cached()
            if kind == "listing":
                self.status(f"Yeni listeleme olayı: {payload.symbol} ({payload.kind})")

    def telegram_ready(self) -> bool:
        s = self.settings
        return bool(s.telegram_enabled and s.telegram_token and s.telegram_chat_id)

    def make_notifier(self) -> TelegramNotifier | None:
        if not self.telegram_ready():
            return None
        s = self.settings
        return TelegramNotifier(TelegramClient(s.telegram_token), s.telegram_chat_id, s.telegram_notify,
                                s.quote_asset, s.telegram_commands, s.telegram_summary_hour)

    def telegram_send(self, text: str):
        """Tek seferlik mesaj (analiz / tarama sonucu)."""
        if not self.telegram_ready():
            self.show_error("Telegram ayarlı değil",
                            "Telegram sekmesinden token ve Chat ID girip bildirimleri etkinleştirin.")
            return
        client, chat_id = TelegramClient(self.settings.telegram_token), self.settings.telegram_chat_id
        self.tasks.run(lambda: client.send_message(chat_id, text),
                       lambda _: self.status("Telegram'a gönderildi."),
                       lambda m: self.show_error("Telegram gönderimi başarısız", m))

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
        self.settings_tab.reload()

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
        prevent_sleep(running and self.settings.prevent_sleep)
        if self.tray is not None:
            self.tray.setIcon(app_icon(running))
            self.tray.setToolTip("KreatifBot — bot çalışıyor" if running else "KreatifBot — bot durduruldu")
            self._tray_toggle.setText("Botu durdur" if running else "Botu başlat")
        self.setWindowIcon(app_icon(running))
        self.settings_tab.setEnabled(not running)
        self.telegram_tab.setEnabled(not running)

    # ---------------------------------------------------------------- kapanış
    def closeEvent(self, event):
        if not self._force_quit and self.tray is not None and self.settings.minimize_to_tray:
            # Pencereyi gizle, bot arka planda çalışmaya devam etsin
            event.ignore()
            self.hide()
            if not self._tray_hint_shown:
                self._tray_hint_shown = True
                self.tray.showMessage("KreatifBot arka planda çalışıyor",
                                      "Simgeye tıklayarak açabilir, sağ tık → Çıkış ile kapatabilirsiniz.",
                                      QSystemTrayIcon.MessageIcon.Information, 5000)
            return
        if self._bot_running and not self._force_quit:
            answer = QMessageBox.question(
                self, "Çıkış", "Bot çalışıyor. Durdurup çıkmak istiyor musunuz?\n"
                "(Açık pozisyonlar kaydedilir ve bot yeniden başlatıldığında takip edilmeye devam eder.)")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.bot.shutdown()
        prevent_sleep(False)
        if self.tray is not None:
            self.tray.hide()
        self.tasks.pool.waitForDone(3000)
        event.accept()
        app = QApplication.instance()
        if app is not None and not app.quitOnLastWindowClosed():
            app.quit()
