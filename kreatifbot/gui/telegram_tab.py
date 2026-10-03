"""Telegram ayar sekmesi."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..telegram import NOTIFY_DEFAULTS, NOTIFY_LABELS, TelegramClient
from .widgets import GREEN, RED

if TYPE_CHECKING:
    from .main_window import MainWindow


class TelegramTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        s = ctx.settings

        conn = QGroupBox("Telegram bağlantısı")
        form = QFormLayout(conn)
        self.enabled = QCheckBox("Telegram bildirimlerini etkinleştir")
        self.enabled.setChecked(s.telegram_enabled)
        self.token = QLineEdit(s.telegram_token)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("123456789:ABCdef... (BotFather'dan)")
        show = QCheckBox("Göster")
        show.toggled.connect(lambda on: self.token.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        token_row = QHBoxLayout()
        token_row.addWidget(self.token)
        token_row.addWidget(show)
        self.chat_id = QLineEdit(s.telegram_chat_id)
        self.chat_id.setPlaceholderText("ör. 123456789")
        self.find_btn = QPushButton("Chat ID'yi otomatik bul")
        self.find_btn.clicked.connect(self.find_chat)
        chat_row = QHBoxLayout()
        chat_row.addWidget(self.chat_id)
        chat_row.addWidget(self.find_btn)
        form.addRow("", self.enabled)
        form.addRow("Bot token", token_row)
        form.addRow("Chat ID", chat_row)

        notif = QGroupBox("Hangi bildirimler gönderilsin?")
        nl = QVBoxLayout(notif)
        self.notify: dict[str, QCheckBox] = {}
        current = {**NOTIFY_DEFAULTS, **(s.telegram_notify or {})}
        for key, label in NOTIFY_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(current[key])
            self.notify[key] = cb
            nl.addWidget(cb)
        hour_row = QHBoxLayout()
        self.summary_hour = QSpinBox()
        self.summary_hour.setRange(0, 23)
        self.summary_hour.setValue(s.telegram_summary_hour)
        self.summary_hour.setSuffix(":00")
        hour_row.addWidget(QLabel("Günlük özet saati"))
        hour_row.addWidget(self.summary_hour)
        hour_row.addStretch()
        nl.addLayout(hour_row)
        self.commands = QCheckBox("Telegram'dan komutlara izin ver (/durum, /pozisyonlar, /islemler, /ozet, /durdur)")
        self.commands.setChecked(s.telegram_commands)
        nl.addWidget(self.commands)

        buttons = QHBoxLayout()
        save = QPushButton("Kaydet")
        save.clicked.connect(self.save)
        self.test_btn = QPushButton("Test mesajı gönder")
        self.test_btn.clicked.connect(self.test)
        buttons.addWidget(save)
        buttons.addWidget(self.test_btn)
        buttons.addStretch()
        self.result = QLabel()
        self.result.setWordWrap(True)

        guide = QLabel(
            "<b>Kurulum (2 dakika):</b><br>"
            "1. Telegram'da <a href='https://t.me/BotFather'>@BotFather</a> ile konuşun, <code>/newbot</code> yazın, "
            "bota bir ad ve sonu <i>bot</i> ile biten bir kullanıcı adı verin.<br>"
            "2. BotFather'ın verdiği <b>token</b>'ı yukarıya yapıştırın.<br>"
            "3. Telegram'da yeni botunuzu açıp <b>/start</b> yazın.<br>"
            "4. <b>Chat ID'yi otomatik bul</b> butonuna basın, ardından <b>Test mesajı gönder</b> ve <b>Kaydet</b>.<br>"
            "5. Botu Bot sekmesinden başlattığınızda bildirimler otomatik gelir.<br><br>"
            "Bir grupta kullanmak için botu gruba ekleyin, grupta bir mesaj yazın ve Chat ID'yi tekrar bulun.<br>"
            "Güvenlik: komutlar yalnızca bu Chat ID'den kabul edilir; token bu bilgisayarda şifrelenerek saklanır."
        )
        guide.setWordWrap(True)
        guide.setOpenExternalLinks(True)
        guide.setTextFormat(Qt.TextFormat.RichText)

        layout = QVBoxLayout(self)
        layout.addWidget(conn)
        layout.addWidget(notif)
        layout.addLayout(buttons)
        layout.addWidget(self.result)
        layout.addWidget(guide)
        layout.addStretch()

    def _client(self) -> TelegramClient:
        return TelegramClient(self.token.text().strip())

    def _show(self, ok: bool, text: str):
        self.result.setStyleSheet(f"color:{GREEN if ok else RED};")
        self.result.setText(("✔ " if ok else "✘ ") + text)

    @Slot()
    def save(self):
        s = self.ctx.settings
        s.telegram_enabled = self.enabled.isChecked()
        s.telegram_token = self.token.text().strip()
        s.telegram_chat_id = self.chat_id.text().strip()
        s.telegram_commands = self.commands.isChecked()
        s.telegram_summary_hour = self.summary_hour.value()
        s.telegram_notify = {k: cb.isChecked() for k, cb in self.notify.items()}
        self.ctx.persist()
        self.ctx.status("Telegram ayarları kaydedildi." + ("" if s.telegram_enabled else " (bildirimler kapalı)"))

    @Slot()
    def find_chat(self):
        client = self._client()
        self.find_btn.setEnabled(False)
        self.result.setStyleSheet("")
        self.result.setText("Botunuza gelen son mesaj aranıyor...")

        def work():
            me = client.get_me()
            chat_id, name = client.find_chat_id()
            return me, chat_id, name

        def done(result):
            me, chat_id, name = result
            self.find_btn.setEnabled(True)
            self.chat_id.setText(chat_id)
            self._show(True, f"@{me.get('username')} botu için sohbet bulundu: {name} (ID {chat_id}). "
                             "Şimdi 'Test mesajı gönder' ve 'Kaydet'.")

        def failed(msg):
            self.find_btn.setEnabled(True)
            self._show(False, msg)

        self.ctx.tasks.run(work, done, failed)

    @Slot()
    def test(self):
        client, chat_id = self._client(), self.chat_id.text().strip()
        if not chat_id:
            self._show(False, "Önce Chat ID girin veya otomatik bulun.")
            return
        self.test_btn.setEnabled(False)

        def work():
            client.send_message(chat_id, "✅ <b>KreatifBot</b> Telegram bağlantısı çalışıyor.\n"
                                         "Bot başlatıldığında sinyal ve işlem bildirimleri buraya gelecek.")

        def done(_):
            self.test_btn.setEnabled(True)
            self._show(True, "Test mesajı gönderildi. Telegram'ı kontrol edin.")

        def failed(msg):
            self.test_btn.setEnabled(True)
            self._show(False, msg)

        self.ctx.tasks.run(work, done, failed)
