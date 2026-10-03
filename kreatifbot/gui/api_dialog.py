"""Binance API anahtarı giriş penceresi."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)

from ..binance_client import BinanceClient
from .widgets import GREEN, RED

if TYPE_CHECKING:
    from .main_window import MainWindow


class ApiKeyDialog(QDialog):
    def __init__(self, ctx: "MainWindow", first_run: bool = False):
        super().__init__(ctx)
        self.ctx = ctx
        self.setWindowTitle("Binance API Anahtarı")
        self.setMinimumWidth(560)
        s = ctx.settings

        intro = QLabel(
            "<h3>🔑 Binance API anahtarınızı girin</h3>"
            "Anahtar olmadan da <b>analiz, tarayıcı, backtest ve kağıt işlem</b> kullanılabilir. "
            "Canlı işlem ve bakiye görüntüleme için anahtar gereklidir.<br><br>"
            "• <b>Testnet</b> (önerilen ilk adım): "
            "<a href='https://testnet.binance.vision'>testnet.binance.vision</a> → GitHub ile giriş → "
            "<i>Generate HMAC_SHA256 Key</i><br>"
            "• <b>Gerçek hesap</b>: Binance → Profil → <i>API Yönetimi</i> → API Oluştur. Yalnızca "
            "<i>Okuma</i> ve <i>Spot ve Marjin İşlemi</i> izinlerini açın.<br>"
            f"• <b style='color:{RED}'>Para çekme (Withdraw) iznini asla açmayın.</b> Mümkünse IP kısıtlaması ekleyin."
        )
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        intro.setTextFormat(Qt.TextFormat.RichText)

        self.api_key = QLineEdit(s.api_key)
        self.api_key.setPlaceholderText("API Key")
        self.api_secret = QLineEdit(s.api_secret)
        self.api_secret.setPlaceholderText("Secret Key")
        self.api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        show = QCheckBox("Göster")
        show.toggled.connect(lambda on: self.api_secret.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        secret_row = QHBoxLayout()
        secret_row.addWidget(self.api_secret)
        secret_row.addWidget(show)
        self.testnet = QCheckBox("Bu bir TESTNET anahtarı (gerçek para yok)")
        self.testnet.setChecked(s.testnet)

        form = QFormLayout()
        form.addRow("API Key", self.api_key)
        form.addRow("Secret Key", secret_row)
        form.addRow("", self.testnet)

        self.result = QLabel()
        self.result.setWordWrap(True)

        self.test_btn = QPushButton("Bağlantıyı Test Et")
        self.test_btn.clicked.connect(self.test)
        save = QPushButton("Kaydet")
        save.setDefault(True)
        save.clicked.connect(self.save_and_close)
        later = QPushButton("Şimdilik geç (kağıt işlemle devam)" if first_run else "İptal")
        later.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.test_btn)
        buttons.addStretch()
        buttons.addWidget(later)
        buttons.addWidget(save)

        note = QLabel("Gizli anahtar bu bilgisayarda Windows DPAPI ile şifrelenerek saklanır; hiçbir yere gönderilmez "
                      "(yalnızca Binance'e istek imzalamak için kullanılır).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b949e;")

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.result)
        layout.addWidget(note)
        layout.addLayout(buttons)

    def _apply(self):
        s = self.ctx.settings
        s.api_key = self.api_key.text().strip()
        s.api_secret = self.api_secret.text().strip()
        s.testnet = self.testnet.isChecked()
        self.ctx.persist()
        self.ctx.settings_changed()

    def save_and_close(self):
        self._apply()
        self.ctx.status("API anahtarı kaydedildi.")
        self.accept()

    def test(self):
        key, secret = self.api_key.text().strip(), self.api_secret.text().strip()
        if not key or not secret:
            self.result.setStyleSheet(f"color:{RED};")
            self.result.setText("API Key ve Secret Key alanlarını doldurun.")
            return
        client = BinanceClient(key, secret, testnet=self.testnet.isChecked())
        net = "Testnet" if client.testnet else "Gerçek Binance"
        self.test_btn.setEnabled(False)
        self.result.setStyleSheet("")
        self.result.setText(f"{net} bağlantısı test ediliyor...")

        def work():
            client.sync_time()
            return client.balances()

        def done(balances):
            self.test_btn.setEnabled(True)
            top = sorted(balances.items(), key=lambda kv: -kv[1]["free"])[:5]
            text = ", ".join(f"{a}: {b['free']:.6g}" for a, b in top) or "bakiye yok"
            self.result.setStyleSheet(f"color:{GREEN};")
            self.result.setText(f"✔ {net}: anahtar geçerli. Bakiyeler: {text}")

        def failed(msg):
            self.test_btn.setEnabled(True)
            self.result.setStyleSheet(f"color:{RED};")
            hint = ""
            if "-2015" in msg or "-2014" in msg:
                hint = ("<br>İpucu: anahtar yanlış, IP kısıtlaması engelliyor ya da testnet/gerçek seçimi "
                        "anahtarla uyuşmuyor.")
            self.result.setText(f"✘ {msg}{hint}")

        self.ctx.tasks.run(work, done, failed)
