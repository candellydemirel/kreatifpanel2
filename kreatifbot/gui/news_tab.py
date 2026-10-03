"""Haberler sekmesi: kripto haber akışı, Binance listelemeleri, listeleme analizi ve backtest."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..intel.config import load_intel_config, save_intel_config
from .intel_tab import df_to_table, fmt_cell, mono
from .widgets import GREEN, RED, fill_table, make_table

if TYPE_CHECKING:
    from .main_window import MainWindow

CATEGORY_COLORS = {"DELISTING": RED, "HACK": RED, "REGULATION_NEG": RED, "LISTING": GREEN, "FUTURES_LISTING": GREEN,
                   "LAUNCHPOOL": GREEN}


def build_monitor(settings, cfg):
    """Ayarlara göre NewsMonitor oluşturur (Bot ve Haberler sekmesi ortak kullanır)."""
    from ..intel.news import RSS_FEEDS, NewsMonitor
    nc = cfg.news
    feeds = {k: v for k, v in RSS_FEEDS.items()
             if {"CoinDesk": nc.rss_coindesk, "Cointelegraph": nc.rss_cointelegraph, "Decrypt": nc.rss_decrypt}[k]}
    return NewsMonitor(rss_feeds=feeds, use_binance=nc.use_binance_announcements,
                       cryptopanic_token=settings.cryptopanic_token, quote=settings.quote_asset,
                       block_hours=nc.block_hours, min_severity_block=nc.min_severity_block)


class NewsTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        self.items = []
        cfg = load_intel_config()

        top = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 Haberleri ve listelemeleri tara")
        self.refresh_btn.clicked.connect(self.refresh)
        self.auto = QCheckBox("Otomatik (2 dk)")
        self.auto.setToolTip("Bot Zeka Motoru ile çalışırken haberleri bot tarar; bu ekran kendiliğinden güncellenir.")
        self.auto.toggled.connect(self._toggle_auto)
        self.category = QComboBox()
        self.category.addItems(["Tümü", "LISTING", "DELISTING", "FUTURES_LISTING", "LAUNCHPOOL", "HACK",
                                "REGULATION_NEG", "PARTNERSHIP", "MACRO", "GENERAL"])
        self.category.currentIndexChanged.connect(lambda _: self._render())
        self.coin = QLineEdit()
        self.coin.setPlaceholderText("Coin filtresi (ör. SOL)")
        self.coin.textChanged.connect(lambda _: self._render())
        for w in (self.refresh_btn, self.auto, QLabel("Kategori"), self.category, self.coin):
            top.addWidget(w)
        top.addStretch()
        self.sources = QLabel("Kaynaklar: henüz taranmadı")
        self.sources.setWordWrap(True)
        self.sources.setStyleSheet("color:#8b949e;")

        self.news_table = make_table(["Zaman (UTC)", "Kaynak", "Kategori", "Coinler", "Duyarlılık", "Başlık"])
        self.news_table.doubleClicked.connect(self._open_news)
        news_page = QWidget()
        nl = QVBoxLayout(news_page)
        nl.addWidget(self.news_table)
        hint = QLabel("Çift tıklayınca haber tarayıcıda açılır. Haberler tek başına işlem açtırmaz: olumsuz haber "
                      "yeni LONG'ları engeller, delist duyurusu açık pozisyonu kapattırır.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b949e;")
        nl.addWidget(hint)

        # listelemeler
        self.list_table = make_table(["Sembol", "Olay", "Tespit (UTC)", "Durum", "Kaynak"])
        self.analyze_btn = QPushButton("Seçili listelemeyi analiz et (1 dk veri)")
        self.analyze_btn.clicked.connect(self.analyze_selected)
        self.list_detail = QPlainTextEdit()
        self.list_detail.setReadOnly(True)
        self.list_detail.setFont(mono())
        bt_row = QHBoxLayout()
        self.bt_symbols = QLineEdit()
        self.bt_symbols.setPlaceholderText("Geçmiş listelemeler (ör. FOOUSDT, BARUSDT) — boşsa listedeki semboller")
        self.bt_btn = QPushButton("Listeleme backtest (gerçek veri)")
        self.bt_btn.clicked.connect(self.listing_backtest)
        bt_row.addWidget(self.bt_symbols, 1)
        bt_row.addWidget(self.bt_btn)
        self.bt_table = make_table(["-"])
        list_page = QWidget()
        ll = QVBoxLayout(list_page)
        row = QHBoxLayout()
        row.addWidget(self.analyze_btn)
        row.addStretch()
        sp = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.list_table)
        lv.addLayout(row)
        sp.addWidget(left)
        sp.addWidget(self.list_detail)
        ll.addWidget(sp, 2)
        ll.addLayout(bt_row)
        ll.addWidget(self.bt_table, 1)

        # öngörüler
        ins_page = QWidget()
        il = QVBoxLayout(ins_page)
        irow = QHBoxLayout()
        self.ins_btn = QPushButton("💡 Öngörüleri üret (haber + temel + teknik)")
        self.ins_btn.clicked.connect(self.run_insights)
        self.perf_btn = QPushButton("Öngörü sonuçlarını güncelle")
        self.perf_btn.clicked.connect(self.update_performance)
        irow.addWidget(self.ins_btn)
        irow.addWidget(self.perf_btn)
        irow.addStretch()
        self.ins_table = make_table(["Sembol", "Sinyal", "Potansiyel", "Katalizör", "Temel", "Teknik",
                                     "Katalizör türleri", "Kaynaklar", "Uyarılar"])
        self.ins_table.doubleClicked.connect(self._show_insight)
        self.ins_detail = QPlainTextEdit()
        self.ins_detail.setReadOnly(True)
        self.ins_detail.setFont(mono())
        self.perf_table = make_table(["-"])
        isp = QSplitter()
        isp.addWidget(self.ins_table)
        isp.addWidget(self.ins_detail)
        il.addLayout(irow)
        il.addWidget(isp, 2)
        perf_label = QLabel("Öngörü performansı — kaydedilen öngörülerin 4s/24s/72s sonraki GERÇEK getirileri "
                            "(geçmiş haber arşivi olmadığı için doğrulama ileriye dönük yapılır):")
        perf_label.setWordWrap(True)
        il.addWidget(perf_label)
        il.addWidget(self.perf_table, 1)
        self._insights: list = []

        # ayarlar
        set_page = QWidget()
        sf = QFormLayout(set_page)
        nc, lc = cfg.news, cfg.listing
        self.s_enabled = QCheckBox("Zeka Motoru'nda haber filtresini kullan")
        self.s_enabled.setChecked(nc.enabled)
        self.s_binance = QCheckBox("Binance duyuruları (listeleme / delist)")
        self.s_binance.setChecked(nc.use_binance_announcements)
        self.s_cd = QCheckBox("CoinDesk RSS")
        self.s_cd.setChecked(nc.rss_coindesk)
        self.s_ct = QCheckBox("Cointelegraph RSS")
        self.s_ct.setChecked(nc.rss_cointelegraph)
        self.s_dc = QCheckBox("Decrypt RSS")
        self.s_dc.setChecked(nc.rss_decrypt)
        self.s_token = QLineEdit(ctx.settings.cryptopanic_token)
        self.s_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.s_token.setPlaceholderText("İsteğe bağlı — cryptopanic.com'dan ücretsiz")
        self.s_block = QSpinBox()
        self.s_block.setRange(1, 168)
        self.s_block.setValue(int(nc.block_hours))
        self.s_block.setSuffix(" saat")
        self.s_delist = QCheckBox("Delist duyurusunda açık pozisyonu kapat")
        self.s_delist.setChecked(nc.delist_exit)
        self.s_listing = QCheckBox("Yeni listelemelerde işlem yap (Yeni Listeleme stratejisi)")
        self.s_listing.setChecked(lc.enabled)
        self.s_stage = QComboBox()
        self.s_stage.addItems(["PAPER", "SHADOW", "LIMITED_LIVE", "FULL_LIVE"])
        self.s_stage.setCurrentText(lc.stage if lc.stage in ("PAPER", "SHADOW", "LIMITED_LIVE", "FULL_LIVE") else "PAPER")
        self.s_wait = QSpinBox()
        self.s_wait.setRange(1, 240)
        self.s_wait.setValue(lc.wait_minutes)
        self.s_wait.setSuffix(" dk")
        cc = cfg.catalyst
        self.c_enabled = QCheckBox("Öngörü motorunu çalıştır (bot çalışırken her 30 dk)")
        self.c_enabled.setChecked(cc.enabled)
        self.c_trade = QCheckBox("AL öngörülerinde işlem aç (aşamaya bağlı)")
        self.c_trade.setChecked(cc.trade_enabled)
        self.c_stage = QComboBox()
        self.c_stage.addItems(["PAPER", "SHADOW", "LIMITED_LIVE", "FULL_LIVE"])
        self.c_stage.setCurrentText(cc.stage if cc.stage in ("PAPER", "SHADOW", "LIMITED_LIVE", "FULL_LIVE")
                                    else "PAPER")
        self.c_any = QCheckBox("İzinli sembol listesi dışındaki Binance USDT çiftlerinde de işlem aç")
        self.c_any.setChecked(cc.any_binance_pair)
        save = QPushButton("Kaydet")
        save.clicked.connect(self.save_settings)
        self.s_msg = QLabel()
        for label, w in (("", self.s_enabled), ("", self.s_binance), ("", self.s_cd), ("", self.s_ct),
                         ("", self.s_dc), ("CryptoPanic API anahtarı", self.s_token),
                         ("Olumsuz haber sonrası LONG yasağı", self.s_block), ("", self.s_delist),
                         ("", self.s_listing), ("Listeleme stratejisi aşaması", self.s_stage),
                         ("Açılıştan sonra bekleme", self.s_wait), ("", self.c_enabled), ("", self.c_trade),
                         ("Öngörü işlemleri aşaması", self.c_stage), ("", self.c_any), ("", save),
                         ("", self.s_msg)):
            sf.addRow(label, w)
        warn = QLabel("⚠ Yeni listelemeler ilk saatlerde aşırı oynaktır. Canlı işlem yalnızca aşama LIMITED_LIVE / "
                      "FULL_LIVE iken ve çok küçük riskle yapılır. Önce 'Listeleme backtest' ile geçmiş "
                      "listelemelerdeki gerçek sonuçlara bakın.")
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color:{RED};")
        sf.addRow(warn)

        tabs = QTabWidget()
        tabs.addTab(news_page, "Haber akışı")
        tabs.addTab(list_page, "Binance listelemeleri")
        tabs.addTab(ins_page, "💡 Öngörüler")
        tabs.addTab(set_page, "Ayarlar")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.sources)
        lay.addWidget(tabs)

        self.timer = QTimer(self)
        self.timer.setInterval(120_000)
        self.timer.timeout.connect(self.refresh)
        self._load_cached()
        self.auto.setChecked(True)

    # ---------------------------------------------------------------- veri
    def _load_cached(self):
        try:
            from ..intel.news import NewsStore
            store = NewsStore()
            self.items = store.recent(hours=72)
            self._render()
            self._render_listings(store.listings())
        except Exception:  # noqa: BLE001 - ilk açılışta veritabanı yoksa sorun değil
            pass

    def _toggle_auto(self, on: bool):
        if on:
            self.timer.start()
            self.refresh()
        else:
            self.timer.stop()

    def bot_handles_news(self) -> bool:
        eng = getattr(self.ctx.bot, "engine", None)
        return bool(eng is not None and eng.running and getattr(eng, "news", None) is not None)

    def on_news_polled(self, info: dict):
        self._load_cached()
        self.sources.setText(f"[Bot] {info.get('new', 0)} yeni haber, {info.get('events', 0)} listeleme olayı. "
                             "Kaynaklar: " + " | ".join(f"{k}: {v}" for k, v in info.get("status", {}).items()))

    @Slot()
    def refresh(self, manual: bool = True):
        if not self.refresh_btn.isEnabled():
            return
        if self.sender() is self.timer and self.bot_handles_news():
            return  # bot zaten tarıyor; çift istek yapma
        settings, cfg = self.ctx.settings, load_intel_config()
        self.refresh_btn.setEnabled(False)
        self.sources.setText("Kaynaklar taranıyor...")

        def work():
            from ..binance_client import BinanceAPIError, BinanceClient
            mon = build_monitor(settings, cfg)
            symbols = None
            try:
                symbols = BinanceClient(testnet=False).exchange_info().get("symbols", [])
                mon.source_status["Binance exchangeInfo"] = "OK"
            except BinanceAPIError as exc:
                mon.source_status["Binance exchangeInfo"] = f"ERİŞİLEMİYOR: {exc}"
            new_items, events = mon.poll(symbols)
            return mon.store.recent(hours=72), mon.store.listings(), mon.source_status, len(new_items), events

        def done(res):
            items, listings, status, n_new, events = res
            self.refresh_btn.setEnabled(True)
            self.items = items
            self._render()
            self._render_listings(listings)
            self.sources.setText(f"{n_new} yeni haber, {len(events)} listeleme olayı. Kaynaklar: " +
                                 " | ".join(f"{k}: {v}" for k, v in status.items()))
            for ev in events:
                self.ctx.status(f"Listeleme: {ev.symbol} ({ev.kind})")

        def failed(msg):
            self.refresh_btn.setEnabled(True)
            self.sources.setText(f"Tarama başarısız: {msg}")

        self.ctx.tasks.run(work, done, failed)

    def _render(self):
        cat = self.category.currentText()
        coin = self.coin.text().strip().upper()
        rows, colors, self._shown = [], [], []
        for it in self.items:
            if cat != "Tümü" and it.category != cat:
                continue
            if coin and coin not in it.symbols:
                continue
            self._shown.append(it)
            rows.append([it.published_at[:16].replace("T", " "), it.source, it.category, ", ".join(it.symbols),
                         f"{it.sentiment:+.2f}", it.title])
            colors.append(CATEGORY_COLORS.get(it.category))
        fill_table(self.news_table, rows, colors)

    def _render_listings(self, listings):
        rows = [[e.symbol, e.kind, e.detected_at[:16].replace("T", " "), e.status, e.source] for e in listings]
        fill_table(self.list_table, rows)

    def _open_news(self, index):
        if 0 <= index.row() < len(self._shown) and self._shown[index.row()].url:
            QDesktopServices.openUrl(QUrl(self._shown[index.row()].url))

    # ---------------------------------------------------------------- listeleme analizi
    def analyze_selected(self):
        row = self.list_table.currentRow()
        item = self.list_table.item(row, 0) if row >= 0 else None
        if item is None:
            self.ctx.show_error("Listeleme", "Önce listeden bir sembol seçin.")
            return
        symbol = item.text()
        cfg = load_intel_config()
        self.analyze_btn.setEnabled(False)

        def work():
            from ..binance_client import BinanceClient
            from ..intel.listing import evaluate_listing, fetch_listing_history
            from ..intel.orderflow import orderbook_features
            c = BinanceClient(testnet=False)
            df = fetch_listing_history(c, symbol, cfg.listing.watch_hours)
            if df.empty:
                return f"{symbol}: DATA_UNAVAILABLE — 1 dk veri alınamadı (henüz işleme açılmamış olabilir)."
            book = orderbook_features(c.depth(symbol, 100))
            return evaluate_listing(df, symbol, cfg.listing, book_stats=book).explain()

        def done(text):
            self.analyze_btn.setEnabled(True)
            self.list_detail.setPlainText(text)

        def failed(msg):
            self.analyze_btn.setEnabled(True)
            self.list_detail.setPlainText(f"Analiz başarısız: {msg}")

        self.ctx.tasks.run(work, done, failed)

    def listing_backtest(self):
        syms = [s.strip().upper() for s in self.bt_symbols.text().replace(";", ",").split(",") if s.strip()]
        if not syms:
            syms = [self.list_table.item(r, 0).text() for r in range(self.list_table.rowCount())]
        syms = list(dict.fromkeys(syms))[:40]
        if not syms:
            self.ctx.show_error("Listeleme backtest", "Sembol girin (ör. son listelenen coinler).")
            return
        cfg = load_intel_config()
        self.bt_btn.setEnabled(False)
        self.list_detail.setPlainText(f"{len(syms)} listeleme için gerçek 1 dk veri indiriliyor...")

        def work():
            from ..binance_client import BinanceClient
            from ..intel.listing import fetch_listing_history, listing_backtest
            c = BinanceClient(testnet=False)
            frames = {}
            for s in syms:
                try:
                    frames[s] = fetch_listing_history(c, s, cfg.listing.watch_hours + 4)
                except Exception:  # noqa: BLE001
                    frames[s] = None
            return listing_backtest(frames, cfg.listing, cfg.costs)

        def done(res):
            self.bt_btn.setEnabled(True)
            df_to_table(self.bt_table, res["trades"], "net_ret_pct")
            lines = ["LİSTELEME BACKTEST (gerçek Binance 1 dk verisi, ücret + yüksek kayma dahil):"]
            lines += [f"  {k}: {fmt_cell(v)}" for k, v in res["summary"].items()]
            lines += ["", "Sembol notları:"] + [f"  {k}: {v}" for k, v in res["notes"].items()]
            self.list_detail.setPlainText("\n".join(lines))

        def failed(msg):
            self.bt_btn.setEnabled(True)
            self.list_detail.setPlainText(f"Backtest başarısız: {msg}")

        self.ctx.tasks.run(work, done, failed)

    # ---------------------------------------------------------------- öngörüler
    def _render_insights(self):
        rows, colors = [], []
        for ins in self._insights:
            fd, tc = ins.fundamentals, ins.technical
            rows.append([ins.symbol, ins.signal, f"{ins.potential:.0f}", f"{ins.catalyst.score:.0f}",
                         "-" if fd.score is None else f"{fd.score:.0f}", "-" if tc.score is None else f"{tc.score:.0f}",
                         ", ".join(ins.catalyst.types), ", ".join(ins.catalyst.sources), "; ".join(ins.warnings[:2])])
            colors.append(GREEN if ins.signal == "AL" else None)
        fill_table(self.ins_table, rows, colors)

    def add_live_insight(self, ins):
        self._insights = [x for x in self._insights if x.symbol != ins.symbol]
        self._insights.insert(0, ins)
        self._render_insights()

    def _show_insight(self, index):
        if 0 <= index.row() < len(self._insights):
            self.ins_detail.setPlainText(self._insights[index.row()].explain())

    def run_insights(self):
        cfg = load_intel_config()
        self.ins_btn.setEnabled(False)
        self.ins_detail.setPlainText("Haberler, DeFiLlama/CoinGecko ve Binance fiyatları taranıyor...")

        def work():
            from ..binance_client import BinanceAPIError, BinanceClient
            from ..intel.catalyst import InsightEngine, InsightStore
            from ..intel.news import NewsStore
            client = BinanceClient(testnet=False)
            try:
                syms = client.exchange_info().get("symbols", [])
                bases = {x["baseAsset"] for x in syms if x.get("quoteAsset") == cfg.quote_asset
                         and x.get("status") == "TRADING"}
            except BinanceAPIError:
                bases = {s[:-len(cfg.quote_asset)] for s in cfg.allowed_symbols}
            items = NewsStore().recent(cfg.catalyst.lookback_hours)
            engine = InsightEngine(cfg.catalyst, client, quote=cfg.quote_asset)
            insights = engine.scan(items, bases)
            store = InsightStore()
            for ins in insights:
                store.save(ins)
            return insights, engine.fund.status, len(items)

        def done(res):
            insights, status, n_items = res
            self.ins_btn.setEnabled(True)
            self._insights = insights
            self._render_insights()
            st = " | ".join(f"{k}: {v}" for k, v in status.items()) or "temel veri kaynağı çağrılmadı"
            if not insights:
                self.ins_detail.setPlainText(f"Son {cfg.catalyst.lookback_hours:.0f} saatteki {n_items} haberde "
                                             "Binance'te işlem gören bir coin için olumlu katalizör bulunamadı.\n"
                                             "Önce 'Haberleri ve listelemeleri tara'ya basın.\n" + st)
            else:
                self.ins_detail.setPlainText(insights[0].explain() + "\n\nKaynaklar: " + st)

        def failed(msg):
            self.ins_btn.setEnabled(True)
            self.ins_detail.setPlainText(f"Öngörü taraması başarısız: {msg}")

        self.ctx.tasks.run(work, done, failed)

    def update_performance(self):
        self.perf_btn.setEnabled(False)

        def work():
            from ..binance_client import BinanceClient
            from ..intel.catalyst import InsightStore
            store = InsightStore()
            n = store.update_outcomes(BinanceClient(testnet=False))
            return n, store.performance()

        def done(res):
            n, perf = res
            self.perf_btn.setEnabled(True)
            df_to_table(self.perf_table, perf, "ort_24s_%")
            self.ctx.status(f"{n} öngörünün gerçekleşen getirisi güncellendi.")

        def failed(msg):
            self.perf_btn.setEnabled(True)
            self.ctx.status(f"Güncelleme başarısız: {msg}")

        self.ctx.tasks.run(work, done, failed)

    # ---------------------------------------------------------------- ayarlar
    def save_settings(self):
        cfg = load_intel_config()
        nc, lc = cfg.news, cfg.listing
        nc.enabled = self.s_enabled.isChecked()
        nc.use_binance_announcements = self.s_binance.isChecked()
        nc.rss_coindesk, nc.rss_cointelegraph, nc.rss_decrypt = (self.s_cd.isChecked(), self.s_ct.isChecked(),
                                                                  self.s_dc.isChecked())
        nc.block_hours = float(self.s_block.value())
        nc.delist_exit = self.s_delist.isChecked()
        lc.enabled = self.s_listing.isChecked()
        lc.stage = self.s_stage.currentText()
        lc.wait_minutes = self.s_wait.value()
        cfg.catalyst.enabled = self.c_enabled.isChecked()
        cfg.catalyst.trade_enabled = self.c_trade.isChecked()
        cfg.catalyst.stage = self.c_stage.currentText()
        cfg.catalyst.any_binance_pair = self.c_any.isChecked()
        try:
            save_intel_config(cfg)
        except (OSError, ValueError) as exc:
            self.s_msg.setStyleSheet(f"color:{RED};")
            self.s_msg.setText(str(exc))
            return
        self.ctx.settings.cryptopanic_token = self.s_token.text().strip()
        self.ctx.persist()
        self.ctx.intel_tab.load_cfg()
        self.s_msg.setStyleSheet(f"color:{GREEN};")
        self.s_msg.setText("Kaydedildi. Çalışan bot yeniden başlatılınca uygulanır.")
