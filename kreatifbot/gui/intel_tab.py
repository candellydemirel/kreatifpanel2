"""Zeka Motoru sekmesi: canlı karar, sinyal veritabanı, strateji yönetimi ve yönetici ayarları."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..i18n import EXPLAIN_TR, tr, tr_reason
from ..intel.config import IntelConfig, config_from_dict, load_intel_config, save_intel_config
from ..intel.strategies import REGISTRY
from .widgets import GREEN, RED, combo_symbol, fill_table, make_table, signal_color, stage_combo

if TYPE_CHECKING:
    from .main_window import MainWindow


# Küçük hesap profili risk çarpanları: trend takipçileri tam, kırılım/dönüş stratejileri daha temkinli,
# çok kısa vadeli (scalp) ve türev verisine dayananlar en temkinli. Bot ayrıca bakımda öğrendiği
# çarpanla riski performansa göre düşürür (kullanılan = bu çarpan × öğrenilen çarpan).
FAMILY_MULT = {"ema": 1.0, "adx": 1.0, "supertrend": 1.0, "macd": 1.0, "structure": 1.0, "meta": 1.0,
               "breakout": 0.75, "momentum": 0.75, "volatility": 0.75, "orderflow": 0.75,
               "mean_reversion": 0.6, "statistical": 0.6, "vwap": 0.6, "derivatives": 0.5}


def small_account_multiplier(spec) -> float:
    mult = FAMILY_MULT.get(spec.family, 0.75)
    if spec.style == "scalp":
        mult = min(mult, 0.5)
    return round(max(0.1, min(1.0, mult)), 2)


def _tip(text: str, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tip:
        item.setToolTip(tip)
    return item


def mono() -> QFont:
    f = QFont("Consolas")
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def fmt_cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "-"
        if math.isinf(v):
            return "∞"
        return f"{v:,.4g}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt_cell(x) for x in v)
    return str(v)


def df_to_table(table: QTableWidget, df: pd.DataFrame | None, color_col: str | None = None):
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Veri yok"])
        return
    df = df.reset_index(drop=False) if df.index.name else df
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    rows, colors = [], []
    for _, r in df.iterrows():
        rows.append([fmt_cell(r[c]) for c in df.columns])
        if color_col and color_col in df:
            v = r[color_col]
            colors.append(GREEN if isinstance(v, (int, float)) and v > 0 else RED if isinstance(v, (int, float))
                          and v < 0 else None)
        else:
            colors.append(None)
    fill_table(table, rows, colors)


class IntelTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        self.cfg: IntelConfig = load_intel_config()
        self.last_decision = None

        top = QHBoxLayout()
        self.symbol = QComboBox()
        self.symbol.setEditable(True)
        self.symbol.addItems(self.cfg.allowed_symbols)
        self.market = QComboBox()
        self.market.addItem("Spot", "SPOT")
        self.market.addItem("USDⓈ-M Futures", "USDM_FUTURES")
        self.market.setCurrentIndex(max(0, self.market.findData(self.cfg.market)))
        self.bars = QSpinBox()
        self.bars.setRange(300, 1500)
        self.bars.setValue(600)
        self.run_btn = QPushButton("🧠 Karar Üret (canlı Binance verisi)")
        self.run_btn.clicked.connect(self.run)
        self.tg_btn = QPushButton("📨 Telegram'a gönder")
        self.tg_btn.setEnabled(False)
        self.tg_btn.clicked.connect(self.send_telegram)
        for label, w in (("Sembol", self.symbol), ("Piyasa", self.market), ("Mum", self.bars)):
            top.addWidget(QLabel(label))
            top.addWidget(w)
        top.addWidget(self.run_btn)
        top.addWidget(self.tg_btn)
        top.addStretch()
        self.tf_label = QLabel()
        self.tf_label.setStyleSheet("color:#8b949e;")

        # --- Karar görünümü
        self.explain = QPlainTextEdit()
        self.explain.setReadOnly(True)
        self.explain.setFont(mono())
        self.explain.setPlainText("Bir sembol seçip 'Karar Üret'e basın. Sistem gerektiğinde NO TRADE der; "
                                  "bu bir hata değil, risk kontrolüdür.")
        self.scores = make_table(["Grup", "LONG", "SHORT", "Ağırlık"])
        self.mtf = make_table(["Rol", "Zaman dilimi", "Eğilim"])
        self.sources = make_table(["Özellik", "Değer", "Kaynak", "Durum"])
        self.strats = make_table(["Strateji", "Aile", "Durum"])
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(mono())
        info = QTabWidget()
        info.addTab(self.scores, "Skorlar")
        info.addTab(self.mtf, "Zaman dilimleri")
        info.addTab(self.sources, "Veri kaynakları")
        info.addTab(self.strats, "Stratejiler")
        info.addTab(self.log, "Karar günlüğü")
        split = QSplitter()
        split.addWidget(self.explain)
        split.addWidget(info)
        split.setSizes([650, 650])
        decision_page = QWidget()
        dl = QVBoxLayout(decision_page)
        dl.addLayout(top)
        dl.addWidget(self.tf_label)
        dl.addWidget(split, 1)

        # --- Canlı motor kararları
        self.live_table = make_table(["Zaman", "Sembol", "Karar", "Güven", "Rejim", "Strateji", "Neden"])
        self.live_table.doubleClicked.connect(self._show_live)
        self._live: list = []

        # --- Sinyal veritabanı
        db_page = QWidget()
        dbl = QVBoxLayout(db_page)
        bar = QHBoxLayout()
        self.only_trades = QCheckBox("Yalnızca işlem sinyalleri")
        self.only_trades.setChecked(True)
        refresh = QPushButton("Yenile")
        refresh.clicked.connect(self.refresh_db)
        bar.addWidget(self.only_trades)
        bar.addWidget(refresh)
        bar.addStretch()
        self.db_table = make_table(["Oluşturma", "Sembol", "Piyasa", "Yön", "Durum", "Strateji", "Güven", "Rejim",
                                    "Giriş", "SL", "TP1", "R:R", "Sonuç", "Çıkış", "PnL", "R", "Süre dk", "ID"])
        self.db_table.doubleClicked.connect(self._show_db)
        self.db_detail = QPlainTextEdit()
        self.db_detail.setReadOnly(True)
        self.db_detail.setFont(mono())
        dsplit = QSplitter(Qt.Orientation.Vertical)
        dsplit.addWidget(self.db_table)
        dsplit.addWidget(self.db_detail)
        dbl.addLayout(bar)
        dbl.addWidget(dsplit)

        # --- Strateji yönetimi
        st_page = QWidget()
        stl = QVBoxLayout(st_page)
        self.st_table = QTableWidget(0, 9)
        self.st_table.setHorizontalHeaderLabels(["Açık", "Strateji", "Aile", "Stil", "Aşama", "Risk çarpanı",
                                                 "Öğrenilen risk", "Sağlık", "Tercih edilen rejimler (LONG)"])
        for col, key in ((4, "stage"), (5, "risk"), (6, "learned"), (8, "regimes")):
            self.st_table.horizontalHeaderItem(col).setToolTip(EXPLAIN_TR[key])
        self.st_table.verticalHeader().setVisible(False)
        st_btns = QHBoxLayout()
        save_st = QPushButton("Strateji ayarlarını kaydet")
        save_st.clicked.connect(self.save_strategies)
        st_btns.addWidget(save_st)
        st_btns.addSpacing(20)
        st_btns.addWidget(QLabel("Hepsinin aşaması:"))
        self.bulk_stage = stage_combo("PAPER")
        st_btns.addWidget(self.bulk_stage)
        bulk_btn = QPushButton("Tümüne uygula")
        bulk_btn.setToolTip("Tablodaki bütün stratejilerin aşamasını seçilen aşamaya getirir (kaydetmek için "
                            "'Strateji ayarlarını kaydet').")
        bulk_btn.clicked.connect(lambda: self.set_all_stages(self.bulk_stage.currentData()))
        st_btns.addWidget(bulk_btn)
        st_btns.addSpacing(20)
        live_btn = QPushButton("💵 Küçük hesap canlı profili (20 USDT)")
        live_btn.setToolTip("Bütün stratejiler + yeni listeleme + haber öngörüleri 'Tam canlı' olur, küçük hesap "
                            "risk ayarları uygulanır ve kaydedilir. Güvenlik kuralları (zarar limiti, devre kesici, "
                            "olasılık tahmini olmadan işlem yok) aynen kalır.")
        live_btn.clicked.connect(self.apply_small_live_profile)
        st_btns.addWidget(live_btn)
        st_btns.addStretch()
        note = QLabel("Aşamalar: Araştırma → Geçmiş test → İleri test → Kağıt işlem → Gölge → Sınırlı canlı → "
                      "Tam canlı. Canlı işlem yalnızca 'Sınırlı canlı' (yarım risk) ve 'Tam canlı' stratejilerle "
                      "yapılır. Geçmiş test iyi diye doğrudan 'Tam canlı'ya geçmeyin. Risk çarpanı 1,00 = temel "
                      "riskin tamamı; yalnızca azaltılabilir (0,1–1,0), asla artırılamaz.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b949e;")
        stl.addWidget(note)
        stl.addWidget(self.st_table)
        stl.addLayout(st_btns)

        # --- Yönetici ayarları (JSON)
        cfg_page = QWidget()
        cl = QVBoxLayout(cfg_page)
        self.cfg_edit = QPlainTextEdit()
        self.cfg_edit.setFont(mono())
        cbtn = QHBoxLayout()
        for text, fn in (("Doğrula", self.validate_cfg), ("Kaydet", self.save_cfg), ("Diskten yükle", self.load_cfg),
                         ("Varsayılanlara dön", self.reset_cfg)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            cbtn.addWidget(b)
        cbtn.addStretch()
        self.cfg_msg = QLabel()
        self.cfg_msg.setWordWrap(True)
        cl.addWidget(QLabel("Tüm kritik değerler (eşikler, ağırlıklar, ATR çarpanları, SL/TP, trailing, breakeven, "
                            "süreler, risk, maruziyet, ücret, kayma, R:R, kaldıraç, izinli sembol/zaman dilimleri, "
                            "strateji aç/kapa) buradan düzenlenir. AI bu değerleri değiştiremez."))
        cl.addWidget(self.cfg_edit)
        cl.addLayout(cbtn)
        cl.addWidget(self.cfg_msg)

        tabs = QTabWidget()
        tabs.addTab(decision_page, "Karar")
        tabs.addTab(self.live_table, "Canlı motor kararları")
        tabs.addTab(db_page, "Sinyal veritabanı")
        tabs.addTab(st_page, "Stratejiler")
        tabs.addTab(cfg_page, "Yönetici ayarları")
        QVBoxLayout(self).addWidget(tabs)
        self._load_editor()
        self._fill_strategies()
        self._update_tf_label()

    # ---------------------------------------------------------------- yapılandırma
    def _update_tf_label(self):
        tf = self.cfg.timeframes
        self.tf_label.setText(f"Zaman dilimleri — giriş {tf.entry} · onay {tf.confirmation} · trend {tf.trend} · "
                              f"ana {tf.major} · makro {tf.macro}  |  min. skor {self.cfg.scoring.min_trade_score:.0f}"
                              f"  |  işlem başı risk %{self.cfg.risk.risk_per_trade_pct}")

    def _load_editor(self):
        self.cfg_edit.setPlainText(json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2))

    def _parse_editor(self) -> IntelConfig | None:
        try:
            cfg = config_from_dict(json.loads(self.cfg_edit.toPlainText()))
        except (ValueError, TypeError) as exc:
            self.cfg_msg.setStyleSheet(f"color:{RED};")
            self.cfg_msg.setText(f"JSON hatası: {exc}")
            return None
        errors = cfg.validate()
        if errors:
            self.cfg_msg.setStyleSheet(f"color:{RED};")
            self.cfg_msg.setText("Geçersiz: " + "; ".join(errors))
            return None
        return cfg

    def validate_cfg(self):
        if self._parse_editor() is not None:
            self.cfg_msg.setStyleSheet(f"color:{GREEN};")
            self.cfg_msg.setText("Ayarlar geçerli.")

    def save_cfg(self):
        cfg = self._parse_editor()
        if cfg is None:
            return
        try:
            save_intel_config(cfg)
        except (OSError, ValueError) as exc:
            self.cfg_msg.setText(str(exc))
            return
        self.cfg = cfg
        self._fill_strategies()
        self._update_tf_label()
        self.cfg_msg.setStyleSheet(f"color:{GREEN};")
        self.cfg_msg.setText("Kaydedildi. Çalışan bot yeniden başlatıldığında uygulanır.")

    def load_cfg(self):
        self.cfg = load_intel_config()
        self._load_editor()
        self._fill_strategies()
        self._update_tf_label()

    def reset_cfg(self):
        if QMessageBox.question(self, "Varsayılanlar", "Zeka Motoru ayarları varsayılana dönsün mü?") == \
                QMessageBox.StandardButton.Yes:
            self.cfg = IntelConfig()
            self._load_editor()
            self.cfg_msg.setText("Varsayılanlar yüklendi (kaydetmek için 'Kaydet').")

    def _fill_strategies(self):
        self.st_table.setRowCount(len(REGISTRY))
        health = self.ctx.intel_health or {}
        learned = self.ctx.intel_learned or {}
        for r, (key, cls) in enumerate(REGISTRY.items()):
            sc = self.cfg.strategy(key)
            cb = QCheckBox()
            cb.setChecked(sc.enabled)
            self.st_table.setCellWidget(r, 0, cb)
            sp = cls.spec
            self.st_table.setItem(r, 1, _tip(f"{sp.name} ({key})", "\n".join(
                x for x in (f"Giriş: {sp.entry}" if sp.entry else "", f"Onay: {sp.confirmation}" if sp.confirmation
                            else "", f"Geçersizleşme: {sp.invalidation}" if sp.invalidation else "",
                            f"Çıkış: {sp.exit}" if sp.exit else "") if x)))
            self.st_table.setItem(r, 2, _tip(tr(sp.family), EXPLAIN_TR.get(sp.family, "")))
            self.st_table.setItem(r, 3, _tip(tr(sp.style), EXPLAIN_TR.get(sp.style, "")))
            stage = stage_combo(sc.stage)
            stage.setToolTip(EXPLAIN_TR["stage"])
            self.st_table.setCellWidget(r, 4, stage)
            rm = QDoubleSpinBox()
            rm.setRange(0.1, 1.0)
            rm.setSingleStep(0.1)
            rm.setValue(min(1.0, sc.risk_multiplier))
            rm.setToolTip(EXPLAIN_TR["risk"])
            self.st_table.setCellWidget(r, 5, rm)
            if key in learned:
                lm, why = learned[key]
                item = _tip(f"{lm:.2f}", why)
                item.setForeground(QColor(GREEN if lm >= 1.0 else RED if lm <= 0.25 else "#d29922"))
            else:
                item = _tip("Henüz yok", "Otomatik pilot açıkken ilk bakım bitince bot bu değeri kendisi öğrenir.")
            self.st_table.setItem(r, 6, item)
            h = str(health.get(key, "ACTIVE"))
            self.st_table.setItem(r, 7, _tip(tr(h), EXPLAIN_TR.get(h, "")))
            self.st_table.setItem(r, 8, _tip(", ".join(sorted(tr(x.value) for x in sp.preferred)),
                                             EXPLAIN_TR["regimes"]))
            self.st_table.item(r, 1).setData(Qt.ItemDataRole.UserRole, key)
        self.st_table.resizeColumnsToContents()

    def set_all_stages(self, stage: str):
        for r in range(self.st_table.rowCount()):
            box = self.st_table.cellWidget(r, 4)
            idx = box.findData(stage)
            if idx >= 0:
                box.setCurrentIndex(idx)

    def apply_small_live_profile(self, confirm: bool = True):
        """20 USDT gibi küçük bir hesap için tüm stratejileri gerçek işleme hazırlar."""
        if confirm and QMessageBox.question(
                self, "Gerçek işlem profili",
                "Bütün stratejiler, yeni listeleme ve haber öngörüleri 'Tam canlı' aşamasına alınacak.\n\n"
                "Bot 'Canlı işlem' modunda başlatılırsa Binance Spot cüzdanınızdaki USDT ile GERÇEK emir "
                "gönderir. Kâr garantisi yoktur; kaybetmeyi göze alabileceğiniz tutarla çalışın.\n\n"
                "Risk çarpanları strateji türüne göre ayarlanır (trend 1.0, kırılım 0.75, ortalamaya dönüş 0.6, "
                "kısa vadeli/türev 0.5).\n\nKorunan güvenlik kuralları: işlem başı risk %0.5 (en küçük emir için en fazla %1.5), günlük "
                "zarar limiti %3, en fazla 3 açık pozisyon, art arda 4 zararda durma, olasılık tahmini olmadan "
                "canlı işlem yok.\n\nDevam edilsin mi?") != QMessageBox.StandardButton.Yes:
            return
        cfg = self.cfg
        for key, cls in REGISTRY.items():
            sc = cfg.strategy(key)
            sc.stage = "FULL_LIVE"
            sc.risk_multiplier = small_account_multiplier(cls.spec)
        cfg.listing.stage = "FULL_LIVE"
        cfg.catalyst.stage = "FULL_LIVE"
        r = cfg.risk
        r.risk_per_trade_pct = 0.5
        r.small_account_max_risk_pct = 1.5
        r.max_open_positions = 3
        r.max_daily_loss_pct = 3.0
        r.max_consecutive_losses = 4
        try:
            save_intel_config(cfg)
        except (OSError, ValueError) as exc:
            self.ctx.show_error("Kaydedilemedi", str(exc))
            return
        self._fill_strategies()
        self._load_editor()
        news = getattr(self.ctx, "news_tab", None)
        for box in (getattr(news, "s_stage", None), getattr(news, "c_stage", None)):
            if box is not None and box.findData("FULL_LIVE") >= 0:
                box.setCurrentIndex(box.findData("FULL_LIVE"))
        self.ctx.status("Küçük hesap canlı profili kaydedildi. Bot sekmesinde 'Canlı işlem' seçip botu yeniden "
                        "başlatın.")

    def save_strategies(self):
        for r in range(self.st_table.rowCount()):
            key = self.st_table.item(r, 1).data(Qt.ItemDataRole.UserRole)
            sc = self.cfg.strategy(key)
            sc.enabled = self.st_table.cellWidget(r, 0).isChecked()
            sc.stage = self.st_table.cellWidget(r, 4).currentData()
            sc.risk_multiplier = self.st_table.cellWidget(r, 5).value()
        try:
            save_intel_config(self.cfg)
        except (OSError, ValueError) as exc:
            self.ctx.show_error("Kaydedilemedi", str(exc))
            return
        self._load_editor()
        self.ctx.status("Strateji ayarları kaydedildi.")

    # ---------------------------------------------------------------- karar
    @Slot()
    def run(self):
        from ..intel.data_quality import check_candles, check_orderbook
        from ..intel.decision import DecisionEngine
        from ..intel.futures_client import BinanceFuturesClient
        from ..intel.market_data import live_snapshot, load_history
        from ..intel.orderflow import orderbook_features
        from ..binance_client import BinanceClient

        cfg = load_intel_config()
        cfg.market = self.market.currentData()
        self.cfg = cfg
        self._update_tf_label()
        symbol = combo_symbol(self.symbol)
        bars = self.bars.value()
        meta, stats, health = self.ctx.intel_meta, self.ctx.intel_stats, self.ctx.intel_health
        self.run_btn.setEnabled(False)
        self.ctx.status(f"{symbol}: Binance verisi alınıyor ve karar üretiliyor...")

        def work():
            spot, fut = BinanceClient(testnet=False), BinanceFuturesClient(testnet=False)
            bundle = load_history(cfg, symbol, bars, spot, fut)
            if not bundle.ok:
                return {"error": "DATA_UNAVAILABLE: " + "; ".join(bundle.errors)}
            client = spot if cfg.market == "SPOT" else fut
            try:
                rules = client.symbol_rules(symbol)
            except Exception as exc:  # noqa: BLE001
                rules = None
                bundle.errors.append(f"sembol filtreleri: {exc}")
            fs, raw = live_snapshot(symbol, spot, fut, want_spot=True,
                                    want_futures=cfg.market != "SPOT" or cfg.use_futures_context)
            book = raw["futures_book"] if cfg.market != "SPOT" else raw["spot_book"]
            dq = check_candles(bundle.entry, cfg.timeframes.entry, cfg.data_quality,
                               min_history=min(cfg.data_quality.min_history_bars, bars - 10))
            dq.merge(check_orderbook(book, cfg.data_quality), "orderbook: ")
            de = DecisionEngine(cfg, meta_model=meta, strategy_stats=stats, health=health,
                                learned=self.ctx.intel_learned)
            btc_f = None
            if bundle.btc_trend is not None and not bundle.btc_trend.empty:
                from ..intel.features import compute_features
                btc_f = compute_features(bundle.btc_trend, cfg.timeframes.trend, cfg.market)
            prep = de.prepare(bundle.entry, symbol, cfg.timeframes.entry, bundle.htf, bundle.derivatives, btc_f)
            funding_now = fs.get("funding_rate")
            d = de.decide(prep, -1, dq=dq, rules=rules, book_stats=orderbook_features(book) if book else None,
                          funding_now=funding_now)
            route_info = {}
            from ..intel.router import route
            from ..intel.types import Regime
            rt = route(Regime(d.market_regime), de.strategies, cfg, cfg.market, health)
            for st in de.strategies:
                c = prep.candidates[st.key]
                if st.key in prep.missing:
                    status = "Veri yok: " + ", ".join(prep.missing[st.key])
                elif c["long"][-1] or c["short"][-1]:
                    status = "ADAY " + ("LONG" if c["long"][-1] else "SHORT")
                elif st.key in rt.inactive:
                    status = "Pasif: " + rt.inactive[st.key]
                else:
                    status = "Aktif (sinyal yok)"
                route_info[st.key] = (st.spec.name, tr(st.spec.family), status)
            return {"decision": d, "fs": fs, "errors": bundle.errors, "route": route_info,
                    "gs": prep.gs.iloc[-1], "avail": prep.f.attrs.get("availability", {})}

        def done(res):
            self.run_btn.setEnabled(True)
            if "error" in res:
                self.explain.setPlainText(res["error"])
                self.ctx.status(res["error"])
                return
            d = res["decision"]
            self.last_decision = d
            self.ctx.intel_last_decision = d
            self.tg_btn.setEnabled(True)
            extra = ("\n\nVeri uyarıları:\n  " + "\n  ".join(res["errors"])) if res["errors"] else ""
            self.explain.setPlainText(d.explanation + extra)
            gs = res["gs"]
            rows = []
            for g, w in self.cfg.scoring.weights.items():
                lv, sv = gs.get(f"{g}_long"), gs.get(f"{g}_short")
                rows.append([tr(g), "Veri yok" if pd.isna(lv) else f"{lv * w:.1f}",
                             "Veri yok" if pd.isna(sv) else f"{sv * w:.1f}", f"{w:.0f}"])
            rows.append(["TOPLAM", fmt_cell(d.long_score), fmt_cell(d.short_score), "100"])
            fill_table(self.scores, rows)
            tf = self.cfg.timeframes
            fill_table(self.mtf, [[role, getattr(tf, role) if role != "score" else "", str(v)]
                                  for role, v in d.mtf.items()])
            src_rows = res["fs"].to_rows() + [[k, "", "özellik motoru", v] for k, v in res["avail"].items()]
            fill_table(self.sources, [[fmt_cell(x) for x in r] for r in src_rows],
                       [RED if str(r[3]) != "OK" else None for r in src_rows])
            st_rows = [[f"{n} ({k})", fam, stt] for k, (n, fam, stt) in res["route"].items()]
            fill_table(self.strats, st_rows, [GREEN if r[2].startswith("ADAY") else None for r in st_rows])
            self.log.setPlainText("\n".join(d.log_lines))
            label = tr(d.direction) if d.is_trade else "İşlem yok"
            self.ctx.status(f"{symbol}: {label} (güven {d.confidence:.0f}, rejim {tr(d.market_regime)})")

        def failed(msg):
            self.run_btn.setEnabled(True)
            self.explain.setPlainText(f"DATA_UNAVAILABLE / hata: {msg}")
            self.ctx.show_error("Karar üretilemedi", msg)

        self.ctx.tasks.run(work, done, failed)

    def send_telegram(self):
        if self.last_decision is None:
            return
        from ..telegram import esc
        d = self.last_decision
        text = f"🧠 <b>Zeka Motoru — {esc(d.symbol)}</b>\n<pre>{esc(d.explanation[:3500])}</pre>"
        self.ctx.telegram_send(text)

    # ---------------------------------------------------------------- canlı motor
    def on_live_decision(self, d):
        self._live.insert(0, d)
        self._live = self._live[:300]
        rows, colors = [], []
        for x in self._live:
            label = tr(x.direction) if x.is_trade else "İşlem yok"
            rows.append([x.created_at[:19].replace("T", " "), x.symbol, label, f"{x.confidence:.0f}",
                         tr(x.market_regime), x.strategy, ", ".join(tr(r) for r in x.no_trade_reasons)])
            colors.append(signal_color("AL" if x.direction == "LONG" and x.is_trade else
                                       "SAT" if x.direction == "SHORT" and x.is_trade else ""))
        fill_table(self.live_table, rows, colors)

    def _show_live(self, index):
        if 0 <= index.row() < len(self._live):
            d = self._live[index.row()]
            QMessageBox.information(self, f"{d.symbol} karar", d.explanation)

    # ---------------------------------------------------------------- veritabanı
    def refresh_db(self):
        from ..intel.signal_store import DatabaseError, SignalStore
        try:
            store = SignalStore()
            df = store.recent(500, only_trades=self.only_trades.isChecked())
        except DatabaseError as exc:
            self.ctx.show_error("Veritabanı", str(exc))
            return
        self._db = df
        rows, colors = [], []
        for _, r in df.iterrows():
            rows.append([str(r["created_at"])[:19], r["symbol"], tr(r["market_type"]), tr(r["direction"]), tr(r["status"]),
                         r["strategy"], fmt_cell(r["confidence"]), tr(r["market_regime"]), fmt_cell(r["entry"]),
                         fmt_cell(r["sl"]), fmt_cell(r["tp1"]), fmt_cell(r["risk_reward"]), r["outcome"] or "",
                         tr_reason(r["exit_reason"] or ""), fmt_cell(r["pnl"]), fmt_cell(r["r_multiple"]),
                         fmt_cell(r["holding_time_min"]), r["signal_id"]])
            colors.append(GREEN if (r["pnl"] or 0) > 0 else RED if (r["pnl"] or 0) < 0 else None)
        fill_table(self.db_table, rows, colors)

    def _show_db(self, index):
        from ..intel.signal_store import SignalStore
        item = self.db_table.item(index.row(), 17)
        if item is None:
            return
        sid = item.text()
        row = self._db[self._db["signal_id"] == sid]
        expl = row["explanation"].iat[0] if not row.empty else ""
        log = "\n".join(SignalStore().log_for(sid))
        self.db_detail.setPlainText(f"{expl}\n\n--- Karar günlüğü ---\n{log}")
