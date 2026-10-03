"""Uygulama sekmeleri: Piyasa Analizi, Tarayıcı, Backtest, Bot ve Ayarlar."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
    QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..analyzer import analyze
from ..backtest import compare_strategies, run_backtest
from ..binance_client import BinanceClient
from ..broker import LiveBroker, PaperBroker
from ..config import data_dir
from ..engine import BotEngine
from ..strategies import STRATEGIES, SmartEnsemble, create_strategy
from ..telegram import esc
from ..utils import fmt_money, fmt_pct, fmt_price, fmt_qty
from .widgets import (
    GREEN, RED, EquityChart, PriceChart, RiskForm, StrategyPicker, combo_symbol, fill_table,
    interval_combo, make_table, pnl_color, signal_color, symbol_combo,
)

if TYPE_CHECKING:
    from .main_window import MainWindow

STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "GBP", "AEUR", "USDE", "PAXG", "XUSD", "USD1"}


def _scroll(widget: QWidget, width: int = 360) -> QScrollArea:
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setMinimumWidth(width)
    area.setMaximumWidth(width + 80)
    return area


def _mono() -> QFont:
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


# ======================================================================== Analiz
class AnalysisTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        top = QHBoxLayout()
        self.symbol = symbol_combo()
        self.interval = interval_combo("1h")
        self.candles = QSpinBox()
        self.candles.setRange(100, 1000)
        self.candles.setValue(500)
        self.button = QPushButton("Analiz Et")
        self.button.setDefault(True)
        self.button.clicked.connect(self.run)
        self.tg_button = QPushButton("📨 Telegram'a gönder")
        self.tg_button.setEnabled(False)
        self.tg_button.clicked.connect(self.send_telegram)
        self.last_analysis = None
        for label, w in (("Sembol", self.symbol), ("Aralık", self.interval), ("Mum", self.candles)):
            top.addWidget(QLabel(label))
            top.addWidget(w)
        top.addWidget(self.button)
        top.addWidget(self.tg_button)
        top.addStretch()

        self.chart = PriceChart()
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(_mono())
        self.signals = make_table(["Strateji", "Sinyal", "Durum"])
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.summary)
        sig_box = QGroupBox("Stratejilerin son mum sinyalleri")
        QVBoxLayout(sig_box).addWidget(self.signals)
        right.addWidget(sig_box)
        right.setSizes([500, 250])

        split = QSplitter()
        split.addWidget(self.chart)
        split.addWidget(right)
        split.setSizes([900, 420])

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split)
        self.summary.setPlainText("Bir sembol seçip 'Analiz Et' butonuna basın.")

    def open_symbol(self, symbol: str, interval: str | None = None):
        self.symbol.setCurrentText(symbol)
        if interval:
            self.interval.setCurrentText(interval)
        self.run()

    @Slot()
    def run(self):
        symbol, interval, n = combo_symbol(self.symbol), self.interval.currentText(), self.candles.value()
        client = self.ctx.data_client()
        self.button.setEnabled(False)
        self.ctx.status(f"{symbol} analiz ediliyor...")

        def work():
            df = client.klines(symbol, interval, limit=n)
            analysis = analyze(df, symbol, interval)
            evals = []
            for key in STRATEGIES:
                strat = create_strategy(key)
                evals.append((strat.name, strat.evaluate(df)))
            ens = SmartEnsemble()
            sig = ens.signals(df)
            return df, analysis, evals, list(sig[sig == 1].index), list(sig[sig == -1].index)

        def done(result):
            df, analysis, evals, buys, sells = result
            self.button.setEnabled(True)
            self.chart.set_data(df, buys, sells, f"{symbol} · {interval}  (işaretler: Akıllı Kombine)")
            self.summary.setPlainText(analysis.summary_text())
            self.last_analysis = (analysis, evals)
            self.tg_button.setEnabled(True)
            rows = [[name, res.text, res.reason] for name, res in evals]
            fill_table(self.signals, rows, [signal_color(r[1]) for r in rows])
            self.ctx.status(f"{symbol}: skor {analysis.score:+.0f} → {analysis.recommendation}")

        def failed(msg):
            self.button.setEnabled(True)
            self.ctx.show_error("Analiz başarısız", msg)

        self.ctx.tasks.run(work, done, failed)


    def send_telegram(self):
        if not self.last_analysis:
            return
        a, evals = self.last_analysis
        sigs = "\n".join(f"• {esc(name)}: <b>{res.text}</b>" for name, res in evals)
        text = (f"🔍 <b>{esc(a.symbol)} ({esc(a.interval)}) analizi</b>\n"
                f"Fiyat: {fmt_price(a.price)} ({fmt_pct(a.change_pct)})\n"
                f"Skor: <b>{a.score:+.0f}</b> → <b>{a.recommendation}</b>\n"
                f"Trend: {esc(a.trend)} · {esc(a.regime)} (ADX {a.adx:.0f})\n"
                f"RSI {a.rsi:.0f} · Volatilite {a.volatility_pct:.2f}%\n"
                f"Destek {fmt_price(a.support)} · Direnç {fmt_price(a.resistance)}\n\n"
                f"<b>Strateji sinyalleri</b>\n{sigs}\n\n<i>Yatırım tavsiyesi değildir.</i>")
        self.ctx.telegram_send(text)


# ======================================================================== Tarayıcı
class ScannerTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        top = QHBoxLayout()
        self.top_mode = QRadioButton("En hacimli")
        self.top_mode.setChecked(True)
        self.top_n = QSpinBox()
        self.top_n.setRange(5, 100)
        self.top_n.setValue(25)
        self.custom_mode = QRadioButton("Özel liste")
        self.custom = QLineEdit("BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT")
        self.interval = interval_combo("1h")
        self.button = QPushButton("Piyasayı Tara")
        self.button.clicked.connect(self.run)
        self.tg_button = QPushButton("📨 Telegram'a gönder")
        self.tg_button.setEnabled(False)
        self.tg_button.clicked.connect(self.send_telegram)
        self.last_rows: list = []
        for w in (self.top_mode, self.top_n, QLabel("parite"), self.custom_mode, self.custom,
                  QLabel("Aralık"), self.interval, self.button, self.tg_button):
            top.addWidget(w)
        self.table = make_table(["Sembol", "Fiyat", "24s Değişim", "24s Hacim", "Trend", "Rejim",
                                 "RSI", "Volatilite", "Skor", "Öneri"])
        self.table.doubleClicked.connect(self._open)
        self.info = QLabel("Satıra çift tıklayarak sembolü Piyasa Analizi sekmesinde açabilirsiniz.")
        self.info.setStyleSheet("color:#8b949e;")
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table)
        layout.addWidget(self.info)

    def send_telegram(self):
        if not self.last_rows:
            return
        rows = self.last_rows

        def line(r):
            sym, a, _ = r
            icon = "🟢" if a.score >= 20 else ("🔴" if a.score <= -20 else "⚪")
            return f"{icon} <b>{esc(sym)}</b> {a.score:+.0f} {a.recommendation} · RSI {a.rsi:.0f} · {esc(a.trend)}"

        top = [line(r) for r in rows[:10]]
        bottom = [line(r) for r in rows[-5:]] if len(rows) > 10 else []
        text = (f"🔎 <b>Piyasa taraması ({esc(self.interval.currentText())}, {len(rows)} parite)</b>\n\n"
                "<b>En güçlü</b>\n" + "\n".join(top))
        if bottom:
            text += "\n\n<b>En zayıf</b>\n" + "\n".join(bottom)
        text += "\n\n<i>Yatırım tavsiyesi değildir.</i>"
        self.ctx.telegram_send(text)

    def _open(self, index):
        item = self.table.item(index.row(), 0)
        if item:
            self.ctx.open_analysis(item.text(), self.interval.currentText())

    @Slot()
    def run(self):
        client = self.ctx.data_client()
        quote = self.ctx.settings.quote_asset
        interval = self.interval.currentText()
        use_top, top_n = self.top_mode.isChecked(), self.top_n.value()
        custom = [s.strip().upper() for s in self.custom.text().replace(";", ",").split(",") if s.strip()]
        self.button.setEnabled(False)
        self.ctx.status("Piyasa taranıyor...")

        def work():
            tickers = {t["symbol"]: t for t in client.ticker_24h()}
            if use_top:
                candidates = []
                for sym, t in tickers.items():
                    if not sym.endswith(quote):
                        continue
                    base = sym[: -len(quote)]
                    if base in STABLE_BASES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                        continue
                    if float(t.get("quoteVolume", 0)) <= 0:
                        continue
                    candidates.append(sym)
                candidates.sort(key=lambda s: float(tickers[s]["quoteVolume"]), reverse=True)
                symbols = candidates[:top_n]
            else:
                symbols = custom
            rows, errors = [], []
            for sym in symbols:
                try:
                    df = client.klines(sym, interval, limit=250)
                    a = analyze(df, sym, interval)
                    rows.append((sym, a, tickers.get(sym, {})))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{sym}: {exc}")
            return rows, errors

        def done(result):
            rows, errors = result
            self.button.setEnabled(True)
            rows.sort(key=lambda r: r[1].score, reverse=True)
            self.last_rows = rows
            self.tg_button.setEnabled(bool(rows))
            table_rows, colors = [], []
            for sym, a, t in rows:
                change = float(t.get("priceChangePercent", "nan") or "nan")
                volume = float(t.get("quoteVolume", "nan") or "nan")
                table_rows.append([
                    sym, (fmt_price(a.price), a.price), (fmt_pct(change), change),
                    (fmt_money(volume), volume), a.trend, a.regime, (f"{a.rsi:.1f}", a.rsi),
                    (f"{a.volatility_pct:.2f}%", a.volatility_pct), (f"{a.score:+.0f}", a.score),
                    a.recommendation,
                ])
                colors.append(signal_color(a.recommendation))
            fill_table(self.table, table_rows, colors)
            self.table.setSortingEnabled(True)
            msg = f"{len(rows)} parite tarandı."
            if errors:
                msg += f" {len(errors)} hata: " + "; ".join(errors[:3])
            self.info.setText(msg + "  Çift tıklayarak detaylı analiz açın.")
            self.ctx.status(msg)

        def failed(msg):
            self.button.setEnabled(True)
            self.ctx.show_error("Tarama başarısız", msg)

        self.ctx.tasks.run(work, done, failed)


# ======================================================================== Backtest
class BacktestTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        self._cache: dict = {}

        panel = QWidget()
        form_box = QGroupBox("Test verisi")
        form = QFormLayout(form_box)
        self.symbol = symbol_combo()
        self.interval = interval_combo("1h")
        self.candles = QSpinBox()
        self.candles.setRange(200, 20000)
        self.candles.setSingleStep(500)
        self.candles.setValue(2000)
        self.capital = QDoubleSpinBox()
        self.capital.setRange(10, 10_000_000)
        self.capital.setValue(1000)
        self.capital.setSuffix(" USDT")
        form.addRow("Sembol", self.symbol)
        form.addRow("Aralık", self.interval)
        form.addRow("Mum sayısı", self.candles)
        form.addRow("Başlangıç sermayesi", self.capital)
        self.strategy = StrategyPicker()
        self.risk = RiskForm()
        self.run_btn = QPushButton("Backtest Çalıştır")
        self.run_btn.clicked.connect(self.run)
        self.cmp_btn = QPushButton("Tüm Stratejileri Karşılaştır")
        self.cmp_btn.clicked.connect(self.compare)
        pl = QVBoxLayout(panel)
        for w in (form_box, self.strategy, self.risk, self.run_btn, self.cmp_btn):
            pl.addWidget(w)
        pl.addStretch()

        self.metrics = make_table(["Metrik", "Değer"])
        self.chart = PriceChart()
        self.equity = EquityChart()
        self.trades = make_table(["Açılış", "Kapanış", "Giriş", "Çıkış", "Miktar", "K/Z", "K/Z %", "Neden"])
        self.compare_table = make_table(["Strateji", "Getiri", "Al-tut", "İşlem", "Kazanma", "Kâr faktörü",
                                         "Maks. düşüş", "Sharpe"])
        self.compare_table.doubleClicked.connect(self._pick_compared)
        self.results = QTabWidget()
        self.results.addTab(self.chart, "Grafik")
        self.results.addTab(self.equity, "Sermaye Eğrisi")
        self.results.addTab(self.trades, "İşlemler")
        self.results.addTab(self.compare_table, "Karşılaştırma")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.metrics)
        right.addWidget(self.results)
        right.setSizes([230, 600])
        split = QSplitter()
        split.addWidget(_scroll(panel))
        split.addWidget(right)
        split.setSizes([380, 1000])
        QVBoxLayout(self).addWidget(split)

    def _load(self, client, symbol, interval, n):
        key = (client.base_url, symbol, interval, n)
        if key not in self._cache:
            df = client.klines_history(symbol, interval, n)
            if len(df) < 100:
                raise ValueError(f"{symbol} için yeterli veri bulunamadı ({len(df)} mum).")
            self._cache = {key: df}
        return self._cache[key]

    def _busy(self, busy: bool):
        self.run_btn.setEnabled(not busy)
        self.cmp_btn.setEnabled(not busy)

    @Slot()
    def run(self):
        symbol, interval, n = combo_symbol(self.symbol), self.interval.currentText(), self.candles.value()
        strategy, risk, capital = self.strategy.create(), self.risk.settings(), self.capital.value()
        client = self.ctx.data_client()
        self._busy(True)
        self.ctx.status(f"{symbol} için backtest çalışıyor...")

        def work():
            df = self._load(client, symbol, interval, n)
            return df, run_backtest(df, strategy, risk, capital, interval, symbol)

        def done(result):
            df, res = result
            self._busy(False)
            self._show_result(df, res, symbol, interval)
            self.ctx.status(f"Backtest tamamlandı: {fmt_pct(res.metrics['Toplam getiri (%)'])}")

        self.ctx.tasks.run(work, done, self._failed)

    @Slot()
    def compare(self):
        symbol, interval, n = combo_symbol(self.symbol), self.interval.currentText(), self.candles.value()
        risk, capital = self.risk.settings(), self.capital.value()
        client = self.ctx.data_client()
        self._busy(True)
        self.ctx.status("Stratejiler karşılaştırılıyor...")

        def work():
            df = self._load(client, symbol, interval, n)
            return df, compare_strategies(df, [create_strategy(k) for k in STRATEGIES], risk, capital,
                                          interval, symbol)

        def done(result):
            df, results = result
            self._busy(False)
            rows, colors = [], []
            for r in results:
                m = r.metrics
                pf = m["Kâr faktörü"]
                rows.append([
                    r.strategy_name, (fmt_pct(m["Toplam getiri (%)"]), m["Toplam getiri (%)"]),
                    (fmt_pct(m["Al-tut getirisi (%)"]), m["Al-tut getirisi (%)"]),
                    (str(m["İşlem sayısı"]), m["İşlem sayısı"]),
                    (fmt_pct(m["Kazanma oranı (%)"], False), m["Kazanma oranı (%)"]),
                    ("∞" if math.isinf(pf) else f"{pf:.2f}", pf if not math.isinf(pf) else 1e9),
                    (fmt_pct(m["Maks. düşüş (%)"]), m["Maks. düşüş (%)"]),
                    (f"{m['Sharpe oranı']:.2f}", m["Sharpe oranı"]),
                ])
                colors.append(pnl_color(m["Toplam getiri (%)"]))
            fill_table(self.compare_table, rows, colors)
            self.compare_table.setSortingEnabled(True)
            self.results.setCurrentWidget(self.compare_table)
            if results:
                self._show_result(df, results[0], symbol, interval, switch=False)
                self.ctx.status(f"En iyi strateji: {results[0].strategy_name} "
                                f"({fmt_pct(results[0].metrics['Toplam getiri (%)'])}). "
                                "Detay için satıra çift tıklayın.")

        self.ctx.tasks.run(work, done, self._failed)

    def _pick_compared(self, index):
        name = self.compare_table.item(index.row(), 0).text()
        for key, cls in STRATEGIES.items():
            if cls.name == name:
                self.strategy.set_strategy(key)
                self.run()
                return

    def _failed(self, msg):
        self._busy(False)
        self.ctx.show_error("Backtest başarısız", msg)

    def _show_result(self, df, res, symbol, interval, switch=True):
        rows = []
        for k, v in res.metrics.items():
            if k == "İşlem sayısı":
                text = str(v)
            elif k == "Son bakiye":
                text = fmt_money(v, "USDT")
            elif k in ("Kâr faktörü", "Sharpe oranı"):
                text = "∞" if math.isinf(v) else f"{v:.2f}"
            else:
                text = fmt_pct(v, signed=k not in ("Kazanma oranı (%)", "Piyasada kalma (%)"))
            rows.append([k, text])
        rows.insert(0, ["Strateji", res.strategy_name])
        rows.insert(1, ["Dönem", f"{df['open_time'].iloc[0]:%d.%m.%Y} – {df['open_time'].iloc[-1]:%d.%m.%Y}"
                                 f" ({len(df)} mum)"])
        fill_table(self.metrics, rows)
        self.chart.set_data(df, res.buy_points, res.sell_points, f"{symbol} · {interval} · {res.strategy_name}")
        bench = df["close"] / df["close"].iloc[0] * res.initial_capital
        bench.index = res.equity.index
        self.equity.set_data(res.equity, bench)
        trows = [[t.opened_at, t.closed_at, fmt_price(t.entry_price), fmt_price(t.exit_price), fmt_qty(t.qty),
                  (f"{t.pnl:+.2f}", t.pnl), (fmt_pct(t.pnl_pct), t.pnl_pct), t.reason] for t in res.trades]
        fill_table(self.trades, trows, [pnl_color(t.pnl) for t in res.trades])
        if switch:
            self.results.setCurrentWidget(self.equity)


# ======================================================================== Bot
class BotTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        self.engine: BotEngine | None = None
        self.notifier = None
        s = ctx.settings

        panel = QWidget()
        mode_box = QGroupBox("Çalışma modu")
        ml = QVBoxLayout(mode_box)
        self.paper = QRadioButton("Kağıt işlem (simülasyon, gerçek emir yok)")
        self.live = QRadioButton("Canlı işlem (Binance'e gerçek emir gönderir)")
        group = QButtonGroup(self)
        group.addButton(self.paper)
        group.addButton(self.live)
        (self.live if s.live_mode else self.paper).setChecked(True)
        self.paper_balance = QDoubleSpinBox()
        self.paper_balance.setRange(10, 10_000_000)
        self.paper_balance.setValue(s.paper_balance)
        self.paper_balance.setSuffix(f" {s.quote_asset}")
        self.reset_paper = QPushButton("Kağıt hesabını sıfırla")
        self.reset_paper.clicked.connect(self._reset_paper)
        self.net_label = QLabel()
        self.net_label.setWordWrap(True)
        ml.addWidget(self.paper)
        pb = QHBoxLayout()
        pb.addWidget(QLabel("Başlangıç bakiyesi"))
        pb.addWidget(self.paper_balance)
        ml.addLayout(pb)
        ml.addWidget(self.reset_paper)
        ml.addWidget(self.live)
        ml.addWidget(self.net_label)

        eng_box = QGroupBox("Motor")
        ef = QFormLayout(eng_box)
        self.engine_type = QComboBox()
        self.engine_type.addItem("Klasik strateji (tek strateji, spot)", "classic")
        self.engine_type.addItem("Zeka Motoru (çoklu strateji + rejim + risk)", "intel")
        self.engine_type.setCurrentIndex(max(0, self.engine_type.findData(s.engine_type)))
        self.intel_market = QComboBox()
        self.intel_market.addItem("Binance Spot", "SPOT")
        self.intel_market.addItem("Binance USDⓈ-M Futures", "USDM_FUTURES")
        self.intel_market.setCurrentIndex(max(0, self.intel_market.findData(s.intel_market)))
        self.leverage = QSpinBox()
        self.leverage.setRange(1, 125)
        self.leverage.setValue(s.intel_leverage)
        self.leverage.setSuffix("x")
        self.intel_note = QLabel("Zeka Motoru; zaman dilimleri, eşikler ve risk ayarlarını 'Zeka Motoru' sekmesinden "
                                 "alır. Strateji aşaması LIMITED_LIVE/FULL_LIVE değilse canlıda işlem açmaz.")
        self.intel_note.setWordWrap(True)
        self.intel_note.setStyleSheet("color:#8b949e;")
        ef.addRow("Motor", self.engine_type)
        ef.addRow("Piyasa", self.intel_market)
        ef.addRow("Kaldıraç (futures)", self.leverage)
        ef.addRow(self.intel_note)
        self.engine_type.currentIndexChanged.connect(lambda _: self._engine_changed())
        self.intel_market.currentIndexChanged.connect(lambda _: self._engine_changed())

        mk_box = QGroupBox("Piyasa")
        mf = QFormLayout(mk_box)
        self.symbols = QLineEdit(", ".join(s.symbols))
        self.symbols.setPlaceholderText("BTCUSDT, ETHUSDT")
        self.interval = interval_combo(s.interval)
        self.poll = QSpinBox()
        self.poll.setRange(5, 3600)
        self.poll.setValue(s.poll_seconds)
        self.poll.setSuffix(" sn")
        mf.addRow("Semboller", self.symbols)
        self.auto_universe = QCheckBox("Coinleri bot seçsin (hacimli + en çok hareket eden USDT çiftleri, saatlik)")
        self.auto_universe.setChecked(s.auto_universe)
        self.auto_universe.setToolTip("Zeka Motoru, Binance'teki USDT çiftlerinden yeterli hacmi olanları seçer: "
                                      "yarısı en yüksek hacimli, yarısı son 24 saatte en çok hareket eden coinler. "
                                      "Stablecoin ve kaldıraçlı tokenlar elenir. Yukarıdaki semboller her zaman "
                                      "listede kalır; açık pozisyonu olan coin listeden çıkarılmaz.")
        self.universe_size = QSpinBox()
        self.universe_size.setRange(2, 40)
        self.universe_size.setValue(s.universe_size)
        self.universe_size.setSuffix(" coin")
        mf.addRow("", self.auto_universe)
        mf.addRow("Takip edilecek coin sayısı", self.universe_size)
        mf.addRow("Mum aralığı", self.interval)
        mf.addRow("Kontrol sıklığı", self.poll)

        self.strategy = StrategyPicker()
        self.strategy.set_strategy(s.strategy, s.strategy_params)
        self.risk = RiskForm()
        self.risk.set_settings(s.risk_settings)

        self.autopilot_btn = QPushButton()
        self.autopilot_btn.clicked.connect(self.toggle_autopilot)
        self.start_btn = QPushButton("▶  Botu Başlat")
        self.start_btn.setStyleSheet(f"QPushButton {{background:{GREEN}; color:white; font-weight:bold; padding:8px;}}"
                                     "QPushButton:disabled {background:#30363d; color:#6e7681;}")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Botu Durdur")
        self.stop_btn.setStyleSheet(f"QPushButton {{background:{RED}; color:white; font-weight:bold; padding:8px;}}"
                                    "QPushButton:disabled {background:#30363d; color:#6e7681;}")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)

        pl = QVBoxLayout(panel)
        for w in (mode_box, eng_box, mk_box, self.strategy, self.risk):
            pl.addWidget(w)
        pl.addStretch()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(_scroll(panel, 380))
        ll.addWidget(self.autopilot_btn)
        ll.addWidget(self.start_btn)
        ll.addWidget(self.stop_btn)

        # --- sağ taraf
        cards = QGridLayout()
        self.cards = {}
        for i, name in enumerate(("Durum", "Serbest bakiye", "Toplam değer", "Günlük K/Z", "Gerçekleşen K/Z")):
            box = QGroupBox(name)
            lbl = QLabel("-")
            lbl.setStyleSheet("font-size:16px; font-weight:bold;")
            QVBoxLayout(box).addWidget(lbl)
            self.cards[name] = lbl
            cards.addWidget(box, 0, i)
        self.cards["Durum"].setText("Durduruldu")

        self.positions = make_table(["Sembol", "Miktar", "Giriş", "Güncel", "Değer", "K/Z", "K/Z %",
                                     "Stop-loss", "Kâr al", "Açılış"])
        pos_buttons = QHBoxLayout()
        self.close_btn = QPushButton("Seçili pozisyonu sat")
        self.close_btn.clicked.connect(self._close_selected)
        self.forget_btn = QPushButton("Takipten çıkar (satmadan)")
        self.forget_btn.clicked.connect(self._forget_selected)
        pos_buttons.addWidget(self.close_btn)
        pos_buttons.addWidget(self.forget_btn)
        pos_buttons.addStretch()
        pos_box = QGroupBox("Açık pozisyonlar")
        pv = QVBoxLayout(pos_box)
        pv.addWidget(self.positions)
        pv.addLayout(pos_buttons)

        self.signals = make_table(["Sembol", "Mum", "Fiyat", "Sinyal", "Açıklama"])
        self.trades = make_table(["Sembol", "Açılış", "Kapanış", "Giriş", "Çıkış", "K/Z", "K/Z %", "Neden"])
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono())
        self.log.setMaximumBlockCount(3000)
        lower = QTabWidget()
        lower.addTab(self.log, "Kayıt")
        lower.addTab(self.signals, "Son sinyaller")
        lower.addTab(self.trades, "İşlem geçmişi")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addLayout(cards)
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.addWidget(pos_box)
        vs.addWidget(lower)
        vs.setSizes([260, 420])
        rl.addWidget(vs)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 1000])
        QVBoxLayout(self).addWidget(split)
        self.refresh_network_label()
        self._engine_changed()
        self._update_autopilot_button()
        self._load_history()

    # ---------------------------------------------------------------- yardımcılar
    # ---------------------------------------------------------------- otomatik pilot
    def _update_autopilot_button(self):
        on = self.ctx.settings.autopilot
        self.autopilot_btn.setText("🚀  Otomatik Pilot: AÇIK (kapatmak için tıkla)" if on else
                                   "🚀  Otomatik Pilotu Başlat (her şey otomatik)")
        color = "#8957e5" if on else "#1f6feb"
        self.autopilot_btn.setStyleSheet(f"QPushButton {{background:{color}; color:white; font-weight:bold; "
                                         "padding:10px;} QPushButton:disabled {background:#30363d;}")
        self.autopilot_btn.setToolTip(
            "Zeka Motoru + haberler + yeni listelemeler + öngörüler + günlük otomatik bakım birlikte çalışır. "
            "Uygulama her açıldığında (Windows ile otomatik başlatma dahil) kendiliğinden devam eder.")

    @Slot()
    def toggle_autopilot(self):
        s = self.ctx.settings
        if s.autopilot:
            s.autopilot = False
            s.start_bot_on_launch = False
            self.ctx.settings_tab.autobot.setChecked(False)
            self.ctx.persist()
            self._update_autopilot_button()
            if self.engine and self.engine.running:
                self.stop()
            self.ctx.status("Otomatik pilot kapatıldı.")
            return
        self.engine_type.setCurrentIndex(self.engine_type.findData("intel"))
        s.autopilot = True
        s.start_bot_on_launch = True
        self.ctx.settings_tab.autobot.setChecked(True)
        self.ctx.persist()
        self._update_autopilot_button()
        self.log.appendPlainText("[Otomatik Pilot] Zeka Motoru, haberler, listelemeler, öngörüler ve otomatik "
                                 "bakım birlikte başlatılıyor.")
        if s.live_mode and not s.unattended_live_confirmed:
            self.log.appendPlainText("[Otomatik Pilot] Not: canlı modda uygulama yeniden açıldığında otomatik devam "
                                     "için Ayarlar'da 'Canlı modda da onay sormadan otomatik başlat' onayı gerekir.")
        if not (self.engine and self.engine.running):
            self.start()

    def is_intel(self) -> bool:
        return self.engine_type.currentData() == "intel"

    def _engine_changed(self):
        intel = self.is_intel()
        self.strategy.setVisible(not intel)
        self.risk.setVisible(not intel)
        self.interval.setEnabled(not intel)
        self.intel_market.setEnabled(intel)
        self.leverage.setEnabled(intel and self.intel_market.currentData() == "USDM_FUTURES")
        self.intel_note.setVisible(intel)
        self.auto_universe.setEnabled(intel)
        self.universe_size.setEnabled(intel)
    def refresh_network_label(self):
        s = self.ctx.settings
        if s.testnet:
            self.net_label.setText("Ağ: <b>TESTNET</b> (sahte bakiye, gerçek para yok). Ayarlar sekmesinden değiştirin.")
            self.net_label.setStyleSheet("color:#ffa726;")
        else:
            self.net_label.setText("Ağ: <b>GERÇEK BINANCE</b> — canlı modda gerçek parayla işlem yapılır!")
            self.net_label.setStyleSheet(f"color:{RED};")
        self.paper_balance.setSuffix(f" {s.quote_asset}")

    def _state_path(self, live: bool):
        if self.is_intel():
            net = "paper" if not live else ("testnet" if self.ctx.settings.testnet else "live")
            return data_dir() / f"state_intel_{net}_{self.intel_market.currentData().lower()}.json"
        if not live:
            return data_dir() / "state_paper.json"
        return data_dir() / ("state_live_testnet.json" if self.ctx.settings.testnet else "state_live.json")

    def _load_history(self):
        """Uygulama açılışında son kağıt/canlı oturumun işlem geçmişini göster."""
        import json
        path = self._state_path(self.live.isChecked())
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        from ..models import ClosedTrade
        self._show_trades([ClosedTrade.from_dict(t) for t in data.get("trades", [])])

    def _collect(self):
        s = self.ctx.settings
        s.symbols = [x.strip().upper() for x in self.symbols.text().replace(";", ",").split(",") if x.strip()]
        s.interval = self.interval.currentText()
        s.poll_seconds = self.poll.value()
        s.strategy = self.strategy.key()
        s.strategy_params = self.strategy.params()
        s.risk = self.risk.settings().to_dict()
        s.paper_balance = self.paper_balance.value()
        s.live_mode = self.live.isChecked()
        s.engine_type = self.engine_type.currentData()
        s.intel_market = self.intel_market.currentData()
        s.intel_leverage = self.leverage.value()
        s.auto_universe = self.auto_universe.isChecked()
        s.universe_size = self.universe_size.value()
        self.ctx.persist()
        return s

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.paper, self.live, self.paper_balance, self.reset_paper, self.symbols, self.interval,
                  self.poll, self.strategy, self.risk, self.engine_type, self.intel_market, self.leverage,
                  self.auto_universe, self.universe_size):
            w.setEnabled(not running)
        if not running:
            self._engine_changed()
        self.ctx.set_bot_running(running)

    # ---------------------------------------------------------------- başlat / durdur
    @Slot()
    def start(self, unattended: bool = False):
        """unattended=True: açılışta otomatik başlatma (onay pencereleri gösterilmez;
        canlı modda yalnızca kullanıcı bunu Ayarlar'da bilinçli olarak onayladıysa çalışır)."""
        if self.engine and self.engine.running:
            return
        s = self._collect()
        live = s.live_mode
        if unattended and live and not s.unattended_live_confirmed:
            self.ctx.status("Canlı mod için otomatik başlatma onaylanmamış; bot başlatılmadı.")
            self.log.appendPlainText("[Otomatik başlatma] Canlı modda onay verilmediği için bot başlatılmadı.")
            return
        self._unattended = unattended
        if self.is_intel():
            self._start_intel(s, live)
            return
        if live:
            if not s.api_key or not s.api_secret:
                self.ctx.show_error("API anahtarı yok", "Canlı işlem için Ayarlar sekmesinden API anahtarlarını girin.")
                return
            net = "TESTNET" if s.testnet else "GERÇEK BINANCE HESABINIZ"
            text = (f"Bot <b>{net}</b> üzerinde gerçek piyasa emirleri gönderecek.<br><br>"
                    f"Semboller: {', '.join(s.symbols)}<br>Strateji: {self.strategy.combo.currentText()}<br>"
                    f"İşlem başı risk: %{s.risk['risk_per_trade_pct']}, maks. pozisyon: %{s.risk['max_position_pct']}"
                    "<br><br>Kripto işlemleri yüksek risk içerir ve kayıp yaşayabilirsiniz. Devam edilsin mi?")
            if not unattended:
                box = QMessageBox(QMessageBox.Icon.Warning, "Canlı işlem onayı", text,
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
                box.setDefaultButton(QMessageBox.StandardButton.No)
                if box.exec() != QMessageBox.StandardButton.Yes:
                    return

        # Kağıt işlemde gerçek piyasa fiyatları (Ayarlar'a göre), canlıda işlem yapılan ağ kullanılır.
        client = self.ctx.trade_client() if live else self.ctx.data_client()
        strategy, risk = self.strategy.create(), self.risk.settings()
        self.start_btn.setEnabled(False)
        self.ctx.status("Bağlantı kontrol ediliyor...")

        def prepare():
            client.sync_time()
            for sym in s.symbols:
                client.symbol_rules(sym)  # sembol geçerli mi?
            if live:
                broker = LiveBroker(client, s.quote_asset, risk.fee_pct)
                broker.quote_balance()  # API anahtarı doğrulaması
            else:
                broker = PaperBroker(s.quote_asset, s.paper_balance, risk.fee_pct, risk.slippage_pct)
            return broker

        def ready(broker):
            notifier = self.ctx.make_notifier()
            gui_emit = self.ctx.bridge.event.emit

            def on_event(kind, payload):
                gui_emit(kind, payload)
                if notifier is not None:
                    notifier.handle_event(kind, payload)

            try:
                self.engine = BotEngine(
                    client, broker, strategy, risk, s.symbols, s.interval, s.quote_asset, s.poll_seconds,
                    state_path=self._state_path(live), on_event=on_event,
                )
            except ValueError as exc:
                self.start_btn.setEnabled(True)
                self.ctx.show_error("Bot başlatılamadı", str(exc))
                return
            if self.notifier is not None:
                self.notifier.stop(timeout=0)
            self.notifier = notifier
            if notifier is not None:
                notifier.attach(self.engine)
                notifier.start()
                self.log.appendPlainText("[Telegram] Bildirimler etkin.")
            self._set_running(True)
            self.ctx.status("Bot çalışıyor.")
            self.log.appendPlainText("")
            self.engine.start()
            self._refresh_positions()
            self._show_trades(self.engine.trades)

        def failed(msg):
            self.start_btn.setEnabled(True)
            self.ctx.show_error("Bot başlatılamadı", msg)

        self.ctx.tasks.run(prepare, ready, failed)

    def _attach_engine(self, engine):
        notifier = self.ctx.make_notifier()
        if self.notifier is not None:
            self.notifier.stop(timeout=0)
        self.notifier = notifier
        if notifier is not None:
            notifier.attach(engine)
            notifier.start()
            self.log.appendPlainText("[Telegram] Bildirimler etkin.")
        self.engine = engine
        self._set_running(True)
        self.ctx.status("Bot çalışıyor.")
        self.log.appendPlainText("")
        engine.start()
        self._refresh_positions()
        self._show_trades(engine.trades)

    def _event_sink(self):
        gui_emit = self.ctx.bridge.event.emit
        holder = {"n": None}

        def on_event(kind, payload):
            gui_emit(kind, payload)
            n = self.notifier
            if n is not None and n.engine is holder.get("engine"):
                n.handle_event(kind, payload)
        return on_event, holder

    def _start_intel(self, s, live: bool):
        """Zeka Motoru'nu (spot/futures, kağıt/canlı) başlatır."""
        from ..intel.config import load_intel_config
        from ..intel.decision import DecisionEngine
        from ..intel.execution import FuturesLiveVenue, PaperVenue, SpotLiveVenue
        from ..intel.futures_client import BinanceFuturesClient
        from ..intel.live_engine import IntelligentBotEngine
        from ..intel.signal_store import SignalStore

        cfg = load_intel_config()
        cfg.market = s.intel_market
        cfg.quote_asset = s.quote_asset
        cfg.risk.leverage = min(s.intel_leverage, cfg.risk.max_leverage) if cfg.market != "SPOT" else 1
        errors = cfg.validate()
        if errors:
            self.ctx.show_error("Zeka Motoru ayarları geçersiz", "\n".join(errors))
            return
        futures_mkt = cfg.market != "SPOT"
        if live:
            if not s.api_key or not s.api_secret:
                self.ctx.show_error("API anahtarı yok", "Canlı işlem için API anahtarı gerekli.")
                return
            net = "TESTNET" if s.testnet else "GERÇEK BINANCE HESABINIZ"
            live_strats = [k for k, sc in cfg.strategies.items() if sc.enabled and sc.stage in ("LIMITED_LIVE",
                                                                                                 "FULL_LIVE")]
            text = (f"Zeka Motoru <b>{net}</b> üzerinde <b>{'USDⓈ-M Futures' if futures_mkt else 'Spot'}</b> "
                    f"piyasasında gerçek emir gönderecek.<br>Semboller: {', '.join(s.symbols)}<br>"
                    f"Canlı izinli stratejiler: {', '.join(live_strats) or 'YOK (hiç işlem açılmaz)'}<br>"
                    f"Kaldıraç: {cfg.risk.leverage}x, işlem başı risk %{cfg.risk.risk_per_trade_pct}<br><br>"
                    "Kaldıraçlı işlemler tasfiye riski taşır. Devam edilsin mi?")
            if not getattr(self, "_unattended", False):
                box = QMessageBox(QMessageBox.Icon.Warning, "Canlı işlem onayı", text,
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
                box.setDefaultButton(QMessageBox.StandardButton.No)
                if box.exec() != QMessageBox.StandardButton.Yes:
                    return
        st = self.ctx.settings
        if live:
            spot = self.ctx.trade_client()
            fut = BinanceFuturesClient(st.api_key, st.api_secret, testnet=st.testnet)
        else:
            spot = self.ctx.data_client()
            fut = BinanceFuturesClient(testnet=not st.mainnet_data and st.testnet)
        market_client = fut if futures_mkt else spot
        self.start_btn.setEnabled(False)
        self.ctx.status("Zeka Motoru hazırlanıyor (Binance bağlantısı ve sembol filtreleri)...")

        def prepare():
            market_client.sync_time()
            for sym in s.symbols:
                market_client.symbol_rules(sym)
            if live:
                venue = FuturesLiveVenue(fut, cfg, s.quote_asset) if futures_mkt else SpotLiveVenue(spot, cfg,
                                                                                                    s.quote_asset)
                venue.available_balance()
            else:
                venue = PaperVenue(cfg.market, s.paper_balance, cfg, s.quote_asset)
            return venue, SignalStore()

        def ready(result):
            venue, store = result
            on_event, holder = self._event_sink()
            try:
                de = DecisionEngine(cfg, meta_model=self.ctx.intel_meta, strategy_stats=self.ctx.intel_stats,
                                    health=self.ctx.intel_health, learned=self.ctx.intel_learned)
                news = None
                if cfg.news.enabled or cfg.listing.enabled or cfg.catalyst.enabled:
                    from .news_tab import build_monitor
                    news = build_monitor(self.ctx.settings, cfg)
                insight_engine = insight_store = None
                if news is not None and cfg.catalyst.enabled:
                    from ..intel.catalyst import InsightEngine, InsightStore
                    insight_engine = InsightEngine(cfg.catalyst, market_client, quote=cfg.quote_asset)
                    insight_store = InsightStore()
                universe = None
                if s.auto_universe:
                    from ..intel.universe import UniverseSelector
                    universe = UniverseSelector(market_client, cfg.quote_asset, s.universe_size,
                                                base_symbols=s.symbols)
                engine = IntelligentBotEngine(cfg, s.symbols, market_client, venue, store, futures_client=fut,
                                              decision_engine=de, poll_seconds=s.poll_seconds,
                                              state_path=self._state_path(live), on_event=on_event,
                                              news_monitor=news, insight_engine=insight_engine,
                                              insight_store=insight_store, universe=universe)
            except ValueError as exc:
                self.start_btn.setEnabled(True)
                self.ctx.show_error("Zeka Motoru başlatılamadı", str(exc))
                return
            holder["engine"] = engine
            self._attach_engine(engine)

        def failed(msg):
            self.start_btn.setEnabled(True)
            self.ctx.show_error("Zeka Motoru başlatılamadı", msg)

        self.ctx.tasks.run(prepare, ready, failed)

    @Slot()
    def stop(self):
        if self.engine:
            self.engine.stop()
            self.stop_btn.setEnabled(False)
            self.cards["Durum"].setText("Durduruluyor...")

    def shutdown(self):
        if self.engine and self.engine.running:
            self.engine.stop(wait=True)
        if self.notifier is not None:
            self.notifier.stop(timeout=5)

    def _reset_paper(self):
        path = self._state_path(False)
        if QMessageBox.question(self, "Kağıt hesabı", "Kağıt işlem bakiyesi, pozisyonları ve geçmişi sıfırlansın mı?") \
                == QMessageBox.StandardButton.Yes:
            path.unlink(missing_ok=True)
            fill_table(self.positions, [])
            fill_table(self.trades, [])
            self.ctx.status("Kağıt hesabı sıfırlandı.")

    def _selected_symbol(self):
        row = self.positions.currentRow()
        item = self.positions.item(row, 0) if row >= 0 else None
        return item.text() if item else None

    def _close_selected(self):
        sym = self._selected_symbol()
        if not sym or not self.engine:
            self.ctx.show_error("Pozisyon", "Bot çalışırken bir pozisyon seçin.")
            return
        if QMessageBox.question(self, "Pozisyonu sat", f"{sym} pozisyonu piyasa fiyatından satılsın mı?") \
                != QMessageBox.StandardButton.Yes:
            return
        engine = self.engine
        self.ctx.tasks.run(lambda: engine.close_position(sym), None,
                           lambda m: self.ctx.show_error("Satış başarısız", m))

    def _forget_selected(self):
        sym = self._selected_symbol()
        if sym and self.engine:
            self.engine.forget_position(sym)

    # ---------------------------------------------------------------- olaylar
    def on_event(self, kind: str, payload):
        if kind == "log":
            self.log.appendPlainText(str(payload))
        elif kind == "status":
            running = payload == "running"
            self.cards["Durum"].setText("Çalışıyor" if running else "Durduruldu")
            self.cards["Durum"].setStyleSheet(
                f"font-size:16px; font-weight:bold; color:{GREEN if running else '#c9d1d9'};")
            if not running:
                self._set_running(False)
        elif kind == "equity":
            q = self.ctx.settings.quote_asset
            self.cards["Serbest bakiye"].setText(fmt_money(payload["quote"], q))
            self.cards["Toplam değer"].setText(fmt_money(payload["equity"], q))
            self._set_card("Günlük K/Z", fmt_pct(payload["day_pct"]), payload["day_pct"])
            self._refresh_positions()
        elif kind == "positions":
            self._refresh_positions()
        elif kind == "trade":
            self._show_trades(self.engine.trades if self.engine else [payload])
        elif kind == "signal":
            self._refresh_signals()

    def _set_card(self, name, text, value):
        self.cards[name].setText(text)
        color = pnl_color(value) or "#c9d1d9"
        self.cards[name].setStyleSheet(f"font-size:16px; font-weight:bold; color:{color};")

    def _refresh_positions(self):
        if not self.engine:
            return
        rows, colors = [], []
        for sym, p in list(self.engine.positions.items()):
            price = self.engine.last_prices.get(sym, p.entry_price)
            pnl, pnl_pct = p.unrealized(price)
            side = getattr(p, "direction", "LONG")
            qty_text = fmt_qty(p.qty) if side == "LONG" else f"SHORT {fmt_qty(p.qty)}"
            rows.append([sym, qty_text, fmt_price(p.entry_price), fmt_price(price),
                         fmt_money(p.qty * price), f"{pnl:+.2f}", fmt_pct(pnl_pct),
                         fmt_price(p.stop_loss) if p.stop_loss else "-",
                         fmt_price(p.take_profit) if p.take_profit else "-", p.opened_at])
            colors.append(pnl_color(pnl))
        fill_table(self.positions, rows, colors)

    def _refresh_signals(self):
        if not self.engine:
            return
        rows = [[sym, d["time"], fmt_price(d["price"]), d["signal"], d["reason"]]
                for sym, d in self.engine.last_signals.items()]
        fill_table(self.signals, rows, [signal_color(r[3]) for r in rows])

    def _show_trades(self, trades):
        trades = list(reversed(trades))
        rows = [[t.symbol, t.opened_at, t.closed_at, fmt_price(t.entry_price), fmt_price(t.exit_price),
                 f"{t.pnl:+.2f}", fmt_pct(t.pnl_pct), t.reason] for t in trades]
        fill_table(self.trades, rows, [pnl_color(t.pnl) for t in trades])
        total = sum(t.pnl for t in trades)
        self._set_card("Gerçekleşen K/Z", fmt_money(total, self.ctx.settings.quote_asset), total)


# ======================================================================== Ayarlar
class SettingsTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        s = ctx.settings
        box = QGroupBox("Binance API")
        form = QFormLayout(box)
        self.api_key = QLineEdit(s.api_key)
        self.api_secret = QLineEdit(s.api_secret)
        self.api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_secret = QCheckBox("Göster")
        self.show_secret.toggled.connect(
            lambda on: self.api_secret.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        secret_row = QHBoxLayout()
        secret_row.addWidget(self.api_secret)
        secret_row.addWidget(self.show_secret)
        self.testnet = QCheckBox("Testnet kullan (testnet.binance.vision — gerçek para yok)")
        self.testnet.setChecked(s.testnet)
        self.mainnet_data = QCheckBox("Analiz, tarayıcı ve backtest için gerçek piyasa verisini kullan (önerilir)")
        self.mainnet_data.setChecked(s.mainnet_data)
        self.quote = QComboBox()
        self.quote.addItems(["USDT", "FDUSD", "USDC", "TRY", "BTC", "EUR"])
        self.quote.setCurrentText(s.quote_asset)
        form.addRow("API Key", self.api_key)
        form.addRow("Secret Key", secret_row)
        form.addRow("", self.testnet)
        form.addRow("", self.mainnet_data)
        form.addRow("Karşı varlık", self.quote)

        buttons = QHBoxLayout()
        save = QPushButton("Kaydet")
        save.clicked.connect(self.save)
        test = QPushButton("Bağlantıyı ve Bakiyeyi Test Et")
        test.clicked.connect(self.test)
        buttons.addWidget(save)
        buttons.addWidget(test)
        buttons.addStretch()

        self.balances = make_table(["Varlık", "Serbest", "Kilitli"])
        self.result = QLabel()
        self.result.setWordWrap(True)

        help_text = QLabel(
            "<b>API anahtarı nasıl alınır?</b><br>"
            "• <b>Testnet</b> (önerilen ilk adım): <a href='https://testnet.binance.vision'>testnet.binance.vision</a>"
            " adresinde GitHub ile giriş yapıp 'Generate HMAC_SHA256 Key' ile anahtar oluşturun.<br>"
            "• <b>Gerçek hesap</b>: Binance → Profil → API Yönetimi → API Oluştur. Yalnızca "
            "<i>Okuma</i> ve <i>Spot ve Marjin İşlemi</i> izinlerini açın.<br>"
            "• <b>Para çekme iznini ASLA açmayın</b> ve mümkünse IP kısıtlaması ekleyin.<br>"
            "• Gizli anahtar bu bilgisayarda Windows DPAPI ile şifrelenerek saklanır.<br><br>"
            "<b>Uyarı:</b> Bu yazılım yatırım tavsiyesi vermez; hiçbir strateji kâr garantisi vermez. "
            "Önce kağıt işlem ve testnet ile deneyin, kaybetmeyi göze alamayacağınız parayla işlem yapmayın."
        )
        help_text.setOpenExternalLinks(True)
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.TextFormat.RichText)

        bg = QGroupBox("Arka planda çalışma (bu bilgisayar)")
        bgl = QVBoxLayout(bg)
        self.tray = QCheckBox("Pencere kapatılınca sistem tepsisinde çalışmaya devam et")
        self.tray.setChecked(s.minimize_to_tray)
        self.nosleep = QCheckBox("Bot çalışırken bilgisayarın uykuya geçmesini engelle (ekran kapanabilir)")
        self.nosleep.setChecked(s.prevent_sleep)
        self.autostart = QCheckBox("Windows açılınca KreatifBot'u başlat (tepside)")
        self.autostart.setChecked(s.autostart)
        self.autobot = QCheckBox("Uygulama açılınca botu son ayarlarla otomatik başlat")
        self.autobot.setChecked(s.start_bot_on_launch)
        self.autolive = QCheckBox("Canlı modda da onay sormadan otomatik başlat (riskli)")
        self.autolive.setChecked(s.unattended_live_confirmed)
        self.autolive.toggled.connect(self._confirm_autolive)
        for w in (self.tray, self.nosleep, self.autostart, self.autobot, self.autolive):
            bgl.addWidget(w)
        bg_note = QLabel("Not: Dizüstü bilgisayarlarda kapak kapanınca uyku Windows güç ayarlarından da kapatılmalı. "
                         "Spot pozisyonların stop/hedefleri uygulama açıkken izlenir; Telegram bildirimleriyle "
                         "uzaktan takip edebilirsiniz.")
        bg_note.setWordWrap(True)
        bg_note.setStyleSheet("color:#8b949e;")
        bgl.addWidget(bg_note)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(bg)
        layout.addLayout(buttons)
        layout.addWidget(self.result)
        layout.addWidget(self.balances, 1)
        layout.addWidget(help_text)

    def _confirm_autolive(self, on: bool):
        if not on or self.ctx.settings.unattended_live_confirmed:
            return
        answer = QMessageBox.warning(
            self, "Canlı modda otomatik başlatma",
            "Bu seçenek açıkken uygulama (ör. elektrik kesintisi sonrası) açıldığında bot CANLI modda onay "
            "sormadan gerçek emir göndermeye başlar.\n\nYalnızca ayarlarınızı test ettiyseniz açın. Devam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            self.autolive.blockSignals(True)
            self.autolive.setChecked(False)
            self.autolive.blockSignals(False)

    def reload(self):
        """API penceresinden yapılan değişiklikleri alanlara yansıt."""
        s = self.ctx.settings
        self.api_key.setText(s.api_key)
        self.api_secret.setText(s.api_secret)
        self.testnet.setChecked(s.testnet)

    @Slot()
    def save(self):
        s = self.ctx.settings
        s.api_key = self.api_key.text().strip()
        s.api_secret = self.api_secret.text().strip()
        s.testnet = self.testnet.isChecked()
        s.mainnet_data = self.mainnet_data.isChecked()
        s.quote_asset = self.quote.currentText()
        s.minimize_to_tray = self.tray.isChecked()
        s.prevent_sleep = self.nosleep.isChecked()
        s.start_bot_on_launch = self.autobot.isChecked()
        s.unattended_live_confirmed = self.autolive.isChecked()
        if self.autostart.isChecked() != s.autostart:
            from ..system import set_autostart
            if set_autostart(self.autostart.isChecked()) or not self.autostart.isChecked():
                s.autostart = self.autostart.isChecked()
            else:
                self.autostart.setChecked(False)
                self.ctx.status("Otomatik başlatma yalnızca Windows'ta ayarlanabilir.")
        self.ctx.persist()
        self.ctx.settings_changed()
        self.ctx.status("Ayarlar kaydedildi.")

    @Slot()
    def test(self):
        self.save()
        client: BinanceClient = self.ctx.trade_client()
        net = "Testnet" if client.testnet else "Gerçek Binance"
        self.result.setText(f"{net} bağlantısı test ediliyor...")

        def work():
            client.ping()
            offset = client.sync_time()
            balances = client.balances() if client.has_keys else None
            return offset, balances

        def done(result):
            offset, balances = result
            msg = f"✔ {net} bağlantısı başarılı (saat farkı {offset} ms)."
            if balances is None:
                msg += " API anahtarı girilmediği için bakiye okunmadı."
                fill_table(self.balances, [])
            else:
                msg += f" API anahtarı geçerli. {len(balances)} varlık bulundu."
                rows = [[a, fmt_qty(b["free"]), fmt_qty(b["locked"])]
                        for a, b in sorted(balances.items(), key=lambda kv: -kv[1]["free"])]
                fill_table(self.balances, rows)
            self.result.setStyleSheet(f"color:{GREEN};")
            self.result.setText(msg)

        def failed(msg):
            self.result.setStyleSheet(f"color:{RED};")
            self.result.setText(f"✘ {msg}")

        self.ctx.tasks.run(work, done, failed)

