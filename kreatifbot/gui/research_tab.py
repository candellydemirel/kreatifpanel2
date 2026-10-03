"""Araştırma sekmesi: gerçek Binance verisiyle backtest, walk-forward, sağlamlık,
ablasyon, kalibrasyon, tutma süresi, çıkış/giriş optimizasyonu ve S&C.

Gerçek veri alınamazsa DATA_UNAVAILABLE gösterilir; sahte sonuç üretilmez.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..intel.config import load_intel_config
from ..intel.research import QUESTIONS, ResearchContext, answer
from .intel_tab import df_to_table, fmt_cell, mono
from .widgets import EquityChart, make_table

if TYPE_CHECKING:
    from .main_window import MainWindow


class _Progress(QObject):
    message = Signal(str)


class ResearchTab(QWidget):
    def __init__(self, ctx: "MainWindow"):
        super().__init__()
        self.ctx = ctx
        self.cache: dict = {}
        self.rc = ResearchContext()
        self.progress = _Progress()
        self.progress.message.connect(self._on_progress)

        cfg = load_intel_config()
        top = QGridLayout()
        self.symbol = QComboBox()
        self.symbol.setEditable(True)
        self.symbol.addItems(cfg.allowed_symbols)
        self.market = QComboBox()
        self.market.addItem("Spot", "SPOT")
        self.market.addItem("USDⓈ-M Futures", "USDM_FUTURES")
        self.bars = QSpinBox()
        self.bars.setRange(1000, 30000)
        self.bars.setSingleStep(1000)
        self.bars.setValue(6000)
        top.addWidget(QLabel("Sembol"), 0, 0)
        top.addWidget(self.symbol, 0, 1)
        top.addWidget(QLabel("Piyasa"), 0, 2)
        top.addWidget(self.market, 0, 3)
        top.addWidget(QLabel("Giriş TF mum sayısı"), 0, 4)
        top.addWidget(self.bars, 0, 5)
        self.buttons = []
        actions = [("Backtest", self.run_backtest), ("Walk-Forward (OOS)", self.run_wf),
                   ("Sağlamlık", self.run_robust), ("Ablasyon", self.run_ablation),
                   ("Çıkış optimizasyonu", self.run_exit), ("Giriş optimizasyonu", self.run_entry),
                   ("Meta modeli eğit", self.run_train_meta)]
        btn_row = QHBoxLayout()
        for text, fn in actions:
            b = QPushButton(text)
            b.clicked.connect(fn)
            self.buttons.append(b)
            btn_row.addWidget(b)
        btn_row.addStretch()
        self.status = QLabel("Gerçek Binance geçmiş verisi kullanılır. Uzun testler birkaç dakika sürebilir.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#8b949e;")

        # sonuç sekmeleri
        self.summary = make_table(["Metrik", "Değer"])
        self.by_regime = make_table(["-"])
        self.by_strategy = make_table(["-"])
        self.holding = make_table(["-"])
        self.calib = make_table(["-"])
        self.equity = EquityChart()
        self.trades = make_table(["-"])
        self.wf_table = make_table(["-"])
        self.wf_text = QPlainTextEdit()
        self.wf_text.setReadOnly(True)
        self.wf_text.setFont(mono())
        self.wf_imp = make_table(["-"])
        self.rob_text = QPlainTextEdit()
        self.rob_text.setReadOnly(True)
        self.rob_text.setFont(mono())
        self.abl = make_table(["-"])
        self.opt_exit = make_table(["-"])
        self.opt_entry = make_table(["-"])
        self.corr = make_table(["-"])

        def page(*widgets, vertical=True):
            w = QWidget()
            lay = QVBoxLayout(w)
            sp = QSplitter()
            if vertical:
                from PySide6.QtCore import Qt
                sp.setOrientation(Qt.Orientation.Vertical)
            for x in widgets:
                sp.addWidget(x)
            lay.addWidget(sp)
            return w

        qa = QWidget()
        ql = QVBoxLayout(qa)
        qrow = QHBoxLayout()
        self.question = QComboBox()
        for k, q in QUESTIONS.items():
            self.question.addItem(q, k)
        ask = QPushButton("Cevapla")
        ask.clicked.connect(self.ask)
        qrow.addWidget(self.question, 1)
        qrow.addWidget(ask)
        self.answer = QPlainTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setFont(mono())
        ql.addLayout(qrow)
        ql.addWidget(self.answer)

        self.results = QTabWidget()
        self.results.addTab(page(self.summary, self.equity), "Özet")
        self.results.addTab(page(self.by_regime, self.by_strategy), "Rejim / Strateji")
        self.results.addTab(page(self.holding, self.calib), "Süre / Kalibrasyon")
        self.results.addTab(self.trades, "İşlemler")
        self.results.addTab(page(self.wf_table, self.wf_text, self.wf_imp), "Walk-Forward")
        self.results.addTab(page(self.rob_text), "Sağlamlık")
        self.results.addTab(page(self.abl, self.corr), "Ablasyon / Korelasyon")
        self.results.addTab(page(self.opt_exit, self.opt_entry), "Çıkış / Giriş")
        self.results.addTab(qa, "Araştırma soruları")

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(btn_row)
        lay.addWidget(self.status)
        lay.addWidget(self.results)

    # ---------------------------------------------------------------- altyapı
    def _on_progress(self, msg: str):
        self.status.setText(msg)

    def _busy(self, busy: bool):
        for b in self.buttons:
            b.setEnabled(not busy)

    def _prepared(self):
        """(engine, prep) — gerçek Binance verisi, önbellekli. Veri yoksa ValueError (DATA_UNAVAILABLE)."""
        from ..binance_client import BinanceClient
        from ..intel.decision import DecisionEngine
        from ..intel.features import compute_features
        from ..intel.futures_client import BinanceFuturesClient
        from ..intel.market_data import load_history
        cfg = load_intel_config()
        cfg.market = self.market.currentData()
        symbol = self.symbol.currentText().strip().upper()
        key = (symbol, cfg.market, tuple(cfg.timeframes.as_dict().values()), self.bars.value())
        de = DecisionEngine(cfg, meta_model=self.ctx.intel_meta, strategy_stats=None, health=None)
        if key in self.cache:
            prep = self.cache[key]
            return de, prep
        self.progress.message.emit(f"{symbol}: Binance geçmiş verisi indiriliyor ({self.bars.value()} mum)...")
        bundle = load_history(cfg, symbol, self.bars.value(), BinanceClient(testnet=False),
                              BinanceFuturesClient(testnet=False))
        if not bundle.ok:
            raise ValueError("DATA_UNAVAILABLE: " + "; ".join(bundle.errors))
        self.progress.message.emit("Özellikler hesaplanıyor...")
        btc_f = compute_features(bundle.btc_trend, cfg.timeframes.trend, cfg.market) \
            if bundle.btc_trend is not None and not bundle.btc_trend.empty else None
        prep = de.prepare(bundle.entry, symbol, cfg.timeframes.entry, bundle.htf, bundle.derivatives, btc_f)
        prep.f.attrs["load_errors"] = bundle.errors
        self.cache = {key: prep}
        return de, prep

    def _run(self, work, done, label):
        self._busy(True)
        self.status.setText(label)

        def ok(res):
            self._busy(False)
            done(res)

        def fail(msg):
            self._busy(False)
            self.status.setText(f"✘ {msg}")
            self.ctx.status(msg)

        self.ctx.tasks.run(work, ok, fail)

    def _data_note(self, prep) -> str:
        f = prep.f
        errs = f.attrs.get("load_errors", [])
        unav = [k for k, v in f.attrs.get("availability", {}).items() if str(v).startswith("UNAVAILABLE")]
        txt = (f"Veri: {prep.symbol} {prep.market} {prep.interval}, {len(f)} mum "
               f"({f['open_time'].iat[0]:%Y-%m-%d} – {f['open_time'].iat[-1]:%Y-%m-%d}).")
        if unav:
            txt += " Eksik: " + ", ".join(unav) + "."
        if errs:
            txt += " Uyarı: " + "; ".join(errs[:3])
        return txt

    # ---------------------------------------------------------------- eylemler
    @Slot()
    def run_backtest(self):
        def work():
            from ..intel.backtest import run_intel_backtest
            de, prep = self._prepared()
            self.progress.message.emit("Backtest çalışıyor...")
            return prep, run_intel_backtest(de, prep, 250)

        def done(res):
            from ..intel.research import confidence_calibration, holding_recommendation
            prep, r = res
            self.rc.backtest = r
            if r.status != "OK":
                self.status.setText(f"{r.status}: {r.metrics.get('reason', '')}")
                return
            rows = [[k, fmt_cell(v)] for k, v in r.metrics.items()]
            rows += [["Sinyal istatistikleri", fmt_cell(str(r.signal_stats))]]
            rows += [["Varsayımlar", "; ".join(r.assumptions)]]
            from .widgets import fill_table
            fill_table(self.summary, rows)
            if len(r.equity):
                bench = prep.f.set_index(pd.to_datetime(prep.f["open_time"], utc=True))["close"].reindex(
                    r.equity.index)
                self.equity.set_data(r.equity, bench / bench.iloc[0] * r.equity.iloc[0])
            df_to_table(self.by_regime, r.by_regime, "net_pnl")
            df_to_table(self.by_strategy, r.by_strategy, "net_pnl")
            df_to_table(self.holding, holding_recommendation(r.trades, prep.interval))
            df_to_table(self.calib, confidence_calibration(r.trades))
            tdf = r.trades_frame()
            if not tdf.empty:
                tdf = tdf.drop(columns=["partial_exits"])
            df_to_table(self.trades, tdf, "net_pnl")
            self.status.setText(self._data_note(prep) + " Not: tam veri backtest'i örneklem içidir; performans için "
                                                       "Walk-Forward sonuçlarına bakın.")
            self.results.setCurrentIndex(0)

        self._run(work, done, "Backtest hazırlanıyor...")

    @Slot()
    def run_wf(self):
        def work():
            from ..intel.research import walk_forward
            de, prep = self._prepared()
            n = len(prep.f) - 250
            cfg = de.cfg
            tr, va, te = cfg.backtest.walk_forward_train, cfg.backtest.walk_forward_validation, \
                cfg.backtest.walk_forward_test
            if n < tr + va + te:  # veri azsa oranla
                tr, va, te = int(n * 0.5), int(n * 0.2), int(n * 0.15)
            return walk_forward(de, prep, train=tr, validation=va, test=te, progress=self.progress.message.emit)

        def done(wf):
            self.rc.walk_forward = wf
            if wf.status != "OK":
                self.wf_text.setPlainText(f"{wf.status}\n" + "\n".join(wf.notes))
                self.results.setCurrentIndex(4)
                return
            df_to_table(self.wf_table, wf.param_table, "test_getiri_%")
            lines = ["ÖRNEKLEM DIŞI (OOS) SONUÇLAR — yalnızca test pencereleri:"]
            lines += [f"  {k}: {fmt_cell(v)}" for k, v in wf.oos_metrics.items()]
            lines += [""] + wf.notes
            if not wf.calibration.empty:
                lines += ["", "Meta model kalibrasyonu (tahmin vs gerçekleşen):"]
                lines += [f"  {r.bin}: n={r.n}, tahmin {r.predicted:.2f}, gerçek {r.actual:.2f}"
                          for r in wf.calibration.itertuples()]
            self.wf_text.setPlainText("\n".join(lines))
            df_to_table(self.wf_imp, wf.importance_stability)
            self.results.setCurrentIndex(4)
            self.status.setText("Walk-forward tamamlandı.")

        self._run(work, done, "Walk-forward hazırlanıyor...")

    @Slot()
    def run_robust(self):
        def work():
            from ..intel.research import robustness
            de, prep = self._prepared()
            return robustness(de, prep, progress=self.progress.message.emit)

        def done(rb):
            self.rc.robustness = rb
            parts = [f"KARAR: {rb.verdict}", ""] + rb.notes
            for title, df in (("Parametre pertürbasyonu", rb.perturbation), ("Güven eşiği duyarlılığı", rb.threshold),
                              ("Ücret stresi", rb.fee_stress), ("Kayma stresi", rb.slippage_stress),
                              ("Rejimlere göre", rb.regimes), ("Dönemlere göre", rb.periods),
                              ("Strateji × rejim", rb.per_strategy_regime)):
                parts += ["", f"== {title} ==", df.to_string(index=False) if not df.empty else "veri yok"]
            parts += ["", "== Monte Carlo ==", str(rb.monte_carlo)]
            self.rob_text.setPlainText("\n".join(parts))
            self.results.setCurrentIndex(5)
            self.status.setText("Sağlamlık testi tamamlandı.")

        self._run(work, done, "Sağlamlık testi hazırlanıyor...")

    @Slot()
    def run_ablation(self):
        def work():
            from ..intel.research import ablation, feature_correlation
            de, prep = self._prepared()
            ab = ablation(de, prep, progress=self.progress.message.emit)
            corr, pairs = feature_correlation(prep)
            return ab, pairs

        def done(res):
            ab, pairs = res
            self.rc.ablation = ab
            df_to_table(self.abl, ab, "Δ net_getiri_%")
            df_to_table(self.corr, pd.DataFrame(pairs, columns=["özellik A", "özellik B", "korelasyon"]))
            self.results.setCurrentIndex(6)
            self.status.setText("Ablasyon tamamlandı. Yüksek korelasyonlu özellikler aynı bilgiyi taşır "
                                "(skorlamada tek grup sayılır).")

        self._run(work, done, "Ablasyon hazırlanıyor...")

    @Slot()
    def run_exit(self):
        def work():
            from ..intel.research import exit_optimization
            de, prep = self._prepared()
            return exit_optimization(de, prep, progress=self.progress.message.emit)

        def done(df):
            df_to_table(self.opt_exit, df, "OOS_getiri_%")
            self.results.setCurrentIndex(7)
            self.status.setText("Çıkış optimizasyonu: IS (ilk %60) ve OOS (son %40) karşılaştırması.")

        self._run(work, done, "Çıkış modları test ediliyor...")

    @Slot()
    def run_entry(self):
        def work():
            from ..intel.research import entry_optimization
            de, prep = self._prepared()
            return entry_optimization(de, prep, progress=self.progress.message.emit)

        def done(df):
            df_to_table(self.opt_entry, df, "OOS_getiri_%")
            self.results.setCurrentIndex(7)
            self.status.setText("Giriş optimizasyonu tamamlandı (maliyet ve kayma dahil).")

        self._run(work, done, "Giriş modları test ediliyor...")

    @Slot()
    def run_train_meta(self):
        def work():
            from ..intel.backtest import run_intel_backtest, strategy_stats_from_trades
            from ..intel.research import strategy_health, train_meta
            de, prep = self._prepared()
            n = len(prep.f)
            meta, labels = train_meta(prep, de.cfg, 250, n, purge_until=n, version=f"{prep.symbol}-{n}")
            r = run_intel_backtest(de, prep, 250, record_decisions=False)
            stats = strategy_stats_from_trades(r.trades, prep.interval)
            health = strategy_health(r.trades)
            return meta, stats, health

        def done(res):
            meta, stats, health = res
            if not meta.ready:
                self.status.setText(f"Meta model eğitilemedi: {meta.metrics.get('status')}")
                return
            self.ctx.intel_meta = meta
            self.ctx.intel_stats = stats
            self.ctx.intel_health = {r["strateji"]: r["durum"] for _, r in health.iterrows()} if not health.empty else {}
            self.status.setText(
                f"Meta model hazır ({meta.version}, {meta.trained_samples} örnek, eğitim AUC "
                f"{meta.metrics.get('train_auc', float('nan')):.3f}). Bu oturumdaki karar ve botta kullanılır. "
                "Örneklem dışı kalitesini Walk-Forward ile doğrulayın; eğitim AUC'si performans kanıtı değildir.")

        self._run(work, done, "Meta model eğitiliyor (purged triple-barrier etiketleri)...")

    def ask(self):
        self.rc.decision = self.ctx.intel_last_decision
        self.answer.setPlainText(answer(self.question.currentData(), self.rc))
