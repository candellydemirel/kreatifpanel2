"""Ortak arayüz bileşenleri: grafikler, formlar, tablolar ve arka plan görevleri."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QPainter, QPicture
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHeaderView,
    QLabel, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import indicators as ind
from ..risk import RiskSettings
from ..strategies import STRATEGIES, Strategy, create_strategy
from ..utils import DEFAULT_SYMBOLS, INTERVALS

GREEN = "#26a69a"
RED = "#ef5350"
BLUE = "#42a5f5"
ORANGE = "#ffa726"
PURPLE = "#ab47bc"
GREY = "#78909c"

pg.setConfigOptions(antialias=True, background="#151a21", foreground="#c9d1d9")


# ---------------------------------------------------------------- arka plan görevleri
class _TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _TaskSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001 - kullanıcıya gösterilir
            traceback.print_exc()
            self.signals.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.signals.done.emit(result)


class TaskRunner(QObject):
    """Ağ işlemlerini arayüzü dondurmadan arka planda çalıştırır."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._active: set[Task] = set()

    def run(self, fn, on_done=None, on_error=None):
        task = Task(fn)
        self._active.add(task)

        def finish():
            self._active.discard(task)

        if on_done:
            task.signals.done.connect(on_done)
        if on_error:
            task.signals.failed.connect(on_error)
        task.signals.done.connect(lambda _: finish())
        task.signals.failed.connect(lambda _: finish())
        self.pool.start(task)
        return task


# ---------------------------------------------------------------- grafikler
class CandlestickItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.picture = QPicture()
        self._rect = QRectF()

    def set_data(self, o, h, l, c):
        pic = QPicture()
        p = QPainter(pic)
        width = 0.35
        for i in range(len(o)):
            color = QColor(GREEN if c[i] >= o[i] else RED)
            p.setPen(pg.mkPen(color, width=1))
            p.drawLine(QPointF(i, l[i]), QPointF(i, h[i]))
            p.setBrush(pg.mkBrush(color))
            top, bottom = max(o[i], c[i]), min(o[i], c[i])
            p.drawRect(QRectF(i - width, bottom, 2 * width, max(top - bottom, 1e-12)))
        p.end()
        self.prepareGeometryChange()
        self.picture = pic
        if len(o):
            self._rect = QRectF(-1, float(np.min(l)), len(o) + 1, float(np.max(h) - np.min(l)) or 1.0)
        else:
            self._rect = QRectF()
        self.update()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return self._rect


class IndexDateAxis(pg.AxisItem):
    """X ekseni mum sırası, etiketler tarih."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times: list = []
        self.fmt = "%d.%m %H:%M"

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            i = int(round(v))
            out.append(self.times[i].strftime(self.fmt) if 0 <= i < len(self.times) else "")
        return out


def _date_fmt(times) -> str:
    if len(times) > 1 and (times[-1] - times[0]).total_seconds() / max(len(times) - 1, 1) >= 86400:
        return "%d.%m.%Y"
    return "%d.%m %H:%M"


class PriceChart(pg.GraphicsLayoutWidget):
    """Mum grafiği + EMA/Bollinger + al/sat işaretleri, hacim ve RSI panelleri."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.axes = [IndexDateAxis(orientation="bottom") for _ in range(3)]
        self.price = self.addPlot(row=0, col=0, axisItems={"bottom": self.axes[0]})
        self.volume = self.addPlot(row=1, col=0, axisItems={"bottom": self.axes[1]})
        self.rsi = self.addPlot(row=2, col=0, axisItems={"bottom": self.axes[2]})
        self.ci.layout.setRowStretchFactor(0, 5)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.ci.layout.setRowStretchFactor(2, 2)
        for plot in (self.volume, self.rsi):
            plot.setXLink(self.price)
        for plot in (self.price, self.volume, self.rsi):
            plot.showGrid(x=True, y=True, alpha=0.15)
        self.price.addLegend(offset=(10, 10))
        self.rsi.setYRange(0, 100)
        self.rsi.setLabel("left", "RSI")
        self.volume.setLabel("left", "Hacim")

    def set_data(self, df: pd.DataFrame, buys=None, sells=None, title: str = ""):
        for plot in (self.price, self.volume, self.rsi):
            plot.clear()
        if df is None or df.empty:
            return
        times = list(pd.to_datetime(df["open_time"]))
        fmt = _date_fmt(times)
        for axis in self.axes:
            axis.times, axis.fmt = times, fmt
        x = np.arange(len(df))
        o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))

        candles = CandlestickItem()
        candles.set_data(o, h, lo, c)
        self.price.addItem(candles)
        close = df["close"]
        for series, color, name in (
            (ind.ema(close, 20), ORANGE, "EMA 20"),
            (ind.ema(close, 50), BLUE, "EMA 50"),
        ):
            self.price.plot(x, series.to_numpy(), pen=pg.mkPen(color, width=1.5), name=name, connect="finite")
        mid, upper, lower = ind.bollinger(close)
        bb_pen = pg.mkPen(GREY, width=1, style=Qt.PenStyle.DashLine)
        self.price.plot(x, upper.to_numpy(), pen=bb_pen, name="Bollinger", connect="finite")
        self.price.plot(x, lower.to_numpy(), pen=bb_pen, connect="finite")

        if buys:
            self.price.plot([i for i in buys], [lo[i] * 0.997 for i in buys], pen=None, symbol="t1",
                            symbolBrush=GREEN, symbolPen=None, symbolSize=12, name="AL")
        if sells:
            self.price.plot([i for i in sells], [h[i] * 1.003 for i in sells], pen=None, symbol="t",
                            symbolBrush=RED, symbolPen=None, symbolSize=12, name="SAT")
        if title:
            self.price.setTitle(title)

        brushes = [pg.mkBrush(GREEN if c[i] >= o[i] else RED) for i in range(len(df))]
        self.volume.addItem(pg.BarGraphItem(x=x, height=df["volume"].to_numpy(), width=0.7, brushes=brushes))

        self.rsi.plot(x, ind.rsi(close).to_numpy(), pen=pg.mkPen(PURPLE, width=1.5), connect="finite")
        for level in (30, 70):
            self.rsi.addItem(pg.InfiniteLine(level, angle=0, pen=pg.mkPen(GREY, style=Qt.PenStyle.DotLine)))
        self.rsi.setYRange(0, 100)

        self.price.setXRange(max(0, len(df) - 150), len(df) + 2, padding=0)
        self.price.enableAutoRange(axis="y")
        self.price.setAutoVisible(y=True)


class EquityChart(pg.PlotWidget):
    def __init__(self, parent=None):
        self.axis = IndexDateAxis(orientation="bottom")
        super().__init__(parent, axisItems={"bottom": self.axis})
        self.showGrid(x=True, y=True, alpha=0.15)
        self.addLegend(offset=(10, 10))

    def set_data(self, equity: pd.Series, benchmark: pd.Series | None = None):
        self.clear()
        times = list(pd.to_datetime(equity.index))
        self.axis.times, self.axis.fmt = times, _date_fmt(times)
        x = np.arange(len(equity))
        if benchmark is not None:
            self.plot(x, benchmark.to_numpy(), pen=pg.mkPen(GREY, width=1, style=Qt.PenStyle.DashLine),
                      name="Al-tut")
        self.plot(x, equity.to_numpy(), pen=pg.mkPen(BLUE, width=2), name="Strateji")


# ---------------------------------------------------------------- formlar
def symbol_combo(default: str = "BTCUSDT") -> QComboBox:
    box = QComboBox()
    box.setEditable(True)
    box.addItems(DEFAULT_SYMBOLS)
    box.setCurrentText(default)
    box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    return box


def interval_combo(default: str = "1h") -> QComboBox:
    box = QComboBox()
    box.addItems(list(INTERVALS))
    box.setCurrentText(default)
    return box


def combo_symbol(box: QComboBox) -> str:
    return box.currentText().strip().upper().replace("/", "").replace("-", "")


class StrategyPicker(QGroupBox):
    def __init__(self, title: str = "Strateji", parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        for key, cls in STRATEGIES.items():
            self.combo.addItem(cls.name, key)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color:#8b949e;")
        self.form = QFormLayout()
        layout.addWidget(self.combo)
        layout.addWidget(self.desc)
        layout.addLayout(self.form)
        self.inputs: dict[str, QWidget] = {}
        self.combo.currentIndexChanged.connect(lambda _: self._rebuild())
        self._rebuild()

    def key(self) -> str:
        return self.combo.currentData()

    def _rebuild(self, params: dict | None = None):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.inputs.clear()
        cls = STRATEGIES[self.key()]
        self.desc.setText(cls.description)
        params = params or {}
        for name, (default, lo, hi, label) in cls.param_specs.items():
            if isinstance(default, int):
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setSingleStep(0.05 if hi <= 1 else 0.5)
                w.setRange(float(lo), float(hi))
            w.setValue(params.get(name, default))
            self.inputs[name] = w
            self.form.addRow(label, w)

    def params(self) -> dict:
        return {k: w.value() for k, w in self.inputs.items()}

    def set_strategy(self, key: str, params: dict | None = None):
        idx = self.combo.findData(key)
        if idx >= 0:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(idx)
            self.combo.blockSignals(False)
            self._rebuild(params)

    def create(self) -> Strategy:
        return create_strategy(self.key(), self.params())


class RiskForm(QGroupBox):
    RANGES = {
        "risk_per_trade_pct": (0.1, 10, 0.1),
        "stop_atr_mult": (0, 10, 0.25),
        "take_profit_rr": (0, 10, 0.25),
        "trailing_atr_mult": (0, 10, 0.25),
        "max_position_pct": (1, 100, 1),
        "max_open_positions": (0, 50, 1),
        "max_daily_loss_pct": (0, 50, 0.5),
        "fee_pct": (0, 1, 0.01),
        "slippage_pct": (0, 2, 0.01),
    }

    def __init__(self, title: str = "Risk Yönetimi", parent=None, show_costs: bool = True):
        super().__init__(title, parent)
        form = QFormLayout(self)
        self.inputs: dict[str, QWidget] = {}
        for name, label in RiskSettings.LABELS.items():
            if not show_costs and name in ("slippage_pct",):
                continue
            lo, hi, step = self.RANGES[name]
            if name == "max_open_positions":
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(2 if step >= 0.1 else 3)
                w.setRange(lo, hi)
                w.setSingleStep(step)
            self.inputs[name] = w
            form.addRow(label, w)
        self.set_settings(RiskSettings())

    def set_settings(self, rs: RiskSettings):
        for name, w in self.inputs.items():
            w.setValue(getattr(rs, name))

    def settings(self) -> RiskSettings:
        data = RiskSettings().to_dict()
        data.update({k: w.value() for k, w in self.inputs.items()})
        return RiskSettings.from_dict(data)


# ---------------------------------------------------------------- tablolar
class SortItem(QTableWidgetItem):
    """Görünen metinden bağımsız olarak sayısal değere göre sıralanan hücre."""

    def __lt__(self, other):
        a = self.data(Qt.ItemDataRole.UserRole)
        b = other.data(Qt.ItemDataRole.UserRole)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


def make_table(headers: list[str], stretch_last: bool = True) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.verticalHeader().setVisible(False)
    t.setAlternatingRowColors(True)
    header = t.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    header.setStretchLastSection(stretch_last)
    return t


def fill_table(table: QTableWidget, rows: list[list], colors: list[str | None] | None = None):
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if isinstance(value, tuple):  # (metin, sıralama değeri)
                item = SortItem(str(value[0]))
                item.setData(Qt.ItemDataRole.UserRole, value[1])
            else:
                item = QTableWidgetItem(str(value))
            if colors and colors[r]:
                item.setForeground(QColor(colors[r]))
            table.setItem(r, c, item)


def pnl_color(value: float) -> str | None:
    if value > 0:
        return GREEN
    if value < 0:
        return RED
    return None


def signal_color(text: str) -> str | None:
    if "AL" in text and "SAT" not in text:
        return GREEN
    if "SAT" in text:
        return RED
    return None
