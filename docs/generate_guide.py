"""KreatifBot Strateji ve Kullanım Rehberi PDF'ini üretir.

Strateji parametreleri ve risk varsayılanları doğrudan koddan okunur; böylece
rehber uygulamayla her zaman uyumlu kalır.

    pip install reportlab
    python docs/generate_guide.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Polygon, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from kreatifbot import __version__  # noqa: E402
from kreatifbot.risk import RiskSettings  # noqa: E402
from kreatifbot.strategies import STRATEGIES  # noqa: E402

OUT = ROOT / "docs" / "KreatifBot_Strateji_Rehberi.pdf"

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
]

NAVY = colors.HexColor("#0d1b2a")
ACCENT = colors.HexColor("#1f6feb")
GREEN = colors.HexColor("#1a7f64")
RED = colors.HexColor("#c62828")
LIGHT = colors.HexColor("#eef3fb")
GRID = colors.HexColor("#c5d0e0")


def register_fonts():
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body",
                                          boldItalic="Body-Bold")
            return
    raise SystemExit("Türkçe karakter destekli TTF yazı tipi bulunamadı (DejaVuSans veya Arial).")


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Body-Bold", fontSize=28, leading=34,
                                textColor=NAVY, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=13, leading=18, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#44546a")),
        "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=18, leading=23, textColor=NAVY,
                             spaceBefore=6, spaceAfter=10, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=13, leading=17, textColor=ACCENT,
                             spaceBefore=10, spaceAfter=5, keepWithNext=1),
        "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.5, spaceAfter=6),
        "small": ParagraphStyle("small", fontName="Body", fontSize=8.5, leading=11.5,
                                textColor=colors.HexColor("#44546a")),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=11.5),
        "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.8, leading=11.5, textColor=colors.white),
        "warn": ParagraphStyle("warn", fontName="Body", fontSize=9.5, leading=13.5, textColor=RED,
                               borderColor=RED, borderWidth=0.8, borderPadding=7, backColor=colors.HexColor("#fff4f4"),
                               spaceBefore=8, spaceAfter=12),
        "tip": ParagraphStyle("tip", fontName="Body", fontSize=9.5, leading=13.5, textColor=colors.HexColor("#0b4f3c"),
                              borderColor=GREEN, borderWidth=0.8, borderPadding=7,
                              backColor=colors.HexColor("#effaf5"), spaceBefore=8, spaceAfter=12),
    }
    return s


S = None


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="body"):
    return ListFlowable([ListItem(P(i, style), leftIndent=12) for i in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontName="Body")


def table(rows, widths, header=True):
    data = []
    for r, row in enumerate(rows):
        st = "cellb" if header and r == 0 else "cell"
        data.append([P(str(c), st) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for r in range(1 if header else 0, len(rows)):
        if r % 2 == 0:
            cmds.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(cmds))
    return t


# ------------------------------------------------------------------ şemalar
def diagram_cross():
    """EMA kesişimi şeması."""
    d = Drawing(460, 120)
    fast = [(10, 90), (60, 80), (110, 62), (160, 50), (210, 48), (260, 60), (310, 82), (360, 98), (410, 104),
            (450, 106)]
    slow = [(10, 70), (60, 68), (110, 64), (160, 60), (210, 58), (260, 60), (310, 64), (360, 70), (410, 76),
            (450, 80)]
    d.add(PolyLine(fast, strokeColor=colors.HexColor("#f39c12"), strokeWidth=2))
    d.add(PolyLine(slow, strokeColor=ACCENT, strokeWidth=2))
    d.add(Polygon([95, 50, 105, 50, 100, 40], fillColor=RED, strokeColor=None))
    d.add(String(70, 28, "SAT: hızlı EMA aşağı keser", fontName="Body", fontSize=8, fillColor=RED))
    d.add(Polygon([255, 44, 265, 44, 260, 54], fillColor=GREEN, strokeColor=None))
    d.add(String(232, 30, "AL: hızlı EMA yukarı keser", fontName="Body", fontSize=8, fillColor=GREEN))
    d.add(String(380, 110, "Hızlı EMA", fontName="Body", fontSize=8, fillColor=colors.HexColor("#f39c12")))
    d.add(String(380, 66, "Yavaş EMA", fontName="Body", fontSize=8, fillColor=ACCENT))
    return d


def diagram_bands():
    """Bollinger / RSI dönüş şeması."""
    d = Drawing(460, 120)
    upper = [(10 + i * 44, 100) for i in range(11)]
    lower = [(10 + i * 44, 25) for i in range(11)]
    mid = [(10 + i * 44, 62) for i in range(11)]
    price = [(10, 60), (54, 48), (98, 30), (142, 18), (186, 32), (230, 50), (274, 66), (318, 82), (362, 98),
             (406, 106), (450, 92)]
    d.add(PolyLine(upper, strokeColor=colors.grey, strokeDashArray=[3, 3]))
    d.add(PolyLine(lower, strokeColor=colors.grey, strokeDashArray=[3, 3]))
    d.add(PolyLine(mid, strokeColor=colors.HexColor("#9aa7b8")))
    d.add(PolyLine(price, strokeColor=NAVY, strokeWidth=2))
    d.add(Polygon([181, 8, 191, 8, 186, 18], fillColor=GREEN, strokeColor=None))
    d.add(String(196, 6, "AL: fiyat alt banttan içeri döner", fontName="Body", fontSize=8, fillColor=GREEN))
    d.add(Polygon([269, 74, 279, 74, 274, 64], fillColor=RED, strokeColor=None))
    d.add(String(284, 50, "SAT: orta banda ulaşır", fontName="Body", fontSize=8, fillColor=RED))
    d.add(String(12, 104, "Üst bant", fontName="Body", fontSize=7, fillColor=colors.grey))
    d.add(String(12, 66, "Orta bant (SMA 20)", fontName="Body", fontSize=7, fillColor=colors.grey))
    d.add(String(12, 29, "Alt bant", fontName="Body", fontSize=7, fillColor=colors.grey))
    return d


def diagram_risk():
    """Giriş, stop ve kâr al seviyeleri."""
    d = Drawing(460, 110)
    for y, label, col in ((95, "Kâr al = giriş + 2 × (giriş − stop)", GREEN), (55, "Giriş fiyatı", NAVY),
                          (25, "Stop-loss = giriş − 2 × ATR", RED)):
        d.add(Line(20, y, 300, y, strokeColor=col, strokeWidth=1.6))
        d.add(String(310, y - 3, label, fontName="Body", fontSize=8.5, fillColor=col))
    d.add(Line(160, 27, 160, 53, strokeColor=RED, strokeWidth=1))
    d.add(String(165, 37, "risk (R)", fontName="Body", fontSize=8, fillColor=RED))
    d.add(Line(160, 57, 160, 93, strokeColor=GREEN, strokeWidth=1))
    d.add(String(165, 72, "ödül (2R)", fontName="Body", fontSize=8, fillColor=GREEN))
    return d


# ------------------------------------------------------------------ içerik
STRATEGY_TEXT = {
    "ensemble": {
        "logic": "Beş göstergeden oylama yapar: trend oyları (EMA 9/21 yönü, MACD histogramı, Supertrend yönü) ve "
                 "dönüş oyları (RSI bölgesi, Bollinger %B). ADX ile piyasa rejimini ölçer: <b>trend piyasasında</b> "
                 "skorun %80'i trend oylarından, <b>yatay piyasada</b> %65'i dönüş oylarından gelir. Güçlü düşüş "
                 "trendinde (ADX yüksek ve −DI &gt; +DI) skoru sıfırın üstüne çıkarmaz, yani düşen bıçağı tutmaz.",
        "buy": "Birleşik skor <i>giriş eşiğini</i> yukarı keser.",
        "sell": "Skor <i>çıkış eşiğinin</i> altına iner (ya da stop-loss / kâr al tetiklenir).",
        "best": "Rejimi bilinmeyen piyasalar; tek strateji seçemediğinizde varsayılan tercih.",
        "weak": "Çok hızlı tersine dönüşlerde birkaç mum gecikebilir; az ama seçici işlem açar.",
    },
    "ema_cross": {
        "logic": "Kısa vadeli ortalama uzun vadeli ortalamanın üstüne çıktığında momentum yukarı dönmüş kabul edilir. "
                 "EMA 200 trend filtresi yalnızca ana trend yukarıyken alım yapılmasını sağlar.",
        "buy": "Hızlı EMA yavaş EMA'yı yukarı keser (ve fiyat trend EMA'sının üzerindedir).",
        "sell": "Hızlı EMA yavaş EMA'yı aşağı keser.",
        "best": "Uzun soluklu, belirgin trendler (ADX &gt; 25).",
        "weak": "Yatay piyasada sık sahte kesişim (whipsaw) ve küçük zararlar üretir.",
    },
    "rsi": {
        "logic": "RSI, son N mumdaki ortalama kazancı ortalama kayba oranlayarak 0–100 arası momentum ölçer. "
                 "Aşırı satımdan çıkış, satış baskısının bittiğine işaret eder.",
        "buy": "RSI aşırı satım seviyesini (varsayılan 30) aşağıdan yukarı keser.",
        "sell": "RSI aşırı alım seviyesini (varsayılan 70) yukarı keser.",
        "best": "Yatay / bant içinde dalgalanan piyasalar (ADX &lt; 20).",
        "weak": "Güçlü düşüş trendinde RSI uzun süre düşük kalabilir; stop-loss mutlaka açık olmalı.",
    },
    "macd": {
        "logic": "MACD = EMA12 − EMA26; sinyal çizgisi MACD'nin 9 periyotluk EMA'sıdır. Kesişimler momentum "
                 "değişimini gösterir.",
        "buy": "MACD çizgisi sinyal çizgisini yukarı keser (isteğe bağlı trend filtresiyle).",
        "sell": "MACD çizgisi sinyal çizgisini aşağı keser.",
        "best": "Orta vadeli dalgalı trendler; 1s–4s grafikler.",
        "weak": "Yatay piyasada çok sayıda sinyal verir; trend filtresini açmak faydalı olabilir.",
    },
    "bollinger": {
        "logic": "Bantlar 20 periyotluk ortalamanın ±2 standart sapmasıdır. Fiyat istatistiksel olarak aşırı "
                 "uzaklaştığında ortalamaya dönme eğilimindedir.",
        "buy": "Kapanış alt bandın altından tekrar bandın içine döner.",
        "sell": "Fiyat orta banda (veya ayara göre üst banda) ulaşır.",
        "best": "Yatay, ortalamaya dönen piyasalar.",
        "weak": "Kırılım (breakout) dönemlerinde fiyat bantların dışında yürüyebilir.",
    },
    "supertrend": {
        "logic": "ATR tabanlı dinamik bir destek/direnç çizgisidir. Fiyat çizginin üstündeyken trend yukarı, "
                 "altındayken aşağıdır. Volatiliteye göre kendini ayarlar.",
        "buy": "Supertrend yönü düşüşten yükselişe döner.",
        "sell": "Supertrend yönü yükselişten düşüşe döner.",
        "best": "Güçlü, volatil trendler (kripto için popülerdir).",
        "weak": "Yatay piyasada yön sık değişir; çarpanı artırmak sinyalleri azaltır.",
    },
    "breakout": {
        "logic": "Turtle Trading yaklaşımı: son N mumun en yükseği kırıldığında yeni bir trend başlıyor olabilir. "
                 "Çıkış daha kısa bir kanalın en düşüğüyle yapılır.",
        "buy": "Kapanış önceki N mumun en yüksek seviyesini aşar.",
        "sell": "Kapanış önceki M mumun en düşük seviyesinin altına iner.",
        "best": "Sıkışma sonrası güçlü kırılımlar, yeni trend başlangıçları.",
        "weak": "Sahte kırılımlar; kazanma oranı düşük ama kazançlar büyük olma eğilimindedir.",
    },
}

INDICATORS = [
    ["Gösterge", "Ne ölçer?", "Nasıl okunur?"],
    ["EMA (Üstel Hareketli Ort.)", "Son fiyatlara daha çok ağırlık veren ortalama", "Fiyat EMA üstünde = yükseliş eğilimi. EMA20 &gt; EMA50 &gt; EMA200 = güçlü yükseliş dizilimi."],
    ["RSI (14)", "Momentum, 0–100", "&gt;70 aşırı alım, &lt;30 aşırı satım. 50 üstü alıcılar baskın."],
    ["MACD (12, 26, 9)", "Trend momentumu", "Histogram pozitif ve büyüyorsa yükseliş momentumu güçleniyor."],
    ["Bollinger (20, 2)", "Volatilite bantları", "%B &gt; 1 fiyat üst bant üstünde, %B &lt; 0 alt bant altında. Bantların daralması kırılım habercisi."],
    ["ATR (14)", "Ortalama mum aralığı (volatilite)", "Stop mesafesi ve pozisyon boyutu ATR'ye göre ayarlanır."],
    ["ADX (14)", "Trendin gücü (yönü değil)", "&lt;20 zayıf/yatay, 20–25 oluşuyor, &gt;25 trend, &gt;40 çok güçlü. +DI &gt; −DI yükseliş."],
    ["Stokastik (14, 3)", "Kapanışın son aralıktaki konumu", "&gt;80 aşırı alım, &lt;20 aşırı satım."],
    ["Supertrend (10, 3)", "ATR tabanlı trend yönü", "Yeşil/yükseliş: fiyat çizginin üstünde."],
    ["Donchian (20)", "Son N mumun en yüksek/en düşük kanalı", "Üst kanal kırılımı yeni trend başlangıcı olabilir."],
    ["Hacim oranı", "Son mum hacmi / 20 mum ortalaması", "&gt;1.5 güçlü ilgi; hareketin güvenilirliğini artırır."],
]


def intel_section(story, W):
    from kreatifbot.intel.config import IntelConfig
    from kreatifbot.intel.strategies import REGISTRY
    ic = IntelConfig()
    story.append(PageBreak())
    story.append(P("11. Zeka Motoru (Trading Intelligence Engine)", "h1"))
    story.append(P("Zeka Motoru yalnızca AL/SAT üretmez: Binance Spot ve USDⓈ-M Futures verileriyle piyasa rejimini "
                   "sınıflandırır, birden fazla stratejiyi birleştirir, sinyal güvenini hesaplar, giriş/SL/TP ve "
                   "pozisyon süresini dinamik belirler, pozisyonu yönetir ve her kararı açıklar. Gerektiğinde "
                   "<b>NO TRADE</b> der; amaç çok sinyal değil, ölçülebilir ve risk kontrollü fırsatlardır."))
    story.append(P("Karar akışı", "h2"))
    story.append(table([
        ["Adım", "Ne yapılır?"],
        ["1. Veri kalitesi", "Eksik/tekrarlanan mum, zaman boşluğu, bayat veri, geçersiz OHLC, yetersiz geçmiş, anormal "
                             "spread/hacim, bayat order book, sembol durumu. Kritik sorun → NO TRADE "
                             "(DATA_QUALITY_FAILURE)."],
        ["2. Özellikler", "Trend, momentum, volatilite, hacim, order flow (taker delta/CVD), istatistik, piyasa yapısı, "
                          "mum formasyonları; futures funding/OI/long-short/taker ayrı sütunlarda ve kaynak etiketli. "
                          "Veri yoksa UNAVAILABLE, sahte değer yok."],
        ["3. Rejim", "STRONG_BULL, BULL, WEAK_BULL, SIDEWAYS, CHOP, LOW/HIGH_VOLATILITY, BEAR, STRONG_BEAR, PANIC, "
                     "UNKNOWN. BTC trendi, funding ve OI bağlam olarak düzeltir."],
        ["4. Yönlendirici", "Rejime uygun stratejiler aktif olur. PANIC/UNKNOWN: yeni işlem yok. HIGH_VOLATILITY: "
                           "daha yüksek eşik ve yarım boyut. Güçlü trendde ortalamaya dönüş kapalı."],
        ["5. Aday sinyaller", "33 bağımsız strateji yalnızca aday üretir; formasyonlar ve order flow tek başına işlem "
                              "açtırmaz."],
        ["6. Skor", "LONG ve SHORT ayrı 0-100. Gruplar: trend, momentum, hacim, volatilite, yapı, MTF, order flow, "
                    "OI/funding, AI. Aynı bilgi kaynağı tek sayılır (çifte sayım yok). İki yön de güçlü → CONFLICT."],
        ["7. MTF", f"Varsayılan: giriş {ic.timeframes.entry}, onay {ic.timeframes.confirmation}, trend "
                   f"{ic.timeframes.trend}, ana {ic.timeframes.major}, makro {ic.timeframes.macro}. Üst zaman "
                   "dilimleri ters ise güven düşer veya NO TRADE."],
        ["8. Meta model", "Birincil sinyalin başarı olasılığını tahmin eder (triple-barrier etiketli, purged "
                          "eğitim). Eşik altı → NO TRADE. AI risk limitlerini değiştiremez."],
        ["9. Seviyeler", "ATR/swing/yapı/Chandelier/Supertrend stop; R:R, yapı, Fibonacci, AI hedefleri; çoklu TP, "
                         "breakeven, kâr kilidi, trailing (stop asla geri gitmez)."],
        ["10. Maliyet & EV", "Maker/taker ücreti, spread, kayma, funding, gecikme. Beklenen değer maliyetler sonrası "
                             "negatifse NO TRADE."],
        ["11. Risk", "Risk tabanlı boyut, maruziyet, korelasyonlu maruziyet, günlük zarar, maksimum düşüş, art arda "
                     "kayıp, devre kesici, futures tasfiye fiyatı stop'tan önce gelmemeli."],
    ], [W * 0.2, W * 0.8]))
    story.append(P("Stratejiler (33)", "h2"))
    rows = [["Strateji", "Aile", "Stil", "Tercih edilen rejimler (LONG; SHORT aynalanır)"]]
    for key, cls in REGISTRY.items():
        rows.append([f"{cls.spec.name}", cls.spec.family, cls.spec.style,
                     ", ".join(sorted(r.value for r in cls.spec.preferred))])
    story.append(table(rows, [W * 0.3, W * 0.14, W * 0.11, W * 0.45]))
    story.append(P("Futures verisi gerektiren stratejiler (OI, funding, tasfiye) veri yoksa çalışmaz. Binance geçmiş "
                   "tasfiye verisini REST ile sunmadığı için Tasfiye Dönüşü stratejisi varsayılan olarak veri bekler.",
                   "small"))
    story.append(P("Araştırma ve doğrulama", "h2"))
    story.append(bullets([
        "<b>Walk-forward</b>: eğitim → doğrulama → örneklem dışı test; yalnızca test sonuçları raporlanır.",
        "<b>Sağlamlık</b>: parametre pertürbasyonu, eşik duyarlılığı, ücret/kayma stresi, Monte Carlo, rejim ve dönem "
        "ayrımı. Tek parametre kombinasyonunda çalışan strateji sağlam sayılmaz.",
        "<b>Ablasyon</b>: RSI, MACD, hacim, OI, funding, CVD, VWAP, ADX, MTF, AI tek tek çıkarılıp katkı ölçülür.",
        "<b>Kalibrasyon</b>: tahmin edilen olasılık ile gerçekleşen sonuç karşılaştırılır (Brier, ECE).",
        "<b>Tutma süresi</b>: strateji bazında p25/medyan/p75/p90; dikey bariyer araştırma sonucundan belirlenir.",
        "<b>Çıkış/giriş optimizasyonu</b>: örneklem içi ve dışı ayrı raporlanır.",
    ]))
    story.append(P("Strateji yaşam döngüsü", "h2"))
    story.append(P("RESEARCH → BACKTEST → WALK_FORWARD → PAPER → SHADOW → LIMITED_LIVE (yarım risk) → FULL_LIVE. "
                   "Varsayılan aşama PAPER'dır; canlı işlem yalnızca LIMITED_LIVE ve FULL_LIVE stratejilerle yapılır. "
                   "Performans bozulursa strateji DEGRADED/PAUSED olur."))
    story.append(P("Canlı güvenlik: AI yoksa deterministik mod (ayara göre NO TRADE); piyasa verisi, sembol filtreleri, "
                   "veritabanı veya Binance API sorunu varsa yeni emir gönderilmez. Futures canlıda borsa tarafında "
                   "koruyucu STOP_MARKET konur. Backtest sonucu canlı performans garantisi değildir.", "warn"))


def build():
    global S
    register_fonts()
    S = styles()
    story = []
    W = A4[0] - 4 * cm

    # ---------------- Kapak
    story += [Spacer(1, 4.5 * cm), P("KreatifBot", "title"), Spacer(1, 4),
              P("Strateji, Piyasa Analizi ve Kullanım Rehberi", "subtitle"), Spacer(1, 6),
              P(f"Sürüm {__version__} · {date.today():%d.%m.%Y}", "subtitle"), Spacer(1, 2.5 * cm)]
    story.append(table([
        ["Bu rehberde"],
        ["1. Uygulamanın yetenekleri<br/>2. Binance API anahtarı kurulumu<br/>3. Piyasa analizi nasıl okunur<br/>"
         "4. Teknik göstergeler<br/>5. Stratejiler (7 adet) ve hangi piyasada kullanılır<br/>6. Risk yönetimi<br/>"
         "7. Backtest metrikleri ve doğru strateji seçimi<br/>8. Önerilen çalışma akışı ve kontrol listesi<br/>"
         "9. Telegram bildirimleri ve komutları<br/>10. Sık karşılaşılan hatalar<br/>"
         "11. Zeka Motoru (Trading Intelligence Engine)"],
    ], [W * 0.7]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(P("<b>Önemli uyarı:</b> Bu yazılım ve rehber yatırım tavsiyesi değildir. Hiçbir strateji kâr garantisi "
                   "vermez; geçmiş performans gelecekteki sonuçların göstergesi değildir. Kripto paralar çok "
                   "volatildir ve sermayenizin tamamını kaybedebilirsiniz. Önce kağıt işlem ve testnet ile deneyin.",
                   "warn"))
    story.append(PageBreak())

    # ---------------- 1. Yetenekler
    story.append(P("1. Uygulamanın yetenekleri", "h1"))
    story.append(table([
        ["Bölüm", "Ne yapar?"],
        ["📈 Piyasa Analizi".replace("📈 ", ""),
         "Mum grafiği (EMA 20/50, Bollinger, hacim, RSI). Trend, ADX ile rejim, volatilite, destek/direnç, "
         "−100…+100 birleşik skor ve GÜÇLÜ AL / AL / NÖTR / SAT / GÜÇLÜ SAT önerisi. Tüm stratejilerin son mum sinyali."],
        ["Tarayıcı", "En hacimli N pariteyi (veya kendi listenizi) analiz edip skora göre sıralar. Stabil coinler ve "
                     "kaldıraçlı tokenlar otomatik elenir."],
        ["Backtest", "Stratejiyi 20.000 muma kadar geçmiş veride test eder (ücret ve kayma dahil). "
                     "\"Tüm Stratejileri Karşılaştır\" ile o parite için en iyi stratejiyi bulur."],
        ["Bot", "Arka planda çalışır; sinyalde alır, stop-loss / kâr al / iz süren stop / strateji sinyaliyle satar. "
                "Kağıt (simülasyon) ve canlı mod. Pozisyonlar diske kaydedilir."],
        ["Risk yönetimi", "ATR tabanlı pozisyon boyutu, maks. pozisyon yüzdesi, maks. açık pozisyon, günlük zarar limiti."],
        ["Güvenlik", "API gizli anahtarı Windows DPAPI ile şifrelenir. Uygulamada para çekme işlevi yoktur."],
    ], [W * 0.22, W * 0.78]))

    # ---------------- 2. API
    story.append(P("2. Binance API anahtarı kurulumu", "h1"))
    story.append(P("Anahtarı uygulamada üst menüdeki <b>🔑 Binance API Anahtarı</b> butonundan (ilk açılışta otomatik "
                   "açılır) veya <b>Ayarlar</b> sekmesinden girebilirsiniz.".replace("🔑 ", "")))
    story.append(P("A) Testnet anahtarı (önerilen ilk adım, gerçek para yok)", "h2"))
    story.append(bullets([
        "https://testnet.binance.vision adresine gidin ve <b>Log In with GitHub</b> ile giriş yapın.",
        "<b>Generate HMAC_SHA256 Key</b> butonuna basın, bir açıklama yazın.",
        "Görünen <b>API Key</b> ve <b>Secret Key</b> değerlerini kopyalayın (Secret yalnızca bir kez gösterilir).",
        "KreatifBot'ta API penceresine yapıştırın, <b>\"Bu bir TESTNET anahtarı\"</b> kutusunu işaretli bırakın, "
        "<b>Bağlantıyı Test Et</b> ve <b>Kaydet</b>.",
    ]))
    story.append(P("B) Gerçek Binance hesabı", "h2"))
    story.append(bullets([
        "Binance → Profil → <b>API Yönetimi</b> → <b>API Oluştur</b> → <i>Sistem tarafından oluşturulan</i>.",
        "İzinler: yalnızca <b>Okumayı Etkinleştir</b> ve <b>Spot ve Marjin İşlemini Etkinleştir</b>.",
        "<b>Para çekmeyi etkinleştir seçeneğini ASLA açmayın.</b>",
        "Mümkünse <b>IP erişimini kısıtla</b> seçeneğiyle yalnızca kendi IP adresinize izin verin.",
        "KreatifBot'ta testnet kutusunun işaretini kaldırıp anahtarı kaydedin. Durum çubuğunda kırmızı "
        "<b>GERÇEK HESAP</b> rozeti görünür.",
    ]))
    story.append(P("Gizli anahtar yalnızca istekleri imzalamak için kullanılır ve %APPDATA%\\KreatifBot\\settings.json "
                   "içinde Windows DPAPI ile şifrelenmiş olarak saklanır; başka bir bilgisayarda çözülemez.", "tip"))

    # ---------------- 3. Analiz
    story.append(PageBreak())
    story.append(P("3. Piyasa analizi nasıl okunur", "h1"))
    story.append(P("Analiz motoru her parite için aşağıdaki bileşenleri −1 ile +1 arasında puanlar ve ağırlıklı toplamı "
                   "−100…+100 arası <b>genel skora</b> çevirir. ADX 20'nin altındaysa (trend yok) skor %25 zayıflatılır, "
                   "çünkü yatay piyasada yön sinyalleri daha az güvenilirdir."))
    story.append(table([
        ["Bileşen", "Ağırlık", "Pozitif (al yönlü) olduğu durum"],
        ["Trend (EMA 20/50/200)", "%30", "Fiyat &gt; EMA20, EMA20 &gt; EMA50, EMA50 &gt; EMA200"],
        ["Momentum (MACD)", "%20", "Histogram pozitif ve yükseliyor"],
        ["Supertrend", "%25", "Supertrend yönü yukarı"],
        ["RSI", "%15", "RSI 50'nin üstünde (75 üstü aşırı alım cezası, 25 altı tepki bonusu)"],
        ["Hacim", "%10", "Hacim ortalamanın 1.2 katından fazla ve mum yeşil"],
    ], [W * 0.3, W * 0.12, W * 0.58]))
    story.append(Spacer(1, 8))
    story.append(table([
        ["Skor", "Öneri", "Anlamı"],
        ["+50 ve üzeri", "GÜÇLÜ AL", "Göstergelerin çoğu aynı yönde yükseliş gösteriyor"],
        ["+20 … +49", "AL", "Yükseliş eğilimi baskın"],
        ["−19 … +19", "NÖTR", "Kararsız / yatay; işlem için net avantaj yok"],
        ["−20 … −49", "SAT", "Düşüş eğilimi baskın"],
        ["−50 ve altı", "GÜÇLÜ SAT", "Göstergelerin çoğu düşüş gösteriyor"],
    ], [W * 0.2, W * 0.18, W * 0.62]))
    story.append(P("Piyasa rejimi", "h2"))
    story.append(table([
        ["Rejim", "Koşul", "Uygun stratejiler"],
        ["Trend piyasası", "ADX ≥ 25", "Supertrend, EMA Kesişimi, MACD, Kanal Kırılımı, Akıllı Kombine"],
        ["Yatay piyasa (range)", "ADX &lt; 25, volatilite normal", "RSI Dönüş, Bollinger Dönüş, Akıllı Kombine"],
        ["Yüksek volatilite", "ATR/fiyat, son 100 mum medyanının 1.8 katından fazla",
         "Pozisyonu küçültün, stop mesafesini ATR ile genişletin veya bekleyin"],
    ], [W * 0.22, W * 0.33, W * 0.45]))
    story.append(P("Skor bir <b>filtre</b> olarak kullanılmalıdır: örneğin yalnızca skoru +20 üzerindeki paritelerde "
                   "bot çalıştırmak, düşen piyasada alım yapma riskini azaltır.", "tip"))

    # ---------------- 4. Göstergeler
    story.append(P("4. Teknik göstergeler", "h1"))
    story.append(table(INDICATORS, [W * 0.24, W * 0.28, W * 0.48]))

    # ---------------- 5. Stratejiler
    story.append(PageBreak())
    story.append(P("5. Stratejiler", "h1"))
    story.append(P("Tüm stratejiler sinyali yalnızca <b>kapanmış mumla</b> hesaplar (ileriye bakma yok). Spot piyasada "
                   "açığa satış olmadığı için SAT sinyali açık pozisyonu kapatır. Parametreler uygulamada strateji "
                   "seçildiğinde düzenlenebilir; aşağıdaki varsayılanlar koddan otomatik alınmıştır."))
    story.append(KeepTogether([diagram_cross(), P("Şekil 1 — Trend takip mantığı (EMA kesişimi).", "small")]))
    story.append(Spacer(1, 4))
    story.append(KeepTogether([diagram_bands(), P("Şekil 2 — Ortalamaya dönüş mantığı (Bollinger / RSI).", "small")]))

    for key, cls in STRATEGIES.items():
        info = STRATEGY_TEXT[key]
        params = ", ".join(f"{spec[3]}: <b>{spec[0]}</b>" for spec in cls.param_specs.values())
        block = [
            P(cls.name, "h2"),
            P(info["logic"]),
            table([
                ["AL koşulu", info["buy"]],
                ["SAT koşulu", info["sell"]],
                ["En iyi piyasa", info["best"]],
                ["Zayıf yönü", info["weak"]],
                ["Varsayılan parametreler", params],
            ], [W * 0.24, W * 0.76], header=False),
        ]
        story.append(KeepTogether(block))

    story.append(P("Strateji seçim matrisi", "h2"))
    story.append(table([
        ["Piyasa durumu", "1. tercih", "2. tercih", "Kaçının"],
        ["Güçlü yükseliş trendi", "Supertrend", "EMA Kesişimi / Kanal Kırılımı", "Bollinger Dönüş (erken çıkar)"],
        ["Yatay / bant", "RSI Dönüş", "Bollinger Dönüş", "Kanal Kırılımı (sahte kırılım)"],
        ["Sıkışma sonrası kırılım", "Kanal Kırılımı", "MACD Momentum", "RSI Dönüş"],
        ["Belirsiz / karışık", "Akıllı Kombine", "MACD (trend filtresiyle)", "—"],
        ["Güçlü düşüş trendi", "İşlem yapmayın (spot)", "Akıllı Kombine (alım yapmaz)", "RSI / Bollinger dönüş"],
    ], [W * 0.24, W * 0.2, W * 0.3, W * 0.26]))

    # ---------------- 6. Risk
    story.append(PageBreak())
    story.append(P("6. Risk yönetimi", "h1"))
    story.append(P("Uzun vadede hayatta kalmayı strateji değil <b>risk yönetimi</b> belirler. KreatifBot her işlemde "
                   "riski sermayenin sabit bir yüzdesiyle sınırlar."))
    story.append(diagram_risk())
    story.append(P("Pozisyon boyutu formülü", "h2"))
    story.append(P("<b>Miktar = (Sermaye × İşlem başı risk %) ÷ (ATR × Stop ATR çarpanı)</b>, ardından "
                   "<i>Maks. pozisyon %</i> sınırıyla kırpılır."))
    story.append(P("<b>Örnek:</b> Sermaye 1.000 USDT, risk %1 (10 USDT), BTC fiyatı 60.000, ATR 600, stop çarpanı 2 → "
                   "stop mesafesi 1.200 USDT. Miktar = 10 ÷ 1.200 = 0,00833 BTC (≈ 500 USDT). Maks. pozisyon %25 "
                   "olduğundan 250 USDT ile sınırlanır → 0,00417 BTC. Stop: 58.800, kâr al (2R): 62.400. Stop "
                   "tetiklenirse kayıp ≈ 5 USDT (sermayenin %0,5'i)."))
    rs = RiskSettings()
    rows = [["Ayar", "Varsayılan", "Açıklama"]]
    desc = {
        "risk_per_trade_pct": "Tek işlemde kaybedilebilecek en fazla sermaye yüzdesi. Yeni başlayanlar için %0,5–1.",
        "stop_atr_mult": "Stop mesafesi. Düşük = sık stop, yüksek = büyük ama seyrek kayıp. 0 = stop yok (önerilmez).",
        "take_profit_rr": "Kâr hedefi, stop mesafesinin katı. 2 → 1 risk için 2 ödül. 0 = sabit hedef yok.",
        "trailing_atr_mult": "Fiyat yükseldikçe stop'u yukarı çeker; trendlerde kârı korur. 0 = kapalı.",
        "max_position_pct": "Tek pozisyona ayrılabilecek en fazla sermaye.",
        "max_open_positions": "Aynı anda açık tutulabilecek pozisyon sayısı.",
        "max_daily_loss_pct": "Gün içi toplam değer bu kadar düşerse o gün yeni pozisyon açılmaz.",
        "fee_pct": "Binance spot ücreti genelde %0,1 (BNB ile ödemede %0,075).",
        "slippage_pct": "Piyasa emrinde beklenen fiyat kayması (kağıt işlem / backtest).",
    }
    for name, label in RiskSettings.LABELS.items():
        rows.append([label, str(getattr(rs, name)), desc[name]])
    story.append(table(rows, [W * 0.3, W * 0.13, W * 0.57]))
    story.append(P("<b>Başlangıç için önerilen ayarlar:</b> risk %0,5–1 · stop 2 × ATR · kâr al 2R · iz süren stop 0 "
                   "veya 2,5 · maks. pozisyon %20–25 · maks. 3 açık pozisyon · günlük zarar limiti %3–5.", "tip"))

    # ---------------- 7. Backtest
    story.append(P("7. Backtest metrikleri ve doğru strateji seçimi", "h1"))
    story.append(P("Backtest'te sinyal bir mumun kapanışında oluşur, emir bir sonraki mumun açılışında gerçekleşir. "
                   "Stop ve kâr al mum içi en düşük/en yüksek ile kontrol edilir; ikisi aynı mumda tetiklenirse "
                   "temkinli davranılıp stop kabul edilir."))
    story.append(table([
        ["Metrik", "Anlamı", "İyi değer"],
        ["Toplam getiri", "Test sonunda sermayedeki değişim", "Al-tut getirisinden iyi olması tercih edilir"],
        ["Al-tut getirisi", "Başta alıp sonda satsaydınız", "Karşılaştırma ölçütü"],
        ["Kazanma oranı", "Kârla kapanan işlemlerin oranı", "Trend stratejilerinde %35–45 normaldir"],
        ["Kâr faktörü", "Toplam kâr ÷ toplam zarar", "&gt; 1,3 iyi, &gt; 1,7 çok iyi, &lt; 1 zarar"],
        ["Maks. düşüş", "Zirveden en derin geri çekilme", "Ne kadar küçükse o kadar iyi (&lt; %20)"],
        ["Sharpe oranı", "Getiri / oynaklık (yıllık)", "&gt; 1 iyi, &gt; 2 çok iyi"],
        ["Piyasada kalma", "Pozisyonda geçen süre oranı", "Düşükse sermaye daha az risk altında"],
    ], [W * 0.2, W * 0.42, W * 0.38]))
    story.append(P("Aşırı uyum (overfitting) tuzağı", "h2"))
    story.append(bullets([
        "Parametreleri tek bir döneme göre mükemmelleştirmeyin; farklı dönemler ve paritelerde de test edin.",
        "En az 30 işlem içermeyen sonuçlar istatistiksel olarak güvenilir değildir.",
        "Önce eski veriyle (ör. 4.000 mum) seçin, sonra son dönemde doğrulayın (ileri test).",
        "Kağıt işlem sonuçları backtest'e benziyorsa ancak o zaman gerçek parayla küçük tutarla başlayın.",
    ]))

    # ---------------- 8. Akış
    story.append(PageBreak())
    story.append(P("8. Önerilen çalışma akışı", "h1"))
    story.append(table([
        ["Adım", "Nerede?", "Ne yapılır?"],
        ["1", "Tarayıcı", "En hacimli 25 pariteyi 1s veya 4s aralıkta tarayın; skoru +20 üzerindekileri not edin."],
        ["2", "Piyasa Analizi", "Adaylarda rejime (trend / yatay) ve destek/dirence bakın."],
        ["3", "Backtest", "\"Tüm Stratejileri Karşılaştır\". Kâr faktörü &gt; 1,3, maks. düşüş &lt; %20 olanı seçin."],
        ["4", "Bot → Kağıt", "Seçilen strateji ve parametrelerle 1–2 hafta kağıt işlem yapın."],
        ["5", "Bot → Canlı + Testnet", "Emir akışını ve hata durumlarını testnette doğrulayın."],
        ["6", "Bot → Canlı + Gerçek", "Küçük sermayeyle başlayın, haftalık sonuçları backtest ile karşılaştırın."],
    ], [W * 0.08, W * 0.22, W * 0.7]))
    story.append(P("Canlıya geçmeden önce kontrol listesi", "h2"))
    story.append(bullets([
        "API anahtarında para çekme izni KAPALI, IP kısıtlaması AÇIK.",
        "Stop-loss açık (stop ATR çarpanı &gt; 0) ve günlük zarar limiti tanımlı.",
        "İşlem başı risk %1 veya altında.",
        "Hesapta yalnızca kaybetmeyi göze alabileceğiniz miktar var.",
        "Bilgisayar ve internet bağlantısı bot çalışırken açık kalacak (bot kapalıyken stop-loss izlenmez).",
        "Kayıt (log) dosyası düzenli kontrol ediliyor: Yardım → Veri / kayıt klasörünü aç.",
    ]))
    story.append(P("Not: Bot, stop-loss ve kâr al seviyelerini kendisi izler ve piyasa emriyle kapatır; borsaya "
                   "bekleyen stop emri koymaz. Bu nedenle bot kapalıyken pozisyonlar korunmaz.", "warn"))

    # ---------------- 9. Telegram
    story.append(PageBreak())
    story.append(P("9. Telegram bildirimleri ve komutları", "h1"))
    story.append(P("KreatifBot, bot çalışırken olan her şeyi Telegram'a gönderebilir ve Telegram'dan komut alabilir. "
                   "Kurulum <b>Telegram</b> sekmesinden yapılır."))
    story.append(P("Kurulum", "h2"))
    story.append(bullets([
        "Telegram'da <b>@BotFather</b> ile konuşun, <b>/newbot</b> yazın; bota bir ad ve sonu <i>bot</i> ile biten "
        "bir kullanıcı adı verin.",
        "BotFather'ın verdiği <b>token</b>'ı (ör. 123456789:ABC...) Telegram sekmesine yapıştırın.",
        "Telegram'da yeni botunuzu açıp <b>/start</b> yazın.",
        "<b>Chat ID'yi otomatik bul</b> → <b>Test mesajı gönder</b> → <b>Kaydet</b>.",
        "Botu Bot sekmesinden başlatın; bildirimler otomatik başlar.",
    ]))
    story.append(P("Gönderilen bildirimler", "h2"))
    story.append(table([
        ["Bildirim", "Örnek"],
        ["Bot başlatıldı / durduruldu", "🟢 KreatifBot başlatıldı · Mod: KAĞIT · Strateji · Semboller".replace("🟢 ", "")],
        ["AL / SAT sinyalleri", "AL sinyali — BTCUSDT · Fiyat · Mum zamanı · gerekçe"],
        ["Alım", "ALIM — BTCUSDT · miktar @ fiyat · tutar · stop-loss · kâr al"],
        ["Satım", "SATIŞ — BTCUSDT (Stop-loss / Kâr al / Strateji sinyali) · K/Z +12,40 USDT (+2,48%)"],
        ["Risk uyarısı", "Günlük zarar limiti aşıldı — bugün yeni pozisyon açılmayacak"],
        ["Hata / uyarı", "Bağlantı, bakiye, minimum işlem tutarı gibi sorunlar"],
        ["Günlük özet", "Seçilen saatte: kapanan işlem sayısı, kazanan/kaybeden, gerçekleşen K/Z, toplam değer"],
    ], [W * 0.28, W * 0.72]))
    story.append(P("Ayrıca <b>Piyasa Analizi</b> ve <b>Tarayıcı</b> sekmelerindeki <b>Telegram'a gönder</b> butonlarıyla "
                   "analiz ve tarama sonuçlarını tek tıkla paylaşabilirsiniz."))
    story.append(P("Komutlar", "h2"))
    story.append(table([
        ["Komut", "Ne yapar?"],
        ["/durum", "Bot durumu, mod, strateji, toplam değer, serbest bakiye, günlük K/Z"],
        ["/pozisyonlar", "Açık pozisyonlar ve anlık kâr/zarar"],
        ["/islemler", "Son 10 kapanmış işlem"],
        ["/ozet", "Bugünün özeti"],
        ["/durdur", "Botu uzaktan durdurur (açık pozisyonlar satılmaz)"],
        ["/yardim", "Komut listesi"],
    ], [W * 0.2, W * 0.8]))
    story.append(P("Güvenlik: komutlar yalnızca kayıtlı Chat ID'den kabul edilir, başka kişilerden gelen mesajlar "
                   "yok sayılır. Token bilgisayarınızda şifrelenerek saklanır. Token'ı kimseyle paylaşmayın; "
                   "sızarsa BotFather'da <b>/revoke</b> ile yenileyin.", "tip"))

    # ---------------- 10. Hatalar
    story.append(P("10. Sık karşılaşılan hatalar", "h1"))
    story.append(table([
        ["Hata", "Neden", "Çözüm"],
        ["-2015 / -2014 Invalid API-key", "Anahtar yanlış, IP izni yok veya testnet/gerçek seçimi uyuşmuyor",
         "Anahtarı yeniden kopyalayın, testnet kutusunu kontrol edin, IP kısıtlamasına bilgisayarınızın IP'sini ekleyin"],
        ["-1021 Timestamp", "Bilgisayar saati Binance'ten farklı",
         "Uygulama otomatik eşitler; sürerse Windows saat eşitlemesini açın"],
        ["-2010 Insufficient balance", "Yeterli USDT yok", "Bakiye ekleyin veya maks. pozisyon yüzdesini düşürün"],
        ["-1013 / NOTIONAL", "İşlem tutarı Binance minimumunun altında (genelde 5 USDT)",
         "Sermayeyi veya işlem başı riski artırın"],
        ["-1121 Invalid symbol", "Sembol yanlış yazılmış", "BTCUSDT biçiminde, araya işaret koymadan yazın"],
        ["Bağlantı hatası", "İnternet / güvenlik duvarı / bölgesel kısıtlama", "Bağlantıyı ve erişimi kontrol edin"],
        ["Telegram: Token geçersiz", "Token eksik/yanlış kopyalanmış", "BotFather'dan token'ı tekrar kopyalayın"],
        ["Telegram: Sohbet bulunamadı", "Bota hiç /start yazılmamış veya Chat ID yanlış",
         "Botu açıp /start yazın, Chat ID'yi otomatik bulun"],
    ], [W * 0.24, W * 0.36, W * 0.4]))
    intel_section(story, W)
    story.append(Spacer(1, 10))
    story.append(P("KreatifBot · Bu belge uygulamayla birlikte otomatik üretilir (docs/generate_guide.py).", "small"))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.HexColor("#7a869a"))
        if doc.page > 1:
            canvas.drawString(2 * cm, 1.2 * cm, "KreatifBot — Strateji ve Kullanım Rehberi")
            canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Sayfa {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="KreatifBot Strateji ve Kullanım Rehberi",
                            author="KreatifBot", subject="Binance trading bot stratejileri ve kullanım")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Oluşturuldu: {OUT}")


if __name__ == "__main__":
    build()
