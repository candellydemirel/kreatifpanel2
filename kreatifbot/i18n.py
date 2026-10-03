"""Kodların Türkçe karşılıkları (ekranlar, Telegram, açıklamalar ve günlük için).

İç mantık İngilizce kodlarla çalışmaya devam eder (veritabanı, testler, filtreler);
kullanıcıya gösterilen her yerde `tr()` ile Türkçe etiket kullanılır.
"""

from __future__ import annotations

REGIME_TR = {
    "STRONG_BULL": "Güçlü yükseliş", "BULL": "Yükseliş", "WEAK_BULL": "Zayıf yükseliş", "SIDEWAYS": "Yatay",
    "CHOP": "Kararsız/dalgalı", "LOW_VOLATILITY": "Düşük oynaklık", "HIGH_VOLATILITY": "Yüksek oynaklık",
    "BEAR": "Düşüş", "STRONG_BEAR": "Güçlü düşüş", "PANIC": "Panik", "UNKNOWN": "Bilinmiyor",
}

STATUS_TR = {
    "SCANNING": "Taranıyor", "CANDIDATE": "Aday", "CONFIRMED": "Onaylandı", "ACTIVE": "Aktif",
    "WEAKENING": "Zayıflıyor", "EXPIRED": "Süresi doldu", "CANCELLED": "İptal", "EXECUTED": "Uygulandı",
    "CLOSED": "Kapandı", "NO_TRADE": "İşlem yok",
    "DEGRADED": "Zayıfladı", "PAUSED": "Duraklatıldı", "DISABLED": "Devre dışı",
}

EXIT_TR = {
    "TP_HIT": "Kâr al", "SL_HIT": "Zarar durdur", "TRAILING_STOP": "İz süren stop", "TIME_EXPIRY": "Süre doldu",
    "SIGNAL_REVERSAL": "Sinyal tersine döndü", "SIGNAL_DECAY": "Sinyal zayıfladı",
    "REGIME_CHANGE": "Piyasa rejimi değişti", "RISK_LIMIT": "Risk limiti", "EMERGENCY_EXIT": "Acil çıkış",
    "MANUAL_EXIT": "Elle kapatma", "EXECUTION_FAILURE": "Emir hatası", "END_OF_DATA": "Veri sonu",
    "BREAKEVEN": "Başa baş stop", "PROFIT_LOCK": "Kâr kilidi", "DELIST": "Delist duyurusu",
}

NO_TRADE_TR = {
    "DATA_QUALITY_FAILURE": "Veri kalitesi yetersiz", "DATA_UNAVAILABLE": "Veri yok",
    "INSUFFICIENT_DATA": "Yetersiz veri", "STALE_DATA": "Eski veri", "LOW_LIQUIDITY": "Düşük likidite",
    "SPREAD_TOO_HIGH": "Alış-satış farkı çok yüksek", "ABNORMAL_VOLATILITY": "Anormal oynaklık",
    "CONFLICTING_TIMEFRAMES": "Zaman dilimleri çelişiyor", "LOW_CONFIDENCE": "Düşük güven",
    "CONFLICT": "LONG/SHORT çelişkisi", "POOR_RR": "Risk/ödül zayıf", "EDGE_BELOW_COSTS": "Avantaj maliyetin altında",
    "NEGATIVE_EXPECTANCY": "Negatif beklenti", "EXCESSIVE_PORTFOLIO_RISK": "Portföy riski fazla",
    "CORRELATED_EXPOSURE": "Korelasyonlu pozisyon fazla", "REGIME_INCOMPATIBLE": "Piyasa rejimine uygun değil",
    "NO_CANDIDATE": "Uygun strateji sinyali yok", "AI_UNCERTAIN": "Yapay zekâ kararsız",
    "META_MODEL_REJECT": "Yapay zekâ modeli reddetti", "API_ERROR": "API hatası",
    "CIRCUIT_BREAKER": "Devre kesici (zarar limiti)", "SHORT_NOT_SUPPORTED_ON_SPOT": "Spot piyasada SHORT yok",
    "POSITION_SIZE_FAILURE": "Pozisyon boyutu hesaplanamadı", "SYMBOL_FILTER_UNKNOWN": "Sembol kuralları bilinmiyor",
    "LIQUIDATION_RISK": "Tasfiye riski", "STRATEGY_NOT_LIVE": "Strateji canlıya hazır değil",
    "STRATEGY_PAUSED": "Strateji duraklatıldı", "POSITION_EXISTS": "Zaten açık pozisyon var",
    "DATABASE_FAILURE": "Veritabanı hatası", "NEWS_RISK": "Olumsuz haber riski",
}

NEWS_CAT_TR = {
    "DELISTING": "Delist (işlemden kaldırma)", "HACK": "Hack/Saldırı", "REGULATION_NEG": "Olumsuz düzenleme",
    "FUTURES_LISTING": "Vadeli listeleme", "LISTING": "Yeni listeleme", "LAUNCHPOOL": "Launchpool/Airdrop",
    "PARTNERSHIP": "Ortaklık/Anlaşma", "MACRO": "Makroekonomi", "GENERAL": "Genel",
}

GROUP_TR = {
    "trend": "Trend", "momentum": "Momentum", "volume": "Hacim", "volatility": "Oynaklık", "structure": "Yapı",
    "mtf": "Çoklu zaman", "orderflow": "Emir akışı", "oi_funding": "OI/Fonlama", "ai": "Yapay zekâ",
}

DIRECTION_TR = {"LONG": "LONG (alış)", "SHORT": "SHORT (açığa satış)", "NONE": "İşlem yok", "NO TRADE": "İşlem yok"}

STAGE_TR = {
    "RESEARCH": "Araştırma", "BACKTEST": "Geçmiş test", "WALK_FORWARD": "İleri test", "PAPER": "Kağıt işlem",
    "SHADOW": "Gölge", "LIMITED_LIVE": "Sınırlı canlı", "FULL_LIVE": "Tam canlı",
}

MISC_TR = {
    "UNAVAILABLE": "Veri yok", "OK": "Var", "NEUTRAL": "Nötr", "POSITIVE": "Pozitif", "NEGATIVE": "Negatif",
    "ELEVATED_POSITIVE": "Yüksek pozitif", "ELEVATED_NEGATIVE": "Yüksek negatif",
    "EXTREME_POSITIVE": "Aşırı pozitif (long kalabalık)", "EXTREME_NEGATIVE": "Aşırı negatif (short kalabalık)",
    "BULLISH_CONFIRMATION": "Yükselişi onaylıyor", "BEARISH_CONFIRMATION": "Düşüşü onaylıyor",
    "BULLISH_DIVERGENCE_CANDIDATE": "Pozitif uyumsuzluk adayı",
    "BEARISH_DIVERGENCE_CANDIDATE": "Negatif uyumsuzluk adayı",
    "PRICE_UP+OI_UP": "Fiyat ↑ + OI ↑", "PRICE_UP+OI_DOWN": "Fiyat ↑ + OI ↓",
    "PRICE_DOWN+OI_UP": "Fiyat ↓ + OI ↑", "PRICE_DOWN+OI_DOWN": "Fiyat ↓ + OI ↓",
    "Bullish": "Yükseliş", "Hafif bullish": "Hafif yükseliş", "Bearish": "Düşüş", "Hafif bearish": "Hafif düşüş",
    "SPOT": "Spot", "USDM_FUTURES": "USDⓈ-M Vadeli", "AL": "AL", "İZLE": "İZLE", "YOK": "Sinyal yok",
}

FAMILY_TR = {
    "adx": "ADX", "breakout": "Kırılım", "derivatives": "Türev (vadeli)", "ema": "Hareketli ortalama",
    "macd": "MACD", "mean_reversion": "Ortalamaya dönüş", "meta": "Birleşik", "momentum": "Momentum",
    "orderflow": "Emir akışı", "statistical": "İstatistiksel", "structure": "Piyasa yapısı",
    "supertrend": "SuperTrend", "volatility": "Oynaklık", "vwap": "VWAP",
}
STYLE_TR = {"intraday": "Gün içi", "scalp": "Kısa vadeli (scalp)", "swing": "Birkaç gün (swing)"}

LISTING_KIND_TR = {"NEW_SYMBOL": "Yeni işlem çifti", "NOW_TRADING": "İşleme açıldı", "ANNOUNCED": "Duyuruldu",
                   "HALTED": "İşlem durduruldu", "STATUS": "Durum değişti"}

_ALL: dict[str, str] = {}
for _d in (FAMILY_TR, STYLE_TR, MISC_TR, REGIME_TR, STATUS_TR, EXIT_TR, NO_TRADE_TR, NEWS_CAT_TR, GROUP_TR, DIRECTION_TR, STAGE_TR):
    _ALL.update(_d)


def tr(code, default: str | None = None) -> str:
    """Kodun Türkçe karşılığı; bilinmeyen kod olduğu gibi döner."""
    if code is None:
        return default or "-"
    key = str(code)
    return _ALL.get(key, _ALL.get(key.upper(), default if default is not None else key))


def tr_reason(reason: str) -> str:
    """'TP_HIT', 'SL_HIT (stop)' gibi çıkış gerekçelerinin baştaki kodunu Türkçeleştirir."""
    if not reason:
        return ""
    head, sep, rest = str(reason).partition(" ")
    label = _ALL.get(head)
    return f"{label}{sep}{rest}" if label else str(reason)
