# KreatifBot — Binance Trading Bot (Windows)

Binance Spot piyasası için piyasa analizi, tarama, backtest ve otomatik işlem yapan Windows masaüstü uygulaması.

> ⚠️ **Uyarı:** Bu yazılım yatırım tavsiyesi vermez ve hiçbir strateji kâr garantisi vermez. Önce **Kağıt işlem** ve **Testnet** ile deneyin. Kaybetmeyi göze alamayacağınız parayla işlem yapmayın.

## 🚀 Otomatik Pilot (önerilen kullanım)
**Bot** sekmesindeki **🚀 Otomatik Pilotu Başlat** düğmesine bir kez basın. Sonrasında hiçbir şeye elle basmanız gerekmez; her şey birlikte ve senkronize çalışır:

| Ne | Ne sıklıkla |
|---|---|
| Zeka Motoru kararları, pozisyon yönetimi (SL/TP/trailing/süre) | Her mum kapanışında / her döngüde |
| Haber taraması, Binance listeleme/delist tespiti | 2 dakikada bir |
| Yeni listeleme stratejisi | Listeleme sonrası sürekli |
| Öngörüler (katalizör + temel + teknik) ve gerçekleşen getirileri | 30 dakikada bir |
| Otomatik bakım: strateji istatistikleri, strateji sağlığı (bozulanlar duraklatılır), meta model yeniden eğitimi | Başlangıçtan 30 sn sonra ve 24 saatte bir |

Meta model yalnızca **son verideki ayrı bir doğrulama penceresinde** (örneklem dışı AUC ≥ 0.55) başarılı olursa devreye girer; aksi halde deterministik modda devam edilir. Ekranlar (Haberler, Öngörüler, Zeka Motoru) bottan gelen olaylarla kendiliğinden güncellenir; Telegram açıksa her şey telefonunuza gelir. Otomatik Pilot açıkken uygulama her açıldığında (Windows ile otomatik başlatma dahil) kendiliğinden devam eder. Canlı modda bu otomatik devam için Ayarlar'da ayrıca onay gerekir.

## Özellikler

| Sekme | Ne yapar? |
|---|---|
| 📈 **Piyasa Analizi** | Mum grafiği (EMA 20/50, Bollinger, hacim, RSI), trend, ADX ile rejim tespiti (trend/yatay/volatil), volatilite, destek/direnç, −100…+100 arası birleşik skor ve **GÜÇLÜ AL / AL / NÖTR / SAT / GÜÇLÜ SAT** önerisi. Tüm stratejilerin son mumdaki sinyalleri. |
| 🔎 **Tarayıcı** | En hacimli N pariteyi (veya kendi listenizi) tarar, skora göre sıralar. Stabil coinler ve kaldıraçlı tokenlar elenir. Satıra çift tıklayınca detaylı analizi açılır. |
| 🧪 **Backtest** | Seçilen stratejiyi 20.000 muma kadar geçmiş veride test eder (ücret + kayma dahil). Getiri, al-tut karşılaştırması, kazanma oranı, kâr faktörü, maks. düşüş, Sharpe, sermaye eğrisi ve işlem listesini gösterir. **Tüm Stratejileri Karşılaştır** ile hangisinin o paritede daha iyi çalıştığını görürsünüz. |
| 🤖 **Bot** | Seçilen semboller için arka planda çalışır: sinyal gelince alır, stop-loss / kâr al / iz süren stop / strateji sinyaliyle satar. Kağıt (simülasyon) veya canlı mod. Pozisyonlar diske kaydedilir, uygulama yeniden açıldığında takibe devam edilir. |
| ⚙ **Ayarlar** | API anahtarı, Testnet/gerçek hesap seçimi, bağlantı ve bakiye testi. |
| 📨 **Telegram** | Sinyaller, alım/satım (K/Z ile), stop-loss/kâr al, hatalar, günlük zarar limiti ve günlük özet Telegram'a gelir. `/durum`, `/pozisyonlar`, `/islemler`, `/ozet`, `/durdur` komutları. Analiz ve tarama sonuçlarını tek tıkla gönderme. |
| 🔑 **Binance API Anahtarı** (üst menü) | İlk açılışta otomatik çıkan, her zaman üst menüden açılabilen API giriş penceresi: anahtarı girin, **Bağlantıyı Test Et**, **Kaydet**. |
| 📘 **Yardım → Strateji Rehberi (PDF)** | Uygulamaya gömülü Türkçe rehber: stratejiler, göstergeler, analiz skoru, risk yönetimi, backtest metrikleri, hata kodları. |

## 🧠 Zeka Motoru (Trading Intelligence Engine)
Binance Spot ve USDⓈ-M Futures verileriyle çalışan çok katmanlı karar motoru. Klasik bot korunmuştur; Bot sekmesinde **Motor: Zeka Motoru** seçilerek kullanılır.

| Katman | İçerik |
|---|---|
| Piyasa verisi | Spot: exchangeInfo, ticker, 24s ticker, klines, trades, aggTrades, order book, bookTicker. Futures: klines, mark/index fiyatı, funding (anlık+geçmiş), OI (anlık+geçmiş), long/short oranı, taker alış/satış hacmi, order book, bookTicker. Her değer kaynak etiketli; veri yoksa **UNAVAILABLE** (tasfiye verisi Binance REST'te yok). |
| Sembol motoru | tickSize, stepSize, minQty, maxQty, minNotional, fiyat/miktar hassasiyeti, işlem durumu; emir öncesi filtre uygulama. |
| Veri kalitesi | Eksik/tekrar mum, boşluk, bayat veri, geçersiz OHLC, yetersiz geçmiş, anormal spread/hacim, bayat order book → `DATA_QUALITY_FAILURE`. |
| Göstergeler | EMA/SMA 9-200, VWAP, anchored VWAP, ADX/DI, SuperTrend, PSAR, Ichimoku, RSI, StochRSI, Stokastik, MACD, ROC, Momentum, CCI, Williams %R, MFI, ATR/ATR%, Bollinger (+genişlik), Keltner, tarihsel/gerçekleşen volatilite, volatilite yüzdeliği, RVOL, hacim ivmesi, OBV, delta/CVD, z-score. |
| Piyasa yapısı | Swing high/low, HH/HL/LH/LL, BOS, CHoCH, destek/direnç, eşit tepe/dip, likidite bölgeleri, likidite süpürmesi, FVG, basit order block, önceki gün/hafta yüksek-düşük, 13 mum formasyonu (yalnızca uyum faktörü). |
| Rejim | STRONG_BULL … PANIC, UNKNOWN; BTC trendi, funding ve OI bağlamı. |
| Stratejiler | 33 bağımsız modül (LONG + SHORT, ayrı parametre seti), rejime göre yönlendirme, aç/kapa, yaşam döngüsü aşaması. |
| Skor | LONG/SHORT ayrı 0-100, ağırlıklar ve eşikler ayarlanabilir, çifte sayım yok, çatışma tespiti. |
| AI/ML | Meta-labeling (numpy lojistik regresyon; scikit-learn varsa Random Forest), triple-barrier etiketleri, kalibrasyon, permütasyon önemi, drift (PSI). LSTM/GRU/Transformer/XGBoost/LightGBM bu sürümde yok. |
| Risk | Dinamik SL/TP, çoklu TP, breakeven, kâr kilidi, 7 trailing yöntemi, risk tabanlı boyut, maliyet/EV, portföy ve korelasyon riski, devre kesici, futures tasfiye fiyatı (Binance izole formül). |
| Yürütme | Kağıt + canlı (spot MARKET; futures MARKET + reduceOnly + koruyucu STOP_MARKET), emir öncesi bakiye/filtre/spread/kayma/likidite kontrolü. |
| Kayıt | SQLite sinyal veritabanı (`%APPDATA%\KreatifBot\signals.sqlite3`) ve karar günlüğü; her sinyal için açıklama. |
| Araştırma | Backtest (ücret, spread, kayma, funding, gecikme, kısmi dolum), walk-forward, sağlamlık, ablasyon, kalibrasyon, tutma süresi, çıkış/giriş optimizasyonu, özellik korelasyonu, strateji sağlığı, 12 araştırma sorusu. |

**Önemli:** Hiçbir strateji, eşik veya model kârlı kabul edilmez. Varsayılan aşama **PAPER**'dır; canlıda yalnızca `LIMITED_LIVE` / `FULL_LIVE` aşamasındaki stratejiler işlem açar. Yönetici ayarları **Zeka Motoru → Yönetici ayarları** sekmesinden (JSON) düzenlenir.

## 📘 Strateji ve Kullanım Rehberi (PDF)
[`docs/KreatifBot_Strateji_Rehberi.pdf`](docs/KreatifBot_Strateji_Rehberi.pdf):
uygulamanın yetenekleri, Binance API kurulumu, piyasa analizi skorunun nasıl okunacağı, 10 teknik gösterge,
7 stratejinin AL/SAT koşulları ve hangi piyasada kullanılacağı (strateji seçim matrisi), pozisyon boyutu
formülü ve örnek hesap, backtest metrikleri, canlıya geçiş kontrol listesi, Telegram kurulumu/komutları ve sık görülen hata kodları.

Rehber koddan üretilir; strateji parametreleri ve risk varsayılanları her zaman uygulamayla aynıdır:
`pip install reportlab && python docs/generate_guide.py`

### Stratejiler
- **Akıllı Kombine (önerilen):** ADX ile piyasa rejimini ölçer; trend piyasasında EMA, MACD ve Supertrend'e, yatay piyasada RSI ve Bollinger'e daha çok ağırlık verir. Güçlü düşüş trendinde alım yapmaz.
- **EMA Kesişimi** (EMA 200 trend filtresiyle)
- **RSI Dönüş**
- **MACD Momentum**
- **Bollinger Dönüş**
- **Supertrend**
- **Kanal Kırılımı (Donchian / Turtle)**

Tüm parametreler arayüzden değiştirilebilir. Sinyaller yalnızca **kapanmış mumlarla** hesaplanır (ileriye bakma yok, testlerle doğrulanır).

### Risk yönetimi
- İşlem başı risk (% sermaye) ve ATR tabanlı pozisyon boyutu
- ATR tabanlı stop-loss, risk/ödül oranına göre kâr al, iz süren stop
- Tek pozisyon için maksimum sermaye yüzdesi, maksimum açık pozisyon sayısı
- **Günlük zarar limiti:** aşılırsa o gün yeni pozisyon açılmaz

## Kurulum

### Seçenek 1 — Hazır EXE (önerilen)
1. GitHub'da depo → **Actions** → **Windows EXE derle** → en son başarılı çalıştırma → **Artifacts** bölümünden `KreatifBot-windows` dosyasını indirin.
2. Zip'i açın, `KreatifBot.exe`'yi çalıştırın. (Windows SmartScreen uyarı verirse *Ek bilgi → Yine de çalıştır*.)

`v1.0.0` gibi bir etiket gönderildiğinde EXE otomatik olarak **Releases** sayfasına da eklenir.

### Seçenek 2 — Kaynaktan çalıştırma
1. [Python 3.11+](https://www.python.org/downloads/) kurun (kurulumda *Add Python to PATH* seçin).
2. `baslat.bat` dosyasına çift tıklayın. İlk açılışta gerekli paketler otomatik kurulur.

### Seçenek 3 — EXE'yi kendiniz derleyin
`build_exe.bat` dosyasını çalıştırın. Sonuç: `dist\KreatifBot.exe`.

## Binance API anahtarı
Uygulamada üst menüdeki **🔑 Binance API Anahtarı** butonuna tıklayın (ilk açılışta otomatik açılır) veya **Ayarlar** sekmesini kullanın.

- **Testnet (önce bunu deneyin):** https://testnet.binance.vision → GitHub ile giriş → *Generate HMAC_SHA256 Key*. Ayarlar'da **Testnet kullan** işaretli olmalı.
- **Gerçek hesap:** Binance → Profil → **API Yönetimi** → API oluştur. Yalnızca **Okuma** ve **Spot ve Marjin İşlemi** izinlerini açın.
- 🔒 **Para çekme iznini asla açmayın**, mümkünse IP kısıtlaması ekleyin.
- Gizli anahtar bilgisayarınızda Windows DPAPI ile şifrelenmiş olarak saklanır: `%APPDATA%\KreatifBot\settings.json`.

## 📰 Haberler ve yeni Binance listelemeleri
**Haberler** sekmesi ve Zeka Motoru şu kaynakları tarar:
- **Binance duyuruları:** yeni listeleme, delist, Binance haberleri (herkese açık CMS uç noktası; resmi belgelenmemiştir, biçim değişirse kaynak "ERİŞİLEMİYOR" görünür)
- **Binance exchangeInfo farkı (resmi API):** yeni açılan / işleme başlayan / durdurulan USDT çiftleri
- **RSS:** CoinDesk, Cointelegraph, Decrypt · **CryptoPanic** (isteğe bağlı ücretsiz API anahtarı)

Haberler deterministik kurallarla sınıflandırılır (LISTING, DELISTING, HACK, REGULATION_NEG, LAUNCHPOOL, ...) ve coin sembolleri eşleştirilir. **Haber tek başına işlem açtırmaz:**
- Olumsuz/yüksek önemli haber (hack, exploit, dava, delist) → o coinde yeni LONG açılmaz (`NEWS_RISK`)
- Delist duyurusu → açık pozisyon kapatılır (`EMERGENCY_EXIT`) ve Telegram'a bildirilir
- Olumlu haber skoru yükseltmez (hype kovalanmaz), yalnızca gerekçelerde gösterilir

**Yeni Listeleme stratejisi:** yeni işleme açılan çiftleri `watch_hours` boyunca izler; açılıştan `wait_minutes` sonra açılış aralığının (ilk 15 dk) hacimli taze kırılımında, fiyat VWAP üstündeyse ve kırılımın çok üstüne çıkmamışsa (kovalamama) LONG açar. Stop aralık dibi (en fazla %8), hedefler 1R/2R/3R, en fazla 240 dk, risk normalin ¼'ü, aynı anda en fazla 1 listeleme işlemi. Varsayılan aşama **PAPER**. **Listeleme backtest** gerçek Binance 1 dk verisiyle (ücret + 3× kayma) geçmiş listelemeleri test eder.

## 💡 Öngörü motoru (haber katalizörü + temel + teknik)
**Haberler → 💡 Öngörüler** ve bot çalışırken her 30 dakikada bir:
1. **Katalizör skoru (0-100):** son 72 saatteki haberlerde ortaklık/anlaşma, ETF, kurumsal ilgi, mainnet/yükseltme, büyük borsa listelemesi, benimseme, yatırım turu, yakım/geri alım (olumlu); token kilidi açılımı, hack, dava, delist (olumsuz). Kaynak güvenilirliği, bağımsız kaynak teyidi, 24 saatlik yarılanma ve "söylenti/iddia" ifadeleri hesaba katılır.
2. **Temel skor (0-100):** whitepaper'ın işlevselliği için ölçülebilir vekil veriler — DeFiLlama TVL, 7 günlük TVL değişimi, piyasa değeri/TVL; CoinGecko geliştirici aktivitesi (4 hafta commit), piyasa değeri sırası, proje yaşı, whitepaper linki. Whitepaper metni otomatik değerlendirilmez. Veri yoksa UNAVAILABLE.
3. **Teknik onay (Binance):** rejim düşüşte değil, 1s EMA50 üstü, 4s eğilim, 24s hacim artışı, yeterli likidite. Haberden beri fiyat %15'ten fazla yükseldiyse **fiyatlanmış** sayılır (kovalanmaz).

Sonuç: **🟢 AL** (güçlü katalizör + teknik onay + eşikler) veya **💡 İZLE**. AL sinyalinde ATR stop, 1.5R/3R hedef, en fazla 72 saat, risk normalin yarısı; risk motoru, haber filtresi ve aşama kontrolü uygulanır (varsayılan **PAPER**, yalnızca izinli semboller). Geçmiş haber arşivi olmadığından klasik backtest yapılamaz: her öngörü kaydedilir ve **4s/24s/72s sonraki gerçek getirileri** "Öngörü performansı" tablosunda katalizör türüne göre ölçülür.

## 🖥 Bilgisayarınızda 7/24 çalıştırma
**Ayarlar → Arka planda çalışma** bölümünden:
- **Sistem tepsisi:** pencereyi kapatınca bot durmaz; saatin yanındaki **K** simgesinde çalışır (yeşil = bot çalışıyor). Simgeye tıklayınca pencere açılır, sağ tık → **Çıkış** ile tamamen kapanır.
- **Uyku engeli:** bot çalışırken Windows uykuya geçmez (ekran kapanabilir). Dizüstünde kapak kapatma ayarını Windows güç seçeneklerinden de "Hiçbir şey yapma" yapın.
- **Windows ile başlat:** bilgisayar açılınca uygulama tepside başlar.
- **Botu otomatik başlat:** uygulama açılınca bot son ayarlarla başlar (elektrik kesintisi / yeniden başlatma sonrası). Canlı modda bu, ayrıca onay verilmedikçe çalışmaz.

Uygulama kapalıyken spot pozisyonların stop/hedefleri izlenmez (futures'ta borsa tarafında koruyucu stop vardır). Uzaktan takip için Telegram bildirimlerini açın. Kalıcı 7/24 çalışma için ileride bir Windows VPS önerilir.

## 📨 Telegram bildirimleri
1. Telegram'da **@BotFather** → `/newbot` → bota ad ve sonu `bot` ile biten kullanıcı adı verin, **token**'ı kopyalayın.
2. Yeni botunuzu açıp **/start** yazın.
3. Uygulamada **Telegram** sekmesi → token'ı yapıştırın → **Chat ID'yi otomatik bul** → **Test mesajı gönder** → bildirimleri etkinleştirip **Kaydet**.
4. Botu başlattığınızda bildirimler gelir. Komutlar: `/durum`, `/pozisyonlar`, `/islemler`, `/ozet`, `/durdur`, `/yardim`.

Komutlar yalnızca kayıtlı Chat ID'den kabul edilir. Token, API anahtarı gibi DPAPI ile şifrelenerek saklanır.

## Önerilen kullanım akışı
1. **Tarayıcı** ile güçlü skorlu pariteleri bulun.
2. **Piyasa Analizi**'nde detaylarına bakın.
3. **Backtest → Tüm Stratejileri Karşılaştır** ile o parite ve zaman aralığında en iyi stratejiyi seçin.
4. **Bot**'u önce **Kağıt işlem** modunda birkaç gün çalıştırın.
5. Sonra **Testnet + Canlı** modda deneyin. Ancak memnun kalırsanız gerçek hesaba geçin ve küçük tutarlarla başlayın.

## Notlar
- Yalnızca **Spot** piyasa desteklenir (açığa satış yok). SAT sinyali açık pozisyonu kapatır.
- Bot, kendi açtığı pozisyonları takip eder. Borsada elle sattığınız bir pozisyonu **Takipten çıkar** butonuyla listeden kaldırabilirsiniz.
- Kayıt dosyası: `%APPDATA%\KreatifBot\logs\kreatifbot.log`
- Testnet fiyatları gerçek piyasadan farklıdır. Bu yüzden analiz, tarayıcı, backtest ve kağıt işlem varsayılan olarak gerçek piyasa verisini kullanır (Ayarlar'dan değiştirilebilir).

## Geliştirme
```bash
pip install -r requirements-dev.txt
python -m pytest -q
python main.py
```

Proje yapısı:
```
main.py                     Başlatıcı
kreatifbot/
  binance_client.py         Binance REST istemcisi (HMAC imzalı)
  indicators.py             EMA, RSI, MACD, Bollinger, ATR, ADX, Stokastik, Supertrend, Donchian, OBV
  strategies.py             Stratejiler
  analyzer.py               Piyasa analizi ve skor
  risk.py                   Risk yönetimi
  backtest.py               Backtest motoru ve metrikler
  broker.py                 Kağıt ve canlı emir yürütme
  engine.py                 Bot motoru (arka plan iş parçacığı)
  telegram.py               Telegram bildirimleri ve komutlar
  system.py                 Windows uyku engeli ve otomatik başlatma
  intel/news.py, listing.py Haber kaynakları, listeleme izleme ve Yeni Listeleme stratejisi
  intel/catalyst.py         Öngörü motoru: katalizör, temel (DeFiLlama/CoinGecko), teknik onay, performans takibi
  intel/                    Zeka Motoru: veri, özellik, rejim, MTF, stratejiler, skor, ML, risk, yürütme,
                            pozisyon yönetimi, karar, backtest, araştırma, sinyal veritabanı, canlı motor
  config.py                 Ayarlar ve DPAPI ile anahtar şifreleme
  gui/                      PySide6 arayüzü
tests/                      Birim, motor ve arayüz testleri
```
