# KreatifBot — Binance Trading Bot (Windows)

Binance Spot piyasası için piyasa analizi, tarama, backtest ve otomatik işlem yapan Windows masaüstü uygulaması.

> ⚠️ **Uyarı:** Bu yazılım yatırım tavsiyesi vermez ve hiçbir strateji kâr garantisi vermez. Önce **Kağıt işlem** ve **Testnet** ile deneyin. Kaybetmeyi göze alamayacağınız parayla işlem yapmayın.

## Özellikler

| Sekme | Ne yapar? |
|---|---|
| 📈 **Piyasa Analizi** | Mum grafiği (EMA 20/50, Bollinger, hacim, RSI), trend, ADX ile rejim tespiti (trend/yatay/volatil), volatilite, destek/direnç, −100…+100 arası birleşik skor ve **GÜÇLÜ AL / AL / NÖTR / SAT / GÜÇLÜ SAT** önerisi. Tüm stratejilerin son mumdaki sinyalleri. |
| 🔎 **Tarayıcı** | En hacimli N pariteyi (veya kendi listenizi) tarar, skora göre sıralar. Stabil coinler ve kaldıraçlı tokenlar elenir. Satıra çift tıklayınca detaylı analizi açılır. |
| 🧪 **Backtest** | Seçilen stratejiyi 20.000 muma kadar geçmiş veride test eder (ücret + kayma dahil). Getiri, al-tut karşılaştırması, kazanma oranı, kâr faktörü, maks. düşüş, Sharpe, sermaye eğrisi ve işlem listesini gösterir. **Tüm Stratejileri Karşılaştır** ile hangisinin o paritede daha iyi çalıştığını görürsünüz. |
| 🤖 **Bot** | Seçilen semboller için arka planda çalışır: sinyal gelince alır, stop-loss / kâr al / iz süren stop / strateji sinyaliyle satar. Kağıt (simülasyon) veya canlı mod. Pozisyonlar diske kaydedilir, uygulama yeniden açıldığında takibe devam edilir. |
| ⚙ **Ayarlar** | API anahtarı, Testnet/gerçek hesap seçimi, bağlantı ve bakiye testi. |

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
- **Testnet (önce bunu deneyin):** https://testnet.binance.vision → GitHub ile giriş → *Generate HMAC_SHA256 Key*. Ayarlar'da **Testnet kullan** işaretli olmalı.
- **Gerçek hesap:** Binance → Profil → **API Yönetimi** → API oluştur. Yalnızca **Okuma** ve **Spot ve Marjin İşlemi** izinlerini açın.
- 🔒 **Para çekme iznini asla açmayın**, mümkünse IP kısıtlaması ekleyin.
- Gizli anahtar bilgisayarınızda Windows DPAPI ile şifrelenmiş olarak saklanır: `%APPDATA%\KreatifBot\settings.json`.

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
  config.py                 Ayarlar ve DPAPI ile anahtar şifreleme
  gui/                      PySide6 arayüzü
tests/                      Birim, motor ve arayüz testleri
```
