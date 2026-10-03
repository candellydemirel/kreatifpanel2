"""Zeka Motoru bot motoru (kağıt / canlı; Spot veya USDⓈ-M Futures).

Her döngüde: veri al → veri kalitesi → (yeni mumda) karar → sinyal veritabanı →
(onaylı ise) ön kontroller → emir → pozisyon yönetimi (her döngüde fiyat, her
yeni mumda süre/trailing/sinyal zayıflaması/rejim) → çıkış → performans kaydı.

Canlı güvenlik: piyasa verisi, sembol filtreleri, risk motoru, veritabanı veya
Binance API sorunu varsa yeni emir gönderilmez (NO TRADE). AI yoksa deterministik
moda düşülür (ayara göre olasılık zorunluysa NO TRADE).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..binance_client import BinanceAPIError
from ..engine import EngineCore
from ..i18n import LISTING_KIND_TR, tr, tr_reason
from ..models import ClosedTrade
from ..utils import INTERVALS
from .config import IntelConfig
from .data_quality import check_candles, check_orderbook
from .decision import DecisionEngine, DecisionObject
from .execution import ExecutionError, pretrade_check
from .market_data import closed_only
from .orderflow import orderbook_features
from .position_manager import ManagedPosition, evaluate_signal_decay, open_position, update_on_bar
from .risk_engine import OpenExposure, PortfolioState, return_correlation
from .signal_store import DatabaseError, SignalStore
from .types import ExitReason, Regime, SignalStatus

logger = logging.getLogger("kreatifbot.intel.live")


def _finite(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if x == x and abs(x) != float("inf") else 0.0


class IntelligentBotEngine(EngineCore):
    def __init__(self, cfg: IntelConfig, symbols: list[str], market_client, venue, store: SignalStore | None,
                 futures_client=None, decision_engine: DecisionEngine | None = None, poll_seconds: float = 20,
                 state_path: str | Path | None = None, on_event=None, kline_limit: int = 600,
                 btc_client=None, news_monitor=None, insight_engine=None, insight_store=None, universe=None,
                 approvals=None):
        self.cfg = cfg
        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not self.symbols:
            raise ValueError("En az bir sembol girin.")
        bad = [s for s in self.symbols if not s.endswith(cfg.quote_asset)]
        if bad:
            raise ValueError(f"Bu semboller {cfg.quote_asset} ile bitmiyor: {', '.join(bad)}")
        not_allowed = [s for s in self.symbols if cfg.allowed_symbols and s not in cfg.allowed_symbols]
        if not_allowed:
            raise ValueError(f"İzin verilen sembol listesinde değil (Zeka Motoru ayarları): {', '.join(not_allowed)}")
        self.universe = universe              # otomatik coin seçimi (None → yalnızca elle girilen semboller)
        self.approvals = approvals            # ApprovalBook → her yeni pozisyon için manuel onay; None → otomatik
        self.batch_size = 25                  # Her turda derin analiz edilen coin sayısı (sırayla döner)
        self._batch_pos = 0
        if cfg.timeframes.entry not in cfg.allowed_timeframes:
            raise ValueError(f"Giriş zaman dilimi izinli değil: {cfg.timeframes.entry}")
        self.market = cfg.market
        self.client = market_client
        self.futures = futures_client
        self.btc_client = btc_client or market_client
        self.venue = venue
        self.broker = venue  # GUI/Telegram uyumluluğu (is_live)
        self.store = store
        self.engine = decision_engine or DecisionEngine(cfg)
        if not self.engine.learned:
            from .autopilot import load_learned
            self.engine.learned = load_learned()
        self.interval = cfg.timeframes.entry
        self.quote_asset = cfg.quote_asset
        self.strategy = SimpleNamespace(name=f"Zeka Motoru ({'Futures' if self.market != 'SPOT' else 'Spot'})")
        self.kline_limit = kline_limit
        self.state_path = Path(state_path) if state_path else None
        self._init_core(poll_seconds, on_event)

        self.positions: dict[str, ManagedPosition] = {}
        self.trades: list[ClosedTrade] = []
        self.last_prices: dict[str, float] = {}
        self.last_signals: dict[str, dict] = {}
        self.last_decisions: dict[str, DecisionObject] = {}
        self.pending: dict[str, DecisionObject] = {}
        self._last_bar: dict[str, pd.Timestamp] = {}
        self._cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
        self._rules: dict = {}
        self._api_errors: list[float] = []
        self.halted_today = False
        self._day = None
        self._day_start_equity = 0.0
        self._peak_equity = 0.0
        self._consecutive_losses = 0
        self.news = news_monitor
        self.listing_watch: dict[str, float] = {}      # sembol -> izlemeye alınma zamanı (epoch)
        self.listing_signals: dict = {}
        self.insights = insight_engine
        self.insight_store = insight_store
        self.last_insights: list = []
        self._last_insight_scan = 0.0
        self._tradable_bases: set = set()
        self.metas: dict = {}
        self.summary_every_s = 900
        self._summary_reset()
        self._last_summary = 0.0
        self.maintenance_result = None
        self._maint_thread = None
        self._load_state()

    # ------------------------------------------------------------------ yardımcılar
    def start_message(self) -> str:
        mode = "CANLI" if self.venue.is_live else "KAĞIT (simülasyon)"
        tf = self.cfg.timeframes
        return (f"Zeka Motoru başlatıldı | {self.market} | Mod: {mode} | Semboller: {', '.join(self.symbols)} | "
                f"Giriş {tf.entry}, onay {tf.confirmation}, trend {tf.trend}, ana {tf.major}, makro {tf.macro}")

    # ------------------------------------------------------------------ otomatik pilot bakımı
    def start(self):
        self._live_readiness()
        super().start()
        if self.cfg.autopilot.enabled and (self._maint_thread is None or not self._maint_thread.is_alive()):
            import threading
            self._maint_thread = threading.Thread(target=self._maintenance_loop, name="IntelMaintenance",
                                                  daemon=True)
            self._maint_thread.start()

    def _live_readiness(self):
        """Canlı modda botun neden işlem açamayacağını baştan açıkça söyler."""
        if not getattr(self.venue, "is_live", False):
            return
        live = [s.key for s in self.engine.strategies
                if self.engine.stage_of(s.key).value in ("LIMITED_LIVE", "FULL_LIVE")]
        if not live:
            self.log("UYARI: Canlı moddasınız ama hiçbir strateji 'Sınırlı canlı' / 'Tam canlı' aşamasında değil → "
                     "bot gerçek emir AÇMAZ, yalnızca analiz eder. Önce 'Kağıt işlem' modunda deneyin ya da "
                     "Zeka Motoru → Stratejiler'den aşamayı yükseltin.", logging.WARNING)
        try:
            free = float(self.venue.available_balance())
        except Exception as exc:  # noqa: BLE001
            self.log(f"Bakiye okunamadı: {exc}", logging.WARNING)
            return
        if free < 10:
            where = "Spot" if self.market == "SPOT" else "USDⓈ-M Vadeli"
            self.log(f"UYARI: Binance {where} cüzdanınızda serbest {self.quote_asset} {free:.2f} → bot alım "
                     f"yapamaz (Binance'in en küçük emir tutarı ~5-10 {self.quote_asset}). Binance'te "
                     f"{self.quote_asset}'yi Fonlama cüzdanından {where} cüzdanına aktarın.", logging.WARNING)

    # ------------------------------------------------------------------ kullanıcıya özet
    def _summary_reset(self):
        from collections import Counter
        self._sum_n = 0
        self._sum_trades = 0
        self._sum_reasons = Counter()
        self._sum_symbols: set = set()
        self._sum_best = None

    def _summary_add(self, d):
        self._sum_n += 1
        self._sum_symbols.add(d.symbol)
        if d.is_trade:
            self._sum_trades += 1
            return
        for r in d.no_trade_reasons[:1]:
            self._sum_reasons[r] += 1
        ls, ss = _finite(d.long_score), _finite(d.short_score)
        score = max(ls, ss)
        if self._sum_best is None or score > self._sum_best[1]:
            side = "LONG" if ls >= ss else "SHORT"
            self._sum_best = (d.symbol, score, side, d.no_trade_reasons[0] if d.no_trade_reasons else "")

    def _summary_maybe_log(self):
        if not self._sum_n or time.time() - self._last_summary < self.summary_every_s:
            return
        first = self._last_summary == 0
        self._last_summary = time.time()
        span = "ilk tarama" if first else f"son ~{max(1, round(self.summary_every_s / 60))} dk"
        parts = [f"Özet ({span}): {len(self._sum_symbols)} coin için {self._sum_n} analiz, "
                 f"{self._sum_trades} işlem sinyali."]
        if self._sum_reasons:
            parts.append("İşlem açılmama nedenleri: " + ", ".join(
                f"{tr(k)} ({v})" for k, v in self._sum_reasons.most_common(3)) + ".")
        if self._sum_best is not None:
            sym, score, side, why = self._sum_best
            parts.append(f"En yakın aday: {sym} {tr(side)} {score:.0f}/100" + (f" — {tr(why)}" if why else "") + ".")
        if not self._sum_trades:
            parts.append("Bot kurallarına uyan güvenli fırsat yokken işlem açmaz; bu normaldir.")
        self.log(" ".join(parts))
        self._summary_reset()

    # ------------------------------------------------------------------ manuel onay
    def _approval(self, key: str, make) -> bool | None:
        """True: emir gönderilebilir · None: onay bekleniyor · False: reddedildi / süresi doldu."""
        if self.approvals is None:
            return True
        from .approval import APPROVED, PENDING, STATE_TR
        state, req, new = self.approvals.gate(key, make)
        if new:
            self.log(f"ONAY BEKLENİYOR: {req.direction} {req.symbol} ({req.strategy}) ~{req.notional:.2f} "
                     f"{self.quote_asset}, stop {req.stop:.6g}. Uygulamadan veya Telegram'dan onaylayın "
                     f"({req.minutes_left:.0f} dk).")
            self._emit("approval_request", req)
        if state == APPROVED:
            return True
        if state == PENDING:
            return None
        self.approvals.finish(key)
        self.log(f"{req.symbol}: işlem açılmadı — {STATE_TR.get(state, state)}"
                 + (f" ({req.resolved_by})" if req.resolved_by else ""))
        self._emit("approval_resolved", req)
        return False

    def resolve_approval(self, request_id: str, approve: bool, by: str = "uygulama") -> str:
        """Arayüz veya Telegram'dan gelen onay/ret (başka iş parçacığından çağrılabilir)."""
        if self.approvals is None:
            return "Onay modu kapalı."
        ok, msg, req = self.approvals.resolve(request_id, approve, by)
        if ok and req is not None:
            self.log(f"{req.symbol}: {'ONAYLANDI' if approve else 'REDDEDİLDİ'} ({by}). "
                     + ("Bir sonraki turda güncel fiyatla kontrol edilip emir gönderilecek." if approve else ""))
            self._emit("approval_resolved", req)
        return msg

    def _approved_external_tick(self):
        if self.approvals is None:
            return
        for req in self.approvals.approved_external():
            try:
                self._open_external(*req.payload)
            except Exception as exc:  # noqa: BLE001
                self.log(f"{req.symbol}: onaylı işlem açılamadı — {exc}", logging.WARNING)
            finally:
                self.approvals.finish(req.key)

    def _batch(self) -> list[str]:
        """Bu turda incelenecek coinler: açık pozisyon/bekleyen emirler her turda, kalanı sırayla."""
        if len(self.symbols) <= self.batch_size:
            return list(self.symbols)
        must = [s for s in self.symbols if s in self.positions or s in self.pending]
        rest = [s for s in self.symbols if s not in must]
        n = max(1, self.batch_size - len(must))
        if self._batch_pos >= len(rest):
            self._batch_pos = 0
        part = rest[self._batch_pos:self._batch_pos + n]
        self._batch_pos += n
        return must + part

    def _universe_tick(self):
        if self.universe is None or not self.universe.due():
            return
        new = self.universe.refresh(keep=list(self.positions))
        if not new:
            self.log(f"Otomatik coin seçimi: {self.universe.status}", logging.WARNING)
            return
        added = [s for s in new if s not in self.symbols]
        removed = [s for s in self.symbols if s not in new]
        self.symbols = new
        if len(new) > len(self._rules) and hasattr(self.client, "all_symbol_rules"):
            try:                              # tek istekle bütün coinlerin emir kuralları
                self._rules.update(self.client.all_symbol_rules(self.quote_asset))
            except Exception as exc:  # noqa: BLE001 - kurallar yine tek tek alınabilir
                logger.info("Toplu sembol kuralları alınamadı: %s", exc)
        if added or removed:
            if len(new) > 40:
                rounds = -(-len(new) // self.batch_size)
                self.log(f"Otomatik coin seçimi: {len(new)} coin taranıyor (her turda {self.batch_size} coin, "
                         f"tüm liste ~{rounds} turda bir; açık pozisyonlar her turda). En hacimliler: "
                         + ", ".join(new[:10]))
            else:
                self.log(f"Otomatik coin seçimi ({len(new)} coin): " + ", ".join(
                    f"{s} ({self.universe.reasons.get(s, '')})" for s in new))
            self._emit("universe", {"symbols": list(new), "reasons": dict(self.universe.reasons)})

    def _maintenance_loop(self):
        ap = self.cfg.autopilot
        if self._stop.wait(ap.first_run_delay_s):
            return
        while not self._stop.is_set():
            self.run_maintenance_now()
            if self._stop.wait(ap.retrain_hours * 3600):
                return

    def run_maintenance_now(self):
        from .autopilot import run_maintenance
        self.log("Otomatik bakım başladı (gerçek Binance verisiyle istatistik, sağlık ve meta model)")
        spot = self.client if self.market == "SPOT" else None
        fut = self.futures if self.market == "SPOT" else (self.futures or self.client)
        try:
            res = run_maintenance(self.cfg, self.symbols[:max(1, self.cfg.autopilot.max_symbols)], spot, fut,
                                  strategies=self.engine.strategies)
        except Exception as exc:  # noqa: BLE001 - bakım hatası ticareti durdurmasın
            self.log(f"Otomatik bakım hatası: {exc}", logging.WARNING)
            return None
        with self._lock:
            self.metas = res.metas
            if res.stats:
                self.engine.stats = res.stats
            if res.health:
                self.engine.health = res.health
            if res.learned:
                self.engine.learned = res.learned
            self.maintenance_result = res
        for line in res.summary().splitlines():
            self.log(line)
        self._emit("maintenance", res)
        return res

    def _api_error(self, exc):
        self._api_errors.append(time.time())
        self._api_errors = [t for t in self._api_errors if time.time() - t < 300]
        self.log(f"Binance API hatası: {exc}", logging.WARNING)

    def _klines(self, client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        key = (id(client), symbol, interval)
        now = time.time()
        cached = self._cache.get(key)
        if cached is not None:
            ts, df = cached
            if not df.empty:
                next_close = pd.to_datetime(df["close_time"].iloc[-1], utc=True).timestamp() + INTERVALS[interval]
                if now < next_close + 1:
                    return df
        df = closed_only(client.klines(symbol, interval, limit=limit))
        self._cache[key] = (now, df)
        return df

    def _rules_for(self, symbol: str):
        if symbol not in self._rules:
            self._rules[symbol] = self.client.symbol_rules(symbol)
        return self._rules[symbol]

    def equity(self) -> float:
        cash = getattr(self.venue, "cash", None)
        if cash is None:
            cash = self.venue.available_balance()
            if self.market != "SPOT":
                cash += sum(p.entry_price * p.qty / max(p.leverage, 1) for p in self.positions.values())
        total = cash
        for sym, p in self.positions.items():
            price = self.last_prices.get(sym, p.entry_price)
            total += p.qty * price if self.market == "SPOT" else p.unrealized(price)[0]
        return total

    def _exposure(self) -> list[OpenExposure]:
        return [OpenExposure(s, p.direction, p.qty * self.last_prices.get(s, p.entry_price),
                             abs(p.entry_price - p.stop_loss) * p.qty) for s, p in self.positions.items()]

    def _derivatives(self, symbol: str) -> dict | None:
        if self.futures is None or (self.market == "SPOT" and not self.cfg.use_futures_context):
            return None
        key = ("deriv", symbol)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < 300:
            return cached[1]
        d = {"period": self.interval if self.interval in ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")
             else "5m"}
        try:
            d["funding"] = self.futures.funding_history(symbol, limit=200)
            d["oi"] = self.futures.open_interest_hist(symbol, d["period"], 500)
            d["long_short"] = self.futures.long_short_ratio(symbol, d["period"], 500)
            d["taker"] = self.futures.taker_volume(symbol, d["period"], 500)
        except (BinanceAPIError, ValueError) as exc:
            self._api_error(exc)
        self._cache[key] = (time.time(), d)
        return d

    # ------------------------------------------------------------------ ana döngü
    def tick(self):
        with self._lock:
            try:
                self._news_tick()
            except Exception as exc:  # noqa: BLE001 - haber hatası ticareti durdurmamalı
                self.log(f"Haber kontrolü hatası: {exc}", logging.WARNING)
            self._approved_external_tick()
            try:
                self._universe_tick()
            except Exception as exc:  # noqa: BLE001 - seçim hatası mevcut listeyle devam etsin
                self.log(f"Otomatik coin seçimi hatası: {exc}", logging.WARNING)
            try:
                self._listing_tick()
            except BinanceAPIError as exc:
                self._api_error(exc)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Listeleme kontrolü hatası: {exc}", logging.WARNING)
            for symbol in self._batch():
                if self._stop.is_set():
                    break
                try:
                    self._process(symbol)
                except BinanceAPIError as exc:
                    self._api_error(exc)
                except (DatabaseError, ExecutionError, ValueError) as exc:
                    self.log(f"{symbol}: {exc}", logging.WARNING)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"{symbol}: beklenmeyen hata - {exc}", logging.ERROR)
                    logger.exception("intel tick")
            self._update_equity()
            self._summary_maybe_log()

    def _process(self, symbol: str):
        cfg = self.cfg
        tf = cfg.timeframes
        entry = self._klines(self.client, symbol, tf.entry, self.kline_limit)
        price = None
        book = None
        try:
            book = self.client.depth(symbol, 100)
            book["_fetched_ms"] = int(time.time() * 1000)
            ob = orderbook_features(book)
            price = ob.get("mid") if ob.get("available") else None
        except BinanceAPIError as exc:
            self._api_error(exc)
        if price is None and not entry.empty:
            try:
                price = self.client.price(symbol)
            except BinanceAPIError as exc:
                self._api_error(exc)
        if price is None:
            self.log(f"{symbol}: fiyat alınamadı → yeni işlem yok", logging.WARNING)
            return
        self.last_prices[symbol] = price

        # 1) Açık pozisyon: anlık fiyat kontrolü
        pos = self.positions.get(symbol)
        if pos is not None:
            tick_row = pd.Series({"open": price, "high": price, "low": price, "close": price, "atr": np.nan})
            acts = update_on_bar(pos, tick_row, cfg.risk, count_bar=False)
            self._apply_actions(symbol, pos, acts, price)

        # 2) Yeni kapanmış mum?
        if entry.empty:
            return
        bar_time = pd.to_datetime(entry["open_time"].iloc[-1], utc=True)
        if self._last_bar.get(symbol) == bar_time:
            self._try_pending(symbol, price, book)
            return
        self._last_bar[symbol] = bar_time

        dq = check_candles(entry, tf.entry, cfg.data_quality, min_history=min(cfg.data_quality.min_history_bars,
                                                                             self.kline_limit - 10))
        dq.merge(check_orderbook(book, cfg.data_quality, int(time.time() * 1000), book.get("_fetched_ms") if book else None),
                 "orderbook: ")
        htf = {}
        for role, interval in tf.higher().items():
            try:
                htf[role] = self._klines(self.client, symbol, interval, 400)
            except BinanceAPIError as exc:
                self._api_error(exc)
        btc = None
        if cfg.btc_context and symbol != "BTCUSDT":
            try:
                btc = self._klines(self.btc_client, "BTCUSDT", tf.trend, 400)
            except BinanceAPIError as exc:
                self._api_error(exc)
        deriv = self._derivatives(symbol)
        prep = self.engine.prepare(entry, symbol, tf.entry, htf, deriv,
                                   None if btc is None or btc.empty else _features(btc, tf.trend, cfg.market))
        i = len(prep.f) - 1
        self._funding_paper(symbol)

        # 3) Açık pozisyonu yeni mumla yönet (süre, trailing, zayıflama, rejim)
        pos = self.positions.get(symbol)
        if pos is not None:
            row = prep.f.iloc[i]
            opened = pd.Timestamp(pos.opened_at)
            if opened.tzinfo is None:
                opened = opened.tz_localize("UTC")
            if pd.to_datetime(row["open_time"], utc=True) >= opened:
                acts = update_on_bar(pos, row, cfg.risk)
                self._apply_actions(symbol, pos, acts, price)
            pos = self.positions.get(symbol)
            if pos is not None:
                d_s = pos.direction.lower()
                o_s = "short" if d_s == "long" else "long"
                conf = float(prep.totals[f"score_{d_s}"].iat[i])
                opp_c = any(prep.candidates[k][o_s][i] for k in prep.candidates)
                opp = float(prep.totals[f"score_{o_s}"].iat[i]) if opp_c else None
                st = self.engine.by_key.get(pos.strategy)
                forbidden = st is not None and Regime(prep.f["regime"].iat[i]) in st.spec.forbidden_for(pos.direction)
                act = evaluate_signal_decay(pos, conf, cfg.risk, opp, forbidden, price)
                if act is not None:
                    self._apply_actions(symbol, pos, [act], price)
                elif pos.status == "WEAKENING":
                    self.log(f"{symbol}: sinyal zayıflıyor (güven {pos.initial_confidence:.0f} → {conf:.0f})")

        # 4) Karar
        ps = PortfolioState(equity=self.equity(), peak_equity=max(self._peak_equity, self.equity()),
                            day_start_equity=self._day_start_equity or self.equity(),
                            consecutive_losses=self._consecutive_losses, api_errors_recent=len(self._api_errors))
        try:
            rules = self._rules_for(symbol)
        except BinanceAPIError as exc:
            self._api_error(exc)
            rules = None
        corr = None
        if len(self.positions) and len(self.symbols) > 1:
            closes = {symbol: prep.f["close"]}
            for s2 in self.positions:
                cached = self._cache.get((id(self.client), s2, tf.entry))
                if cached is not None and not cached[1].empty:
                    closes[s2] = cached[1]["close"].reset_index(drop=True)
            corr = return_correlation(closes)
        ob = orderbook_features(book) if book else None
        live = bool(self.venue.is_live)
        if self.maintenance_result is not None:
            # Otomatik bakımdan sonra: doğrulanmış model varsa onu, yoksa deterministik mod
            self.engine.meta = self.metas.get(symbol)
        news_ctx = self.news.context(symbol) if self.news is not None else None
        d = self.engine.decide(prep, i, equity=ps.equity, open_exposure=self._exposure(), portfolio=ps,
                               rules=rules, book_stats=ob, live=live, dq=dq, has_position=symbol in self.positions,
                               available_balance=self.venue.available_balance(), correlations=corr,
                               news_ctx=news_ctx)
        if self.store is not None:
            try:
                self.store.save_decision(d, source="live" if live else "paper")
            except DatabaseError as exc:
                self.log(f"Veritabanı hatası: {exc} → yeni emir gönderilmeyecek", logging.ERROR)
                return
        self.last_decisions[symbol] = d
        self._summary_add(d)
        self._emit("decision", d)
        sig_text = {"LONG": "AL", "SHORT": "SAT"}.get(d.direction, "BEKLE") if d.is_trade else "BEKLE"
        self.last_signals[symbol] = {
            "signal": sig_text, "time": bar_time.strftime("%Y-%m-%d %H:%M"), "price": price,
            "reason": (f"{d.direction} güven {d.confidence:.0f}, {d.market_regime}, {d.strategy}" if d.is_trade else
                       f"NO TRADE: {', '.join(d.no_trade_reasons) or d.signal_status} ({d.market_regime})"),
        }
        self._emit("signal", {"symbol": symbol, **self.last_signals[symbol]})
        if d.is_trade:
            self.log(f"{symbol}: {tr(d.direction)} sinyali (güven {d.confidence:.0f}, {d.strategy}, R:R {d.risk_reward})")
            self.pending[symbol] = d
            self._try_pending(symbol, price, book)

    def _funding_paper(self, symbol: str):
        """Kağıt futures: 00/08/16 UTC funding kesimlerinde son funding oranıyla ödeme/tahsilat."""
        pos = self.positions.get(symbol)
        if pos is None or self.market == "SPOT" or self.venue.is_live or self.futures is None:
            return
        now = pd.Timestamp.now(tz="UTC")
        last = getattr(pos, "_last_funding", None) or pd.Timestamp(pos.opened_at)
        last = last if last.tzinfo else last.tz_localize("UTC")
        cuts = [now.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=h) for h in (0, 8, 16, 24, 32, 40)]
        crossed = [c for c in cuts if last < c <= now]
        if not crossed:
            return
        try:
            rate = float(self.futures.premium_index(symbol)["lastFundingRate"])
        except (BinanceAPIError, KeyError) as exc:
            self._api_error(exc)
            return
        amount = len(crossed) * rate * pos.qty * self.last_prices.get(symbol, pos.entry_price) * pos.sign
        pos.funding_paid += amount
        self.venue.pay_funding(amount)
        object.__setattr__(pos, "_last_funding", now)
        self.log(f"{symbol}: funding {'ödendi' if amount > 0 else 'alındı'} {abs(amount):.4f} {self.quote_asset} "
                 f"(oran %{rate * 100:.4f})")

    # ------------------------------------------------------------------ haberler ve listelemeler
    def _news_tick(self):
        if self.news is None or not self.cfg.news.enabled:
            return
        if time.time() - self.news.last_poll < self.cfg.news.poll_seconds:
            return
        symbols = None
        try:
            symbols = self.client.exchange_info().get("symbols", [])
        except BinanceAPIError as exc:
            self._api_error(exc)
        if symbols:
            self._tradable_bases = {x.get("baseAsset") for x in symbols
                                    if x.get("quoteAsset") == self.quote_asset and x.get("status") == "TRADING"}
        new_items, events = self.news.poll(symbols)
        self._emit("news_polled", {"new": len(new_items), "events": len(events),
                                   "status": dict(self.news.source_status)})
        watched = {s[:-len(self.quote_asset)] for s in set(self.symbols) | set(self.positions)}
        for it in new_items:
            relevant = bool(set(it.symbols) & watched)
            if it.severity >= self.cfg.news.notify_min_severity or relevant:
                self._emit("news", it)
        for ev in events:
            self._emit("listing", ev)
            self.log(f"Listeleme olayı: {ev.symbol} {LISTING_KIND_TR.get(ev.kind, ev.kind)} ({ev.source})")
            if ev.kind in ("NEW_SYMBOL", "NOW_TRADING") and ev.status == "TRADING":
                self.listing_watch.setdefault(ev.symbol, time.time())
        if self.cfg.news.delist_exit:
            from .position_manager import ExitAction
            for sym, pos in list(self.positions.items()):
                ctx = self.news.context(sym)
                if ctx.force_exit:
                    self.log(f"{sym}: DELIST duyurusu → pozisyon kapatılıyor", logging.WARNING)
                    price = self.last_prices.get(sym) or self.client.price(sym)
                    self._apply_actions(sym, pos, [ExitAction(ExitReason.EMERGENCY_EXIT, pos.qty, price, True,
                                                              "Delist duyurusu")], price)

    EXTERNAL = ("new_listing", "news_catalyst")

    def _external_limits(self, strategy: str):
        if strategy == "news_catalyst":
            c = self.cfg.catalyst
            return c.max_hold_hours * 60, c.tp_levels_r, c.tp_fractions
        lc = self.cfg.listing
        return lc.max_hold_minutes, lc.tp_levels_r, lc.tp_fractions

    def _manage_external(self):
        """Ana sembol listesi dışındaki (listeleme / haber katalizörü) pozisyonları yönetir."""
        from .position_manager import ExitAction
        for sym, pos in list(self.positions.items()):
            if pos.strategy not in self.EXTERNAL or sym in self.symbols:
                continue
            max_min, levels, fracs = self._external_limits(pos.strategy)
            price = self.client.price(sym)
            self.last_prices[sym] = price
            acts = update_on_bar(pos, pd.Series({"open": price, "high": price, "low": price, "close": price,
                                                 "atr": np.nan}), self._listing_risk_cfg(levels, fracs),
                                 count_bar=False)
            self._apply_actions(sym, pos, acts, price)
            pos = self.positions.get(sym)
            if pos is not None:
                opened = pd.Timestamp(pos.opened_at)
                opened = opened if opened.tzinfo else opened.tz_localize("UTC")
                if (pd.Timestamp.now(tz="UTC") - opened).total_seconds() / 60 >= max_min:
                    self._apply_actions(sym, pos, [ExitAction(ExitReason.TIME_EXPIRY, pos.qty, price, True,
                                                              f"Maksimum {max_min:.0f} dk")], price)

    def _listing_tick(self):
        self._manage_external()
        try:
            self._insight_tick()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Öngörü taraması hatası: {exc}", logging.WARNING)
        lc = self.cfg.listing
        if not lc.enabled:
            return
        now = time.time()
        for sym, since in list(self.listing_watch.items()):
            if now - since > lc.watch_hours * 3600:
                self.listing_watch.pop(sym, None)
                continue
            if sym in self.positions:
                continue
            self._evaluate_listing(sym)

    def _listing_risk_cfg(self, levels=None, fracs=None):
        import copy
        rc = copy.deepcopy(self.cfg.risk)
        rc.tp_levels_r = list(levels or self.cfg.listing.tp_levels_r)
        rc.tp_fractions = list(fracs or self.cfg.listing.tp_fractions)
        rc.trailing_method = "percent"
        return rc

    def _insight_tick(self):
        cc = self.cfg.catalyst
        if self.insights is None or self.news is None or not cc.enabled:
            return
        if time.time() - self._last_insight_scan < cc.scan_minutes * 60:
            return
        self._last_insight_scan = time.time()
        items = self.news.store.recent(cc.lookback_hours)
        bases = self._tradable_bases or {s[:-len(self.quote_asset)] for s in self.symbols}
        insights = self.insights.scan(items, bases)
        self.last_insights = insights
        if self.insight_store is not None:
            try:
                self.insight_store.update_outcomes(self.client)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Öngörü sonuçları güncellenemedi: {exc}", logging.WARNING)
        for ins in insights:
            fresh = self.insight_store.save(ins) if self.insight_store is not None else True
            if ins.signal in ("AL", "İZLE") and fresh:
                self._emit("insight", ins)
                self.log(f"Öngörü {ins.symbol}: {ins.signal} (potansiyel {ins.potential:.0f}, katalizör "
                         f"{ins.catalyst.score:.0f})")
            if ins.signal != "AL" or not cc.trade_enabled or ins.symbol in self.positions:
                continue
            if (not cc.any_binance_pair and self.cfg.allowed_symbols and ins.symbol not in self.cfg.allowed_symbols
                    and ins.symbol not in self.symbols):
                self.log(f"{ins.symbol}: AL öngörüsü var ama izinli sembol listesinde değil (işlem yok)")
                continue
            self._open_external(ins.symbol, ins.entry, ins.stop, ins.targets, cc.tp_fractions, "news_catalyst",
                                ins.explain(), ins.reasons, cc.risk_multiplier, cc.stage, cc.max_concurrent,
                                min(95.0, ins.potential), "NEWS")

    def _open_external(self, symbol, entry, stop, targets, fractions, strategy, explanation, reasons,
                       risk_mult, stage, max_concurrent, confidence, regime_label) -> bool:
        """Harici sinyal (listeleme / haber katalizörü) için ortak, risk kontrollü emir yolu."""
        live = bool(self.venue.is_live)
        from .types import LIVE_STAGES, LifecycleStage
        if live and LifecycleStage(stage) not in LIVE_STAGES:
            self.log(f"{symbol}: {strategy} sinyali var ama aşama {tr(stage)} (canlı işlem izni yok)")
            return False
        if sum(1 for p in self.positions.values() if p.strategy == strategy) >= max_concurrent:
            return False
        if len(self.positions) >= self.cfg.risk.max_open_positions:
            self.log(f"{symbol}: maksimum açık pozisyon dolu, {strategy} sinyali atlandı")
            return False
        if self.news is not None and self.news.context(symbol).block_long:
            self.log(f"{symbol}: {strategy} sinyali olumsuz haber nedeniyle reddedildi")
            return False
        from .risk_engine import position_size
        try:
            rules = self._rules_for(symbol)
            book = self.client.depth(symbol, 100)
        except BinanceAPIError as exc:
            self._api_error(exc)
            return False
        ob = orderbook_features(book) if book else {"available": False}
        price = ob.get("mid", entry)
        if (price - stop) <= 0:
            self.log(f"{symbol}: fiyat stop'un altında, {strategy} sinyali iptal")
            return False
        sz = position_size(self.equity(), price, stop, self.cfg.risk, confidence, risk_mult, True, self.market,
                           self.venue.available_balance(), self._exposure(),
                           min_notional=float(getattr(rules, "min_notional", 0) or 0))
        if not sz.ok:
            self.log(f"{symbol}: {strategy} boyutu hesaplanamadı ({'; '.join(sz.reasons)})")
            return False
        chk = pretrade_check(rules, sz.qty, price, "LONG", self.cfg, book, self.venue.available_balance(),
                             self.market, 1)
        if not chk.ok:
            self.log(f"{symbol}: {strategy} emir ön kontrolü başarısız → {'; '.join(chk.errors)}", logging.WARNING)
            return False
        if symbol in self.positions:
            return False
        args = (symbol, entry, stop, targets, fractions, strategy, explanation, reasons, risk_mult, stage,
                max_concurrent, confidence, regime_label)
        ok = self._approval(f"{strategy}:{symbol}", lambda: dict(
            symbol=symbol, direction="LONG", strategy=strategy, price=price, qty=float(chk.qty),
            notional=float(chk.qty) * price, stop=stop, targets=list(targets), confidence=confidence,
            reason="; ".join(str(r) for r in list(reasons)[:2]), live=live, payload=args))
        if not ok:
            return False
        d = DecisionObject(signal_id=uuid.uuid4().hex[:16], symbol=symbol, market=self.market,
                           timeframe="1m" if strategy == "new_listing" else "1h", direction="LONG",
                           signal_status=SignalStatus.CONFIRMED.value, strategy=strategy, confidence=confidence,
                           market_regime=regime_label, entry=entry, stop_loss=stop, take_profit=targets[-1],
                           take_profit_levels=list(targets), position_size=float(chk.qty),
                           notional=float(chk.qty) * price, reasons=list(reasons), explanation=explanation,
                           created_at=datetime.now(timezone.utc).isoformat())
        if self.store is not None:
            try:
                self.store.save_decision(d, source="live" if live else "paper")
            except DatabaseError as exc:
                self.log(f"Veritabanı hatası: {exc} → emir gönderilmedi", logging.ERROR)
                return False
        try:
            fill = self.venue.open(symbol, "LONG", float(chk.qty), price, rules, 1, stop)
        except (ExecutionError, BinanceAPIError) as exc:
            self.log(f"{symbol}: {strategy} emri başarısız — {exc}", logging.ERROR)
            self._status(d, SignalStatus.CANCELLED)
            return False
        pos = open_position(symbol, self.market, "LONG", fill.price, fill.qty, stop,
                            [fill.price + (t - entry) for t in targets], fractions, strategy,
                            datetime.now(timezone.utc).isoformat(), 0, 0, 0, confidence, regime_label, d.signal_id)
        pos.fees_paid = fill.fee
        self.positions[symbol] = pos
        self._status(d, SignalStatus.EXECUTED)
        label = "YENİ LİSTELEME ALIMI" if strategy == "new_listing" else "HABER KATALİZÖRÜ ALIMI"
        self.log(f"{label} {symbol}: {fill.qty:.8g} @ {fill.price:.8g} | SL {stop:.8g}")
        self._save_state()
        self._emit("positions", None)
        self._emit("opened", pos)
        return True

    def _evaluate_listing(self, symbol: str):
        from .listing import evaluate_listing
        lc = self.cfg.listing
        df = closed_only(self.client.klines(symbol, "1m", limit=1000, start_time=0))
        if df is None or df.empty:
            return
        age_h = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(df["open_time"].iloc[0], utc=True)).total_seconds() / 3600
        if age_h > lc.watch_hours:
            self.listing_watch.pop(symbol, None)
            return
        book = None
        try:
            book = self.client.depth(symbol, 100)
        except BinanceAPIError as exc:
            self._api_error(exc)
        ob = orderbook_features(book) if book else {"available": False}
        sig = evaluate_listing(df, symbol, lc, book_stats=ob)
        self.listing_signals[symbol] = sig
        self._emit("listing_signal", sig)
        if not sig.ok:
            return
        if self._open_external(symbol, sig.entry, sig.stop, sig.targets, lc.tp_fractions, "new_listing",
                               sig.explain(), sig.reasons, lc.risk_multiplier, lc.stage, lc.max_concurrent, 75.0,
                               "LISTING"):
            self.listing_watch.pop(symbol, None)

    # ------------------------------------------------------------------ giriş
    def _try_pending(self, symbol: str, price: float, book: dict | None):
        d = self.pending.get(symbol)
        if d is None or symbol in self.positions:
            return
        now = datetime.now(timezone.utc)
        if d.signal_expiry and now > pd.Timestamp(d.signal_expiry).to_pydatetime():
            self.pending.pop(symbol, None)
            if self.approvals is not None:
                self.approvals.finish(d.signal_id)
            self.log(f"{symbol}: sinyal süresi doldu")
            self._status(d, SignalStatus.EXPIRED)
            return
        s = 1 if d.direction == "LONG" else -1
        if (price - d.stop_loss) * s <= 0:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            self.log(f"{symbol}: fiyat stop'un ötesinde, sinyal iptal")
            return
        if self.store is not None and not self.store.healthy():
            self.log("Veritabanı erişilemiyor → yeni emir yok", logging.ERROR)
            return
        if len(self._api_errors) >= self.cfg.circuit_breaker.api_error_limit:
            self.log("API kararsız (devre kesici) → yeni emir yok", logging.WARNING)
            return
        rules = self._rules.get(symbol)
        risk_amount = abs(d.entry - d.stop_loss) * d.position_size
        qty = min(risk_amount / abs(price - d.stop_loss), d.notional / price * 1.05)
        chk = pretrade_check(rules, qty, price, d.direction, self.cfg, book, self.venue.available_balance(),
                             self.market, d.leverage)
        if not chk.ok:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            self.log(f"{symbol}: emir ön kontrolü başarısız → {'; '.join(chk.errors)}", logging.WARNING)
            return
        ok = self._approval(d.signal_id, lambda: dict(
            symbol=symbol, direction=d.direction, strategy=d.strategy, price=price, qty=float(chk.qty),
            notional=float(chk.qty) * price, stop=d.stop_loss, targets=list(d.take_profit_levels),
            confidence=d.confidence, reason="; ".join(str(r) for r in d.reasons[:2]), live=bool(self.venue.is_live)))
        if ok is None:
            return                            # onay bekleniyor; sinyal bekleyen listede kalır
        if ok is False:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            return
        if self.approvals is not None:
            self.approvals.finish(d.signal_id)
        try:
            fill = self.venue.open(symbol, d.direction, float(chk.qty), price, rules, d.leverage, d.stop_loss)
        except (ExecutionError, BinanceAPIError) as exc:
            self.pending.pop(symbol, None)
            self._status(d, SignalStatus.CANCELLED)
            self.log(f"{symbol}: emir hatası — {exc}", logging.ERROR)
            return
        self.pending.pop(symbol, None)
        pos = open_position(symbol, self.market, d.direction, fill.price, fill.qty, d.stop_loss, d.take_profit_levels,
                            self.cfg.risk.tp_fractions[:len(d.take_profit_levels)], d.strategy,
                            datetime.now(timezone.utc).isoformat(), 0, d.max_hold_bars, d.expected_hold_bars,
                            d.confidence, d.market_regime, d.signal_id, d.leverage, d.liquidation_price)
        pos.fees_paid = fill.fee
        self.positions[symbol] = pos
        self._status(d, SignalStatus.EXECUTED)
        self.log(f"{'ALIM' if d.direction == 'LONG' else 'SHORT'} {symbol}: {fill.qty:.8g} @ {fill.price:.8g} | "
                 f"SL {d.stop_loss:.8g} | TP {', '.join(f'{t:.6g}' for t in d.take_profit_levels)}")
        self._save_state()
        self._emit("positions", None)
        self._emit("opened", pos)

    def _status(self, d: DecisionObject, status: SignalStatus):
        if self.store is not None:
            try:
                self.store.update_status(d.signal_id, status.value)
            except DatabaseError as exc:
                self.log(str(exc), logging.ERROR)

    # ------------------------------------------------------------------ çıkış
    def _apply_actions(self, symbol: str, pos: ManagedPosition, actions, price: float):
        prev_stop = pos.stop_loss
        for a in actions:
            try:
                fill = self.venue.close(symbol, pos.direction, a.qty, a.price if not self.venue.is_live else price,
                                        pos.entry_price, self._rules.get(symbol), full=a.full)
            except (ExecutionError, BinanceAPIError) as exc:
                # Emir gerçekleşmedi: miktarı geri ekle. Tam çıkışlar (SL/süre vb.) sonraki döngüde yeniden tetiklenir;
                # kısmi TP reddedilirse (ör. minimum tutar altı) pozisyon sonraki hedef/stop ile yönetilmeye devam eder.
                pos.qty += a.qty
                self.log(f"{symbol}: çıkış emri başarısız ({tr_reason(a.reason)}) — {exc}. Pozisyon korunuyor, tekrar denenecek.",
                         logging.ERROR)
                return
            gross = (fill.price - pos.entry_price) * pos.sign * fill.qty
            pos.realized_pnl += gross
            pos.fees_paid += fill.fee
            pos.confidence_history.append(f"{a.reason}:{a.note}")
            self.log(f"{symbol}: {tr_reason(a.reason)} {a.note} — {fill.qty:.8g} @ {fill.price:.8g} (brüt {gross:+.2f})")
            if a.full or pos.qty <= 1e-12:
                self._finalize(symbol, pos, a.reason)
                return
        if self.positions.get(symbol) is pos and pos.stop_loss != prev_stop:
            try:
                self.venue.update_stop(symbol, pos.direction, pos.stop_loss, self._rules.get(symbol))
            except (ExecutionError, BinanceAPIError) as exc:
                self.log(f"{symbol}: koruyucu stop güncellenemedi — {exc}", logging.WARNING)
            self._save_state()
            self._emit("positions", None)

    def _finalize(self, symbol: str, pos: ManagedPosition, reason):
        net = pos.realized_pnl - pos.fees_paid - pos.funding_paid
        base = pos.entry_price * pos.initial_qty / (pos.leverage if self.market != "SPOT" else 1)
        risk = pos.risk_per_unit * pos.initial_qty
        opened = pd.Timestamp(pos.opened_at)
        hold_min = (pd.Timestamp.now(tz="UTC") - (opened if opened.tzinfo else opened.tz_localize("UTC"))) \
            .total_seconds() / 60
        trade = ClosedTrade(
            symbol=symbol, qty=pos.initial_qty, entry_price=pos.entry_price,
            exit_price=pos.entry_price + pos.realized_pnl / (pos.initial_qty * pos.sign) if pos.initial_qty else 0.0,
            cost=base, proceeds=base + net, pnl=net, pnl_pct=net / base * 100 if base else 0.0,
            opened_at=pos.opened_at[:19].replace("T", " "), closed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=str(reason), strategy=pos.strategy, side=pos.direction, exit_reason=str(reason),
            r_multiple=net / risk if risk else 0.0, gross_pnl=pos.realized_pnl, fees=pos.fees_paid,
            signal_id=pos.signal_id)
        self.trades.append(trade)
        self._consecutive_losses = self._consecutive_losses + 1 if net <= 0 else 0
        self.positions.pop(symbol, None)
        if self.store is not None and pos.signal_id:
            try:
                self.store.record_outcome(pos.signal_id, "WIN" if net > 0 else "LOSS", str(reason), net,
                                          pos.realized_pnl, pos.fees_paid, trade.r_multiple, hold_min)
            except DatabaseError as exc:
                self.log(str(exc), logging.ERROR)
        self.log(f"KAPANDI {symbol} ({tr_reason(reason)}): net {net:+.2f} {self.quote_asset}, R {trade.r_multiple:+.2f}")
        self._save_state()
        self._emit("positions", None)
        self._emit("trade", trade)

    def close_position(self, symbol: str, reason: str = "MANUAL_EXIT"):
        with self._lock:
            pos = self.positions.get(symbol)
            if pos is None:
                raise ValueError(f"{symbol} için açık pozisyon yok.")
            price = self.client.price(symbol)
            self.last_prices[symbol] = price
            from .position_manager import ExitAction
            self._apply_actions(symbol, pos, [ExitAction(ExitReason.MANUAL_EXIT, pos.qty, price, True, "Elle")], price)

    def forget_position(self, symbol: str):
        with self._lock:
            if self.positions.pop(symbol, None) is not None:
                self.log(f"{symbol} pozisyonu takipten çıkarıldı (emir gönderilmedi).")
                self._save_state()
                self._emit("positions", None)

    def _update_equity(self):
        eq = self.equity()
        today = datetime.now(timezone.utc).date()
        if self._day != today:
            self._day, self._day_start_equity = today, eq
            self.halted_today = False
        self._peak_equity = max(self._peak_equity, eq)
        day_pct = (eq / self._day_start_equity - 1) * 100 if self._day_start_equity else 0.0
        if not self.halted_today and day_pct <= -self.cfg.risk.max_daily_loss_pct:
            self.halted_today = True
            self._emit("halt", f"Günlük değişim %{day_pct:.2f}")
        self._emit("equity", {"equity": eq, "quote": self.venue.available_balance(), "day_pct": day_pct})

    # ------------------------------------------------------------------ durum
    def _save_state(self):
        if not self.state_path:
            return
        data = {"positions": [p.to_dict() for p in self.positions.values()],
                "trades": [t.to_dict() for t in self.trades[-500:]]}
        if not self.venue.is_live:
            data["paper_cash"] = self.venue.cash
            data["margin_used"] = getattr(self.venue, "margin_used", {})
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            logger.error("Durum kaydedilemedi: %s", exc)

    def _load_state(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for p in data.get("positions", []):
            pos = ManagedPosition.from_dict(p)
            if (pos.symbol in self.symbols or pos.strategy in self.EXTERNAL) and pos.market == self.market:
                self.positions[pos.symbol] = pos
        self.trades = [ClosedTrade.from_dict(t) for t in data.get("trades", [])]
        if not self.venue.is_live and "paper_cash" in data:
            self.venue.cash = float(data["paper_cash"])
            if hasattr(self.venue, "margin_used"):
                self.venue.margin_used = {k: float(v) for k, v in data.get("margin_used", {}).items()}


def _features(df: pd.DataFrame, interval: str, market: str) -> pd.DataFrame:
    from .features import compute_features
    return compute_features(df, interval, market)
