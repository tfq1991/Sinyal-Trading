# =====================================
# ===  MrT Scalper Combo + Swift Ultra v2.1  ===
# ===  OKX + Binance fallback + Debug  ===
# ===  VWAP Div + RSI Opt + ADX/BB/Stoch/Candle/Hybrid + Swift Ultra ===
# =====================================

import logging
import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timezone

# -------------------------
# CONFIG
# -------------------------
TIMEFRAMES = ["5m", "15m", "1h"]  # added 5m
BASE_SIGNAL_TFS = ["5m", "15m", "1h"]

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "TRXUSDT", "OKBUSDT", "LTCUSDT", "SHIBUSDT", "UNIUSDT",
    "ATOMUSDT", "BCHUSDT", "FILUSDT", "ETCUSDT", "APTUSDT",
    "NEARUSDT", "ARBUSDT", "OPUSDT", "ICPUSDT", "TONUSDT",
    "AAVEUSDT", "SANDUSDT", "THETAUSDT", "EGLDUSDT", "FLOWUSDT",
    "COREUSDT", "XTZUSDT", "MANAUSDT", "GALAUSDT", "PEPEUSDT",
    "SNXUSDT", "CRVUSDT", "INJUSDT", "XLMUSDT", "CFXUSDT",
    "CHZUSDT", "NEOUSDT", "COMPUSDT", "IMXUSDT", "ZROUSDT",
    "WLDUSDT", "SUIUSDT", "PYTHUSDT"
]

TP1_PERCENT = 0.05
TP2_PERCENT = 0.10
SL_PERCENT = 0.05
SCAN_INTERVAL = 10  # minutes (used by external scheduler like GitHub Actions)

APPLY_HYBRID_FILTER = False
HYBRID_THRESHOLD = 4

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

LAST_SIGNALS_FILE = "last_signals.json"
SIGNALS_CSV = "signals_log.csv"
last_signals = {}

# -------------------------
# LOGGING SETUP
# -------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("bot_debug.log"),
        logging.StreamHandler()
    ]
)
logging.info(f"🔧 Logging level set to {LOG_LEVEL}")

# -------------------------
# TELEGRAM + UTILITIES
# -------------------------
def send_message(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.debug("Telegram not configured; message suppressed.")
        print(msg)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Telegram API {resp.status_code}: {resp.text}")
    except Exception as e:
        logging.error(f"[ERROR] Telegram failed: {e}")

def load_last_signals():
    global last_signals
    try:
        if os.path.exists(LAST_SIGNALS_FILE):
            with open(LAST_SIGNALS_FILE, "r") as f:
                last_signals.update(json.load(f))
                logging.info(f"Loaded last_signals: {len(last_signals)}")
    except Exception as e:
        logging.error(f"Failed loading last_signals: {e}")

def save_last_signals():
    try:
        with open(LAST_SIGNALS_FILE, "w") as f:
            json.dump(last_signals, f, indent=2)
    except Exception as e:
        logging.error(f"Failed saving last_signals: {e}")

def log_signal_csv(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(SIGNALS_CSV):
        df.to_csv(SIGNALS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(SIGNALS_CSV, index=False)

# -------------------------
# GET KLINES (OKX primary, Binance fallback)
# -------------------------
def get_klines(symbol, interval="5m", limit=500):
    """
    Returns DataFrame with columns:
    open_time, open, high, low, close, volume, volCcy, volCcyQuote, confirm, close_time
    OKX: uses instId and bar param
    Binance fallback: /api/v3/klines returns arrays per kline
    """
    tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W"
    }
    interval_param = tf_map.get(interval, interval)
    inst = symbol.replace("USDT", "-USDT") if "-" not in symbol else symbol

    okx_endpoints = [
        "https://www.okx.com",
        "https://www.okx.cab",
        "https://www.okx.co",
        "https://aws.okx.com",
    ]

    # Binance fallback
    binance_base = "https://api.binance.com"

    # Try OKX endpoints first
    for base_url in okx_endpoints:
        try:
            url = f"{base_url}/api/v5/market/candles"
            params = {"instId": inst, "bar": interval_param, "limit": limit}
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                raw = data["data"][::-1]
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "volCcy", "volCcyQuote", "confirm"
                ])
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = df[c].astype(float)
                df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms")
                df["close_time"] = df["open_time"]
                return df
        except Exception as e:
            logging.debug(f"get_klines OKX error {base_url} {inst}: {e}")

    # Fallback: Binance klines (api/v3/klines)
    try:
        path = "/api/v3/klines"
        params = {"symbol": symbol, "interval": interval_param, "limit": limit}
        r = requests.get(f"{binance_base}{path}", params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Binance returns list of lists:
            # [ Open time, Open, High, Low, Close, Volume, Close time, ... ]
            raw = data[::-1]
            df = pd.DataFrame(raw)
            if df.shape[1] >= 6:
                df = df.iloc[:, :7]  # take up to close time
                df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = df[c].astype(float)
                # Binance times are in ms for open_time and close_time
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                return df
    except Exception as e:
        logging.debug(f"get_klines Binance error {symbol}: {e}")

    logging.error(f"All endpoints failed for {symbol}")
    return pd.DataFrame()

# -------------------------
# Helper: Engulfing candle detector
# -------------------------
def is_bullish_engulfing(df):
    if df.shape[0] < 2:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_body = abs(prev['close'] - prev['open'])
    last_body = abs(last['close'] - last['open'])
    return (prev['close'] < prev['open']) and (last['close'] > last['open']) and (last['close'] > prev['open']) and (last['open'] < prev['close']) and (last_body > prev_body)

def is_bearish_engulfing(df):
    if df.shape[0] < 2:
        return False
    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev_body = abs(prev['close'] - prev['open'])
    last_body = abs(last['close'] - last['open'])
    return (prev['close'] > prev['open']) and (last['close'] < last['open']) and (last['open'] > prev['close']) and (last['close'] < prev['open']) and (last_body > prev_body)

# -------------------------
# INDICATORS & SIGNAL DETECTORS (5m, 15m, 1h)
# -------------------------
def detect_signal_generic(df, tf_label):
    """Generic trend+momentum detector used for 1h and 15m (lagging confirm)"""
    if df.shape[0] < 50:
        return None
    try:
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
        df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    except Exception as e:
        logging.error(f"{tf_label} Indicator calc error: {e}")
        return None

    last = df.iloc[-1]
    reasons = []
    score = 0.0
    if last["ema50"] > last["ema200"]:
        reasons.append("EMA50>200")
        score += 1.5
    else:
        reasons.append("EMA50<200")
        score += -0.5

    if last["macd"] > last["macd_signal"]:
        reasons.append("MACD>Signal")
        score += 1.5
    else:
        score += -0.5

    if last["rsi14"] > 55:
        reasons.append("RSI14>55")
        score += 1.0
    elif last["rsi14"] < 45:
        reasons.append("RSI14<45")
        score += -1.0

    if last["ema50"] > last["ema200"] and last["macd"] > last["macd_signal"] and last["rsi14"] > 55:
        return "BUY", f"Trend Confirm ({tf_label})", "Strong", {"score": score, "reasons": reasons}
    elif last["ema50"] < last["ema200"] and last["macd"] < last["macd_signal"] and last["rsi14"] < 45:
        return "SELL", f"Trend Confirm ({tf_label})", "Strong", {"score": score, "reasons": reasons}
    return None

def detect_signal_15m(df):
    return detect_signal_generic(df, "15m")

def detect_signal_1h(df):
    return detect_signal_generic(df, "1h")

def detect_signal_5m(df):
    """Faster Swift Ultra detection for 5m using EMA5/13 + RSI7 + VWAP/OBV"""
    if df.shape[0] < 50:
        return None
    try:
        df["ema5"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
        df["ema13"] = ta.trend.EMAIndicator(df["close"], window=13).ema_indicator()
        df["rsi7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
        ).volume_weighted_average_price()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["obv_ma"] = df["obv"].rolling(20).mean()
    except Exception as e:
        logging.error(f"5m Indicator error: {e}")
        return None

    last = df.iloc[-1]
    reasons = []
    score = 0.0
    base_buy = (last["ema5"] > last["ema13"]) and (last["rsi7"] > 50)
    base_sell = (last["ema5"] < last["ema13"]) and (last["rsi7"] < 50)

    if last.get("vwap", np.nan) and last["close"] > last["vwap"]:
        reasons.append("Close>VWAP")
        score += 0.8
    if last.get("obv_ma", np.nan) and last["obv"] > last["obv_ma"]:
        reasons.append("OBV>MA")
        score += 0.8

    if base_buy and last["macd"] > last["macd_signal"]:
        reasons.append("EMA5>13, RSI7>50, MACD>Signal")
        score += 2.0
        return "BUY", "Swift Ultra (5m)", "Fast", {"score": score, "reasons": reasons}
    elif base_sell and last["macd"] < last["macd_signal"]:
        reasons.append("EMA5<13, RSI7<50, MACD<Signal")
        score += 2.0
        return "SELL", "Swift Ultra (5m)", "Fast", {"score": score, "reasons": reasons}
    return None

# -------------------------
# Multi-timeframe confirmation
# -------------------------
def confirm_multi(signals):
    """
    signals: dict with tf->(signal, mode, strength, meta) or None
    Return combined signal if >=2 TF agree. Merge meta & sum scores.
    """
    votes = {}
    metas = []
    total_score = 0.0
    reasons = []
    modes = []

    for tf, res in signals.items():
        if res:
            s_signal, s_mode, s_strength, s_meta = res
            votes.setdefault(s_signal, []).append(tf)
            metas.append(s_meta)
            modes.append(s_mode)
            total_score += s_meta.get("score", 0.0)
            reasons += s_meta.get("reasons", [])

    # Find majority
    for sig, tfs in votes.items():
        if len(tfs) >= 2:  # at least 2 TF agree
            mode_str = "+".join(sorted(modes))
            merged_meta = {"score": total_score, "reasons": list(dict.fromkeys(reasons))}
            return sig, f"Strong Confirmed ({','.join(tfs)})", mode_str, merged_meta
    return None

# -------------------------
# ATR-based SL/TP (aggressive)
# -------------------------
def compute_atr_based_levels(df, entry, signal):
    """
    Aggressive multipliers:
    SL: tight => 0.8 * ATR
    TP1: 1.8 * ATR
    TP2: 3.6 * ATR
    If ATR fails, fallback to percent-based
    """
    try:
        atr_series = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        atr = float(atr_series.iloc[-1])
        if np.isnan(atr) or atr <= 0:
            raise ValueError("bad atr")
        if signal == "BUY":
            sl = entry - 0.8 * atr
            tp1 = entry + 1.8 * atr
            tp2 = entry + 3.6 * atr
        else:
            sl = entry + 0.8 * atr
            tp1 = entry - 1.8 * atr
            tp2 = entry - 3.6 * atr

        # Ensure SL is not on wrong side of price (safety)
        if signal == "BUY" and sl >= entry:
            sl = entry * (1 - SL_PERCENT)
        if signal == "SELL" and sl <= entry:
            sl = entry * (1 + SL_PERCENT)

        return tp1, tp2, sl, atr
    except Exception as e:
        # fallback to percent based
        if signal == "BUY":
            tp1 = entry * (1 + TP1_PERCENT)
            tp2 = entry * (1 + TP2_PERCENT)
            sl = entry * (1 - SL_PERCENT)
        else:
            tp1 = entry * (1 - TP1_PERCENT)
            tp2 = entry * (1 - TP2_PERCENT)
            sl = entry * (1 + SL_PERCENT)
        return tp1, tp2, sl, None

# -------------------------
# SCANNER
# -------------------------
def scan_once():
    total_signals = 0
    summary = {"buy": 0, "sell": 0}

    for symbol in SYMBOLS:
        try:
            # Fetch klines for needed TFs (limit enough for indicators)
            df5 = get_klines(symbol, "5m", limit=200)
            df15 = get_klines(symbol, "15m", limit=200)
            df1h = get_klines(symbol, "1h", limit=300)

            if df5.empty or df15.empty or df1h.empty:
                logging.debug(f"{symbol}: ❌ Data kosong pada salah satu timeframe")
                continue

            # Volume filter (use 5m volume for entry aggressiveness)
            last5 = df5.iloc[-1]
            vol_mean_30 = df5["volume"].rolling(30).mean().iloc[-1]
            if vol_mean_30 and last5["volume"] < 0.7 * vol_mean_30:
                logging.debug(f"{symbol}: skip karena volume rendah (5m)")
                continue

            # Detect signals on each timeframe
            res5 = detect_signal_5m(df5)
            res15 = detect_signal_15m(df15)
            res1h = detect_signal_1h(df1h)

            signals = {"5m": res5, "15m": res15, "1h": res1h}
            result = confirm_multi(signals)
            if not result:
                continue

            signal, strength_label, mode_str, meta = result
            score = meta.get("score", 0.0)

            # Check engulfing patterns to boost confidence
            eng5_buy = is_bullish_engulfing(df5)
            eng5_sell = is_bearish_engulfing(df5)
            eng15_buy = is_bullish_engulfing(df15)
            eng15_sell = is_bearish_engulfing(df15)

            if signal == "BUY" and (eng5_buy or eng15_buy):
                score += 1.5
                strength_label += " + Engulfing Boost 🔥"
                meta.setdefault("reasons", []).append("Engulfing")
            if signal == "SELL" and (eng5_sell or eng15_sell):
                score += 1.5
                strength_label += " + Engulfing Boost 🔥"
                meta.setdefault("reasons", []).append("Engulfing")

            # Swift extra confirmation (reuse 5m logic)
            swift_res = res5
            if swift_res and swift_res[0] == signal:
                score += 1.0
                strength_label += " + Swift Confirmed 🚀"
            elif swift_res and swift_res[0] != signal:
                strength_label += " ⚠️ Swift Divergence"

            if last_signals.get(symbol) == signal:
                continue

            # Entry price from last 5m close for more aggressive entry
            entry1 = float(last5["close"])
            # small buffer entry2 slightly more conservative
            entry2 = entry1 * (0.997 if signal == "BUY" else 1.003)

            # ATR-based dynamic levels (aggressive multipliers)
            tp1, tp2, sl, atr = compute_atr_based_levels(df5, entry1, signal)

            emoji = "🟢" if signal == "BUY" else "🔴"
            title = "BUY Signal Detected" if signal == "BUY" else "SELL Signal Detected"

            reasons_text = ", ".join(meta.get("reasons", [])) if meta.get("reasons") else "—"
            atr_line = f"🔎 ATR: {atr:.6f}\n" if atr else ""

            # Format message (kept pretty)
            msg = (
                f"{emoji} *{title}*\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"📊 *Strength:* {strength_label}\n"
                f"💪 *Score:* {score:.2f}\n\n"
                f"💱 *Pair:* `{symbol}`\n"
                f"⏱ *Timeframe:* 5m, 15m & 1h\n"
                f"🕒 *Time:* {last5['close_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                f"🎯 *Entry Zone:*\n"
                f"  • Entry 1: `{entry1:.4f}`\n"
                f"  • Entry 2: `{entry2:.4f}`\n\n"
                f"💰 *Targets:*\n"
                f"  • TP1: `{tp1:.4f}`\n"
                f"  • TP2: `{tp2:.4f}`\n\n"
                f"🛑 *Stop Loss:* `{sl:.4f}`\n"
                f"{atr_line}"
                f"📈 *Reasons:* {reasons_text}\n"
                f"⚙️ _Info only — no auto order._\n"
                f"━━━━━━━━━━━━━━━━━━━"
            )

            send_message(msg)

            last_signals[symbol] = signal
            total_signals += 1
            if signal == "BUY":
                summary["buy"] += 1
            else:
                summary["sell"] += 1

            log_signal_csv({
                "time_utc": last5["close_time"].strftime('%Y-%m-%d %H:%M:%S'),
                "pair": symbol,
                "signal": signal,
                "score": score,
                "entry1": entry1,
                "entry2": entry2,
                "tp1": tp1,
                "tp2": tp2,
                "sl": sl,
                "atr": atr
            })

        except Exception as e:
            logging.error(f"{symbol}: ⚠️ Error scanning: {e}")

    summary["total"] = summary["buy"] + summary["sell"]
    logging.info(f"📊 Summary: BUY={summary['buy']}, SELL={summary['sell']}, TOTAL={summary['total']}")
    return summary

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    load_last_signals()
    send_message("🚀 MrT Combo+Swift Ultra v2.1 aktif (5m/15m/1h) — ATR-based SL/TP agresif")
    total = scan_once()
    save_last_signals()
    send_message(f"✅ Scan selesai. {total} sinyal baru ditemukan.")

if __name__ == "__main__":
    main()
