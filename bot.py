# =====================================
# ===  MrT Scalper Combo + Booster  ===
# ===  OKX + MEXC fallback + Debug  ===
# ===  VWAP Div + RSI Opt + ADX/BB/Stoch/Candle/Hybrid ===
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
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
TIMEFRAMES = ["15m", "1h"]

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

TP_PERCENT = 0.05
SL_PERCENT = 0.025
SCAN_INTERVAL = 15

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
        print(msg)  # tampilkan di log agar tetap kelihatan di Actions
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
# GET KLINES
# -------------------------
def get_klines(symbol, interval="15m", limit=300):
    tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W"
    }
    interval = tf_map.get(interval, interval)
    inst = symbol.replace("USDT", "-USDT") if "-" not in symbol else symbol

    endpoints = [
        "https://www.okx.com",
        "https://www.okx.cab",
        "https://www.okx.co",
        "https://aws.okx.com",
    ]

    for base_url in endpoints:
        try:
            url = f"{base_url}/api/v5/market/candles"
            params = {"instId": inst, "bar": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if "data" not in data or len(data["data"]) == 0:
                continue
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
            logging.debug(f"get_klines error {base_url} {inst}: {e}")
    logging.error(f"All OKX endpoints failed for {symbol}")
    return pd.DataFrame()

# -------------------------
# INDICATORS & SIGNAL DETECTORS
# -------------------------
def detect_signal(df, interval="1h"):
    if df.shape[0] < 50:
        return None
    try:
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    except Exception as e:
        logging.error(f"Indicator calc error: {e}")
        return None

    last = df.iloc[-1]
    if last["ema50"] > last["ema200"] and last["macd"] > last["macd_signal"] and last["rsi"] > 55:
        return "BUY", "Trend Confirm", "Strong"
    elif last["ema50"] < last["ema200"] and last["macd"] < last["macd_signal"] and last["rsi"] < 45:
        return "SELL", "Trend Confirm", "Strong"
    return None

def detect_signal_15m(df):
    if df.shape[0] < 60:
        return None
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
        ).volume_weighted_average_price()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["obv_ma"] = df["obv"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    except Exception as e:
        logging.error(f"Indicator error 15m: {e}")
        return None

    last = df.iloc[-1]
    base_buy = (
        last["rsi"] > 50 and last["macd"] > last["macd_signal"] and
        last["close"] > last["vwap"] and last["obv"] > last["obv_ma"] and
        last["volume_ratio"] > 1.0
    )
    base_sell = (
        last["rsi"] < 50 and last["macd"] < last["macd_signal"] and
        last["close"] < last["vwap"] and last["obv"] < last["obv_ma"] and
        last["volume_ratio"] > 1.0
    )

    if base_buy:
        return "BUY", "RSI+MACD+VWAP+OBV", "Strong", {"score": 6, "reasons": ["RSI>50", "MACD>Signal"]}
    elif base_sell:
        return "SELL", "RSI+MACD+VWAP+OBV", "Strong", {"score": 6, "reasons": ["RSI<50", "MACD<Signal"]}
    return None

def confirm_signal(signal_small_tf, signal_big_tf):
    if not signal_small_tf or not signal_big_tf:
        return None
    s_signal, s_mode, s_strength, s_meta = signal_small_tf
    b_signal, b_mode, b_strength = signal_big_tf
    if s_signal == b_signal:
        return s_signal, f"Strong Confirmed", f"{s_mode}+{b_mode}", s_meta
    return None

# -------------------------
# SCANNER
# -------------------------
def scan_once():
    total_signals = 0
    for symbol in SYMBOLS:
        try:
            df15 = get_klines(symbol, "15m")
            df1h = get_klines(symbol, "1h")

            if df15.empty or df1h.empty:
                logging.debug(f"{symbol}: ❌ Data kosong (15m={df15.empty}, 1h={df1h.empty})")
                continue

            res15 = detect_signal_15m(df15)
            res1h = detect_signal(df1h, "1h")

            if not res15:
                logging.debug(f"{symbol}: ⚪ Tidak ada sinyal 15m (indikator belum cocok)")
                continue
            if not res1h:
                logging.debug(f"{symbol}: ⚪ Tidak ada sinyal 1h (trend belum konfirmasi)")
                continue

            b_signal, b_mode, b_strength = res1h
            result = confirm_signal(res15, (b_signal, b_mode, b_strength))
            if not result:
                logging.debug(f"{symbol}: ⚪ 15m dan 1h tidak searah, sinyal batal")
                continue

            signal, strength_label, mode_str, meta = result
            score = meta.get("score", 0.0)
            logging.info(f"{symbol}: ✅ {signal} ({strength_label}) | score={score:.2f}")

            if last_signals.get(symbol) == signal:
                logging.debug(f"{symbol}: Sama seperti sinyal terakhir, skip.")
                continue

            last_row = df15.iloc[-1]
            entry = float(last_row["close"])
            tp = entry * (1 + TP_PERCENT) if signal == "BUY" else entry * (1 - TP_PERCENT)
            sl = entry * (1 - SL_PERCENT) if signal == "BUY" else entry * (1 + SL_PERCENT)

            emoji = "🟢" if signal == "BUY" else "🔴"
            msg = (
                f"{emoji} *{signal} Signal*\n"
                f"Strength: *{strength_label}* (score: {score:.2f})\n"
                f"Pair: `{symbol}` | TF: 15m & 1h\n"
                f"Entry: `{entry:.4f}`\nTP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                f"Reasons: {', '.join(meta.get('reasons', []))}\n"
                f"Time: {last_row['close_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                "_Info only — no auto order._"
            )
            send_message(msg)
            last_signals[symbol] = signal
            total_signals += 1

            log_signal_csv({
                "time_utc": last_row["close_time"].strftime('%Y-%m-%d %H:%M:%S'),
                "pair": symbol,
                "signal": signal,
                "score": score,
                "entry": entry,
                "tp": tp,
                "sl": sl
            })

        except Exception as e:
            logging.error(f"{symbol}: ⚠️ Error scanning: {e}")

    return total_signals

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    load_last_signals()
    send_message("🚀 MrT Combo+Booster Debug aktif")
    total = scan_once()
    save_last_signals()
    send_message(f"✅ Scan selesai. {total} sinyal baru ditemukan.")

if __name__ == "__main__":
    main()
