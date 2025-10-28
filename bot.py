# =====================================
# ===  MrT Scalper Combo + Booster  ===
# ===  OKX + MEXC fallback + Debug  ===
# ===  VWAP Divergence + RSI Opt ===
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

# === KONFIGURASI ===
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

TP_PERCENT = 0.06
SL_PERCENT = 0.03
SCAN_INTERVAL = 15

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.error("❌ Missing TELEGRAM_TOKEN atau CHAT_ID di environment variables.")
    sys.exit(1)

LAST_SIGNALS_FILE = "last_signals.json"
last_signals = {}


# === TELEGRAM ===
def send_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Telegram API {resp.status_code}: {resp.text}")
    except Exception as e:
        logging.error(f"[ERROR] Gagal kirim pesan Telegram: {e}")


# === LAST SIGNALS ===
def load_last_signals():
    global last_signals
    try:
        if os.path.exists(LAST_SIGNALS_FILE):
            with open(LAST_SIGNALS_FILE, "r") as f:
                last_signals.update(json.load(f))
    except Exception as e:
        logging.error(f"Gagal memuat {LAST_SIGNALS_FILE}: {e}")


def save_last_signals():
    try:
        with open(LAST_SIGNALS_FILE, "w") as f:
            json.dump(last_signals, f, indent=2)
    except Exception as e:
        logging.error(f"Gagal menyimpan {LAST_SIGNALS_FILE}: {e}")


# === GET KLINES ===
def get_klines(symbol, interval="15m", limit=200):
    tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D", "1w": "1W"
    }
    interval = tf_map.get(interval, "15m")

    if "-" not in symbol and symbol.endswith("USDT"):
        symbol = symbol.replace("USDT", "-USDT")

    endpoints = ["https://www.okx.com", "https://www.okx.cab", "https://aws.okx.com"]

    for base_url in endpoints:
        try:
            url = f"{base_url}/api/v5/market/candles"
            params = {"instId": symbol, "bar": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            raw = data["data"][::-1]
            df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = df["open_time"]
            return df
        except Exception:
            continue
    return pd.DataFrame()


# === DETEKSI SINYAL 1H ===
def detect_signal(df, interval="1h"):
    if df.shape[0] < 50:
        return None
    df["ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    macd = ta.trend.MACD(df["close"], 26, 12, 9)
    df["macd"], df["macd_signal"] = macd.macd(), macd.macd_signal()

    last = df.iloc[-1]
    if last["ema50"] > last["ema200"] and last["macd"] > last["macd_signal"] and last["rsi"] > 50:
        return "BUY", "Trend Confirm", "Strong"
    elif last["ema50"] < last["ema200"] and last["macd"] < last["macd_signal"] and last["rsi"] < 50:
        return "SELL", "Trend Confirm", "Strong"
    return None


# === DETEKSI SINYAL 15m (VWAP Divergence + RSI Optimized) ===
def detect_signal_15m(df):
    if df.shape[0] < 50:
        return None

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["rsi_fast"] = ta.momentum.RSIIndicator(df["close"], 7).rsi()
    df["rsi_smooth"] = df["rsi"].rolling(3).mean()

    macd = ta.trend.MACD(df["close"], 26, 12, 9)
    df["macd"], df["macd_signal"] = macd.macd(), macd.macd_signal()

    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
    ).volume_weighted_average_price()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["obv_ma"] = df["obv"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # === VWAP Divergence ===
    df["price_change"] = df["close"].pct_change()
    df["vwap_change"] = df["vwap"].pct_change()
    df["vwap_div"] = df["price_change"] * df["vwap_change"]

    last = df.iloc[-1]

    # === Kondisi BUY ===
    if (
        last["rsi_smooth"] > 52
        and last["rsi_fast"] > 50
        and last["macd"] > last["macd_signal"]
        and last["close"] > last["vwap"]
        and last["obv"] > last["obv_ma"]
        and last["vwap_div"] < 0
        and last["volume_ratio"] > 1.0
    ):
        return "BUY", "RSI_Opt+VWAP_Div", "Confirmed"

    # === Kondisi SELL ===
    elif (
        last["rsi_smooth"] < 48
        and last["rsi_fast"] < 50
        and last["macd"] < last["macd_signal"]
        and last["close"] < last["vwap"]
        and last["obv"] < last["obv_ma"]
        and last["vwap_div"] < 0
        and last["volume_ratio"] > 1.0
    ):
        return "SELL", "RSI_Opt+VWAP_Div", "Confirmed"

    return None


# === KONFIRMASI MULTI TF ===
def confirm_signal(signal_small, signal_big):
    if not signal_small or not signal_big:
        return None
    s_sig, s_mode, s_str = signal_small
    b_sig, b_mode, b_str = signal_big
    if s_sig == b_sig:
        strength = "Strong Confirmed" if "Strong" in [s_str, b_str] else "Confirmed"
        mode = f"{s_mode}+{b_mode}"
        return s_sig, strength, mode
    return None


# === SCAN SEKALI ===
def scan_once():
    total_signals = 0
    for symbol in SYMBOLS:
        try:
            df_15m = get_klines(symbol, "15m")
            df_1h = get_klines(symbol, "1h")
            if df_15m.empty or df_1h.empty:
                continue

            res_15m = detect_signal_15m(df_15m)
            res_1h = detect_signal(df_1h)
            result = confirm_signal(res_15m, res_1h)

            if not result:
                continue

            signal, strength, mode = result
            if last_signals.get(symbol) == signal:
                continue

            last = df_15m.iloc[-1]
            entry = last["close"]
            tp = entry * (1 + TP_PERCENT if signal == "BUY" else 1 - TP_PERCENT)
            sl = entry * (1 - SL_PERCENT if signal == "BUY" else 1 + SL_PERCENT)
            emoji = "🟢" if signal == "BUY" else "🔴"

            msg = (
                f"{emoji} *{signal} Signal*\n"
                f"Strength: {strength}\n"
                f"Mode: `{mode}`\n"
                f"Pair: `{symbol}` | TF: 15m+1h\n"
                f"Entry: `{entry:.4f}`\nTP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                f"RSI(opt): {last['rsi_smooth']:.2f} | VWAP_Div: {last['vwap_div']:.4f}\n"
                f"Volume x: {last['volume_ratio']:.2f}\n"
                f"Time: {last['close_time']}\n"
            )
            send_message(msg)
            last_signals[symbol] = signal
            total_signals += 1

        except Exception as e:
            logging.error(f"{symbol} error: {e}")
    return total_signals


# === MAIN LOOP ===
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_last_signals()
    send_message(f"🚀 MrT Combo+Booster aktif\n🎯 VWAP Divergence + RSI Optimized ON\n⏱ Scan tiap {SCAN_INTERVAL} menit")

    while True:
        total = scan_once()
        save_last_signals()
        send_message(f"✅ Scan selesai. {total} sinyal baru ditemukan.")
        time.sleep(SCAN_INTERVAL * 60)


if __name__ == "__main__":
    main()
