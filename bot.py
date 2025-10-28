# bot.py
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
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "TRXUSDT", "UNIUSDT", "SUIUSDT", "SHIBUSDT", "MATICUSDT",
    "LTCUSDT", "BCHUSDT", "ICPUSDT", "ALGOUSDT", "AAVEUSDT"
]

TP_MULTIPLIER = 1.5
SL_MULTIPLIER = 1.0
SCAN_INTERVAL = 15  # menit antar scan

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.error("❌ Missing TELEGRAM_TOKEN or CHAT_ID di environment variables.")
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

# === LAST SIGNALS PERSISTENCE ===
def load_last_signals():
    global last_signals
    try:
        if os.path.exists(LAST_SIGNALS_FILE):
            with open(LAST_SIGNALS_FILE, "r") as f:
                last_signals.update(json.load(f))
                logging.info(f"Last signals loaded: {len(last_signals)} entries")
    except Exception as e:
        logging.error(f"Gagal memuat {LAST_SIGNALS_FILE}: {e}")

def save_last_signals():
    try:
        with open(LAST_SIGNALS_FILE, "w") as f:
            json.dump(last_signals, f, indent=2)
    except Exception as e:
        logging.error(f"Gagal menyimpan {LAST_SIGNALS_FILE}: {e}")

# === GET KLINES (Coingecko) ===
def get_klines(symbol, interval="15m", limit=200):
    """
    Ambil data candlestick (klines) dari Bybit.
    Fitur:
    - Otomatis ganti endpoint jika diblokir (HTTP 451)
    - Retry otomatis hingga 3x per endpoint
    """
    endpoints = [
        "https://www.okx.com/api/v5/market/candles"
    ]

    for url in endpoints:
        for attempt in range(3):  # maksimal 3 percobaan per endpoint
            try:
                params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
                response = requests.get(url, params=params, timeout=10)

                # Jika diblokir (HTTP 451)
                if response.status_code == 451:
                    logging.warning(f"⚠️ Endpoint {url} diblokir (451) — mencoba endpoint lain...")
                    break  # langsung ke endpoint berikutnya

                response.raise_for_status()
                data = response.json()

                # Pastikan data valid
                if not isinstance(data, list) or len(data) == 0:
                    logging.warning(f"⚠️ Data kosong dari {url} untuk {symbol} ({interval}) — mencoba ulang...")
                    time.sleep(2)
                    continue

                # Buat DataFrame
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "trades",
                    "taker_base_volume", "taker_quote_volume", "ignore"
                ])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

                logging.info(f"✅ get_klines OK untuk {symbol} ({interval}) dari {url}")
                return df

            except requests.exceptions.Timeout:
                logging.warning(f"⏱ Timeout ({attempt+1}/3) dari {url} untuk {symbol} ({interval}) — retrying...")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                logging.error(f"[ERROR] Request gagal {symbol} ({interval}) dari {url}: {e}")
                time.sleep(2)
            except Exception as e:
                logging.error(f"[ERROR] Parsing gagal {symbol} ({interval}) dari {url}: {e}")
                time.sleep(2)

        # lanjut ke endpoint berikutnya setelah 3 percobaan gagal
        logging.warning(f"🚫 Gagal 3x di {url} untuk {symbol} ({interval}) — mencoba endpoint lain...")

    # Semua endpoint gagal
    logging.error(f"❌ Semua endpoint gagal untuk {symbol} ({interval})")
    return pd.DataFrame()

# === DETEKSI SINYAL ===
def detect_signal(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_diff"] = df["close"] - df["vwap"]

    if len(df) < 3:
        return None

    last, prev = df.iloc[-1], df.iloc[-2]
    bullish_div = (last["close"] > prev["close"]) and (last["vwap_diff"] < prev["vwap_diff"])
    bearish_div = (last["close"] < prev["close"]) and (last["vwap_diff"] > prev["vwap_diff"])

    def kernel_smooth(series, kernel_size=5):
        kernel = np.exp(-0.5 * (np.linspace(-2, 2, kernel_size) ** 2))
        kernel /= kernel.sum()
        return np.convolve(series, kernel, mode='same')

    df["rsi_kernel"] = kernel_smooth(df["rsi"].fillna(method="bfill"))
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    signal, strength, mode = None, None, None
    if bullish_div and df["rsi_kernel"].iloc[-1] < 40 and df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]:
        signal, mode = "BUY", "VWAP-RSI-Kernel"
        strength = "Strong" if df["volume_ratio"].iloc[-1] > 1.5 else "Normal"
    elif bearish_div and df["rsi_kernel"].iloc[-1] > 60 and df["macd"].iloc[-1] < df["macd_signal"].iloc[-1]:
        signal, mode = "SELL", "VWAP-RSI-Kernel"
        strength = "Strong" if df["volume_ratio"].iloc[-1] > 1.5 else "Normal"

    if signal:
        details = {
            "rsi": df["rsi_kernel"].iloc[-1],
            "macd": df["macd"].iloc[-1],
            "macd_signal": df["macd_signal"].iloc[-1],
            "volume_ratio": df["volume_ratio"].iloc[-1],
            "ema50": df["ema50"].iloc[-1],
            "atr": df["atr"].iloc[-1],
        }
        return signal, strength, mode, details
    return None

# === KONFIRMASI MULTI TF ===
def confirm_signal(signal_small_tf, signal_big_tf):
    if not signal_small_tf or not signal_big_tf:
        return None
    if signal_small_tf[0] == signal_big_tf[0]:
        signal, strength, mode, details = signal_small_tf
        return signal, f"{strength}+Confirmed", f"{mode} MTF", details
    return None

# === SCAN SEKALI ===
def scan_once():
    total_signals = 0
    for symbol in SYMBOLS:
        try:
            df_small = get_klines(symbol, TIMEFRAMES[0])
            df_big = get_klines(symbol, TIMEFRAMES[2])
            if df_small.empty or df_big.empty:
                continue

            res_small = detect_signal(df_small)
            res_big = detect_signal(df_big)
            result = confirm_signal(res_small, res_big)

            if result:
                signal, strength, mode, details = result
                if last_signals.get(symbol) == signal:
                    continue  # abaikan duplikat

                total_signals += 1
                last_signals[symbol] = signal
                last = df_small.iloc[-1]
                close_price = last["close"]
                atr = details["atr"]

                if signal == "BUY":
                    entry = close_price
                    tp = entry + (atr * TP_MULTIPLIER)
                    sl = entry - (atr * SL_MULTIPLIER)
                    emoji = "🟢"
                else:
                    entry = close_price
                    tp = entry - (atr * TP_MULTIPLIER)
                    sl = entry + (atr * SL_MULTIPLIER)
                    emoji = "🔴"

                msg = (
                    f"{emoji} *{signal} Signal ({strength})*\n"
                    f"Mode: `{mode}`\n"
                    f"Pair: `{symbol}` | TF: `{TIMEFRAMES[0]} & {TIMEFRAMES[2]}`\n"
                    f"Entry: `{entry:.4f}`\n"
                    f"TP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                    f"ATR: {atr:.4f}\n"
                    f"RSI-Kernel: {details['rsi']:.2f}\n"
                    f"MACD: {details['macd']:.4f} | Signal: {details['macd_signal']:.4f}\n"
                    f"Volume: {details['volume_ratio']:.2f}x rata-rata\n"
                    f"EMA50: {details['ema50']:.2f}\n"
                    f"Time: {last['close_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                    "_Info only — no auto order._"
                )
                send_message(msg)
                logging.info(f"{symbol} {signal} ({strength}) {mode}")

        except Exception as e:
            logging.error(f"Error {symbol}: {e}")
    return total_signals

# === MAIN LOOP ===
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_last_signals()
    send_message(f"🚀 Combo+Booster aktif\n📊 {len(SYMBOLS)} pair | TF: {', '.join(TIMEFRAMES)}\n⏱ Scan tiap {SCAN_INTERVAL} menit")

    while True:
        start = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Mulai scan ({start} UTC)")
        total = scan_once()
        save_last_signals()

        if total > 0:
            send_message(f"✅ Scan selesai ({start}). {total} sinyal baru ditemukan.")
        else:
            logging.info("Tidak ada sinyal baru kali ini.")

        time.sleep(SCAN_INTERVAL * 60)

if __name__ == "__main__":
    main()
