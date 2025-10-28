# =====================================
# ===  MrT Scalper Combo + Booster  ===
# ===  OKX + MEXC fallback + Debug  ===
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
TIMEFRAMES = ["5m", "15m", "1h", "4h"]

# === 50 Pair Utama di OKX (beberapa duplikat dihapus) ===
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


# === GET KLINES (OKX) ===
def get_klines(symbol, interval="15m", limit=200):
    tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W"
    }
    interval = tf_map.get(interval, "15m")

    if "-" not in symbol and symbol.endswith("USDT"):
        symbol = symbol.replace("USDT", "-USDT")

    endpoints = [
        "https://www.okx.com",
        "https://www.okx.cab",
        "https://www.okx.co",
        "https://aws.okx.com",
    ]

    for base_url in endpoints:
        for attempt in range(3):
            try:
                url = f"{base_url}/api/v5/market/candles"
                params = {"instId": symbol, "bar": interval, "limit": limit}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if "data" not in data or len(data["data"]) == 0:
                    logging.warning(f"⚠️ Data kosong dari {base_url} untuk {symbol}")
                    continue

                raw = data["data"][::-1]
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close",
                    "volume", "volCcy", "volCcyQuote", "confirm"
                ])

                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)

                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = df["open_time"]

                return df

            except Exception as e:
                logging.error(f"[ERROR] {symbol} ({interval}) di {base_url}: {e}")
                time.sleep(2)

        logging.warning(f"🚫 Gagal 3x di {base_url} untuk {symbol}")

    logging.error(f"❌ Semua endpoint OKX gagal untuk {symbol}")
    return pd.DataFrame()


# === DETEKSI SINYAL ===
def detect_signal(df, interval="1h"):
    if df.shape[0] < 50:
        return None

    try:
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
        ).volume_weighted_average_price()
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    except Exception as e:
        logging.error(f"Indicator calc error: {e}")
        return None

    df["vwap_diff"] = df["close"] - df["vwap"]

    df["ema9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()

    stoch_rsi = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    signal, mode, strength = None, None, "Neutral"
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_div = (last["close"] > prev["close"]) and (last["vwap_diff"] < prev["vwap_diff"])
    bearish_div = (last["close"] < prev["close"]) and (last["vwap_diff"] > prev["vwap_diff"])

    if bullish_div and last["rsi"] < 40 and last["macd"] > last["macd_signal"]:
        signal, mode, strength = "BUY", "VWAP-RSI-Kernel", "Classic"
    elif bearish_div and last["rsi"] > 60 and last["macd"] < last["macd_signal"]:
        signal, mode, strength = "SELL", "VWAP-RSI-Kernel", "Classic"

    bullish_trend = last["ema50"] > last["ema200"]
    bearish_trend = last["ema50"] < last["ema200"]

    if bullish_trend and last["macd"] > last["macd_signal"] and last["rsi"] > 50:
        signal, mode, strength = "BUY", "Trend-MACD-RSI", "Strong"
    elif bearish_trend and last["macd"] < last["macd_signal"] and last["rsi"] < 50:
        signal, mode, strength = "SELL", "Trend-MACD-RSI", "Strong"

    if last["close"] > last["vwap"] and last["macd"] > last["macd_signal"]:
        signal, mode, strength = "BUY", "VWAP-MACD", "Confirmed"
    elif last["close"] < last["vwap"] and last["macd"] < last["macd_signal"]:
        signal, mode, strength = "SELL", "VWAP-MACD", "Confirmed"

    if interval in ["1m", "3m", "5m", "15m"]:
        if df["ema9"].iloc[-2] < df["ema21"].iloc[-2] and df["ema9"].iloc[-1] > df["ema21"].iloc[-1]:
            signal, mode, strength = "BUY", "EMA9-21 Cross", "Scalp"
        elif df["ema9"].iloc[-2] > df["ema21"].iloc[-2] and df["ema9"].iloc[-1] < df["ema21"].iloc[-1]:
            signal, mode, strength = "SELL", "EMA9-21 Cross", "Scalp"

        if last["close"] > last["bb_high"] and last.get("volume_ratio", 1.0) > 1.2:
            signal, mode, strength = "BUY", "Bollinger Breakout", "Fast"
        elif last["close"] < last["bb_low"] and last.get("volume_ratio", 1.0) > 1.2:
            signal, mode, strength = "SELL", "Bollinger Breakout", "Fast"

        if last["stoch_rsi_k"] < 20 and last["stoch_rsi_d"] < 20 and last["macd"] > last["macd_signal"]:
            signal, mode, strength = "BUY", "StochRSI Reversal", "Short"
        elif last["stoch_rsi_k"] > 80 and last["stoch_rsi_d"] > 80 and last["macd"] < last["macd_signal"]:
            signal, mode, strength = "SELL", "StochRSI Reversal", "Short"

    return signal, mode, strength


# === KONFIRMASI MULTI TF ===
def confirm_signal(signal_small_tf, signal_big_tf, tf_small="5m", tf_big="1h"):
    if not signal_small_tf or not signal_big_tf:
        return None

    s_signal, s_mode, s_strength = signal_small_tf
    b_signal, b_mode, b_strength = signal_big_tf

    if s_signal == b_signal:
        if "Strong" in [s_strength, b_strength]:
            strength = "Strong Confirmed"
        elif "Confirmed" in [s_strength, b_strength]:
            strength = "Confirmed"
        elif "Classic" in [s_strength, b_strength]:
            strength = "Classic Confirmed"
        else:
            strength = "Normal"

        mode = f"{s_mode}+{b_mode}"
        tf_info = f"[{tf_small} + {tf_big}]"

        return s_signal, strength, f"{mode} MTF {tf_info}"

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

            res_small = detect_signal(df_small, interval=TIMEFRAMES[0])
            res_big = detect_signal(df_big, interval=TIMEFRAMES[2])
            result = confirm_signal(res_small, res_big, TIMEFRAMES[0], TIMEFRAMES[2])

            if result:
                signal, strength, mode = result

                if last_signals.get(symbol) == signal:
                    continue

                allowed_strength = ["Strong", "Confirmed", "Classic", "Strong Confirmed", "Classic Confirmed"]
                if strength not in allowed_strength:
                    continue

                last = df_small.iloc[-1]

                try:
                    atr_calc = ta.volatility.AverageTrueRange(
                        high=df_small["high"], low=df_small["low"], close=df_small["close"], window=14
                    )
                    atr_series = atr_calc.average_true_range()
                    atr = float(atr_series.iloc[-1]) if not atr_series.isna().all() else 0.0
                except Exception:
                    atr = 0.0

                details = {
                    "atr": float(atr),
                    "rsi": float(df_small["rsi"].iloc[-1]) if "rsi" in df_small.columns else None,
                    "macd": float(df_small["macd"].iloc[-1]) if "macd" in df_small.columns else None,
                    "macd_signal": float(df_small["macd_signal"].iloc[-1]) if "macd_signal" in df_small.columns else None,
                    "volume_ratio": float(df_small["volume_ratio"].iloc[-1]) if "volume_ratio" in df_small.columns else 1.0,
                    "ema50": float(df_small["ema50"].iloc[-1]) if "ema50" in df_small.columns else float(df_small["close"].iloc[-1])
                }

                close_price = float(last["close"])
                if signal == "BUY":
                    entry = close_price
                    tp = entry + (details["atr"] * TP_MULTIPLIER)
                    sl = entry - (details["atr"] * SL_MULTIPLIER)
                    emoji = "🟢"
                else:
                    entry = close_price
                    tp = entry - (details["atr"] * TP_MULTIPLIER)
                    sl = entry + (details["atr"] * SL_MULTIPLIER)
                    emoji = "🔴"

                total_signals += 1
                last_signals[symbol] = signal

                rsi_str = f"{details['rsi']:.2f}" if details['rsi'] is not None else "N/A"
                macd_str = f"{details['macd']:.4f}" if details['macd'] is not None else "N/A"
                macd_sig_str = f"{details['macd_signal']:.4f}" if details['macd_signal'] is not None else "N/A"

                msg = (
                    f"{emoji} *{signal} Signal ({strength})*\n"
                    f"Mode: `{mode}`\n"
                    f"Pair: `{symbol}` | TF: `{TIMEFRAMES[0]} & {TIMEFRAMES[2]}`\n"
                    f"Entry: `{entry:.4f}`\n"
                    f"TP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                    f"ATR: {details['atr']:.4f}\n"
                    f"RSI-Kernel: {rsi_str}\n"
                    f"MACD: {macd_str} | Signal: {macd_sig_str}\n"
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

    send_message(
        f"🚀 Combo+Booster aktif\n"
        f"📊 *{len(SYMBOLS)} pair aktif* | TF: {', '.join(TIMEFRAMES)}\n"
        f"⏱ Scan tiap *{SCAN_INTERVAL} menit*"
    )

    while True:
        start = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        send_message(f"🚀 Mulai scan otomatis ({start} UTC)\n📊 {len(SYMBOLS)} pair | TF: {', '.join(TIMEFRAMES)}")

        total = scan_once()
        save_last_signals()

        if total > 0:
            send_message(f"✅ Scan selesai ({start}). {total} sinyal baru ditemukan.")
        else:
            send_message("✅ Scan selesai. 0 sinyal baru ditemukan.")

        time.sleep(SCAN_INTERVAL * 60)


if __name__ == "__main__":
    main()
