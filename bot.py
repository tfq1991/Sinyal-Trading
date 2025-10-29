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

# TP / SL default (swing short-term conservative)
TP_PERCENT = 0.05   # +5%
SL_PERCENT = 0.025  # -2.5%

SCAN_INTERVAL = 15  # minutes

# Hybrid filter behavior:
APPLY_HYBRID_FILTER = False   # <-- DEFAULT: False (does not block original signals)
HYBRID_THRESHOLD = 4         # if APPLY_HYBRID_FILTER True, require score >= this

# Logging / persistence
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.error("❌ Missing TELEGRAM_TOKEN or CHAT_ID in environment variables.")
    # don't sys.exit here so dev can run in debug without telegram, but warn:
    # If you want to disable telegram requirement uncomment next line:
    # sys.exit(1)

LAST_SIGNALS_FILE = "last_signals.json"
SIGNALS_CSV = "signals_log.csv"
last_signals = {}

# -------------------------
# UTIL: Telegram + persistence
# -------------------------
def send_message(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.debug("Telegram not configured; message suppressed.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Telegram API {resp.status_code}: {resp.text}")
    except Exception as e:
        logging.error(f"[ERROR] Failed send telegram message: {e}")

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
# GET KLINES (OKX endpoints fallback)
# -------------------------
def get_klines(symbol, interval="15m", limit=300):
    tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W"
    }
    interval = tf_map.get(interval, "15m")

    # OKX wants like BTC-USDT
    if "-" not in symbol and symbol.endswith("USDT"):
        inst = symbol.replace("USDT", "-USDT")
    else:
        inst = symbol

    endpoints = [
        "https://www.okx.com",
        "https://www.okx.cab",
        "https://www.okx.co",
        "https://aws.okx.com",
    ]

    for base_url in endpoints:
        for attempt in range(2):
            try:
                url = f"{base_url}/api/v5/market/candles"
                params = {"instId": inst, "bar": interval, "limit": limit}
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if "data" not in data or len(data["data"]) == 0:
                    logging.warning(f"Empty data {base_url} {inst}")
                    break
                raw = data["data"][::-1]  # reverse to chronological
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "volCcy", "volCcyQuote", "confirm"
                ])
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = df[c].astype(float)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = df["open_time"]
                return df
            except Exception as e:
                logging.debug(f"OKX fetch error {base_url} {inst}: {e}")
                time.sleep(1)
        logging.warning(f"Endpoint failed: {base_url} for {inst}")
    logging.error(f"All OKX endpoints failed for {inst}")
    return pd.DataFrame()

# -------------------------
# Extra signal detectors (ADDITIONAL) - non-destructive
# -------------------------
def detect_extra_signals(df):
    """
    Calculate ADX, Bollinger breakout, StochRSI, and simple candlestick patterns.
    Returns dictionary of flags and small diagnostic values.
    """
    out = {
        "adx": None,
        "+di": None,
        "-di": None,
        "bb_break": None,
        "stoch_k": None,
        "stoch_d": None,
        "stoch_signal": None,
        "bullish_engulfing": False,
        "bearish_engulfing": False,
        "hammer": False,
        "shooting_star": False
    }

    try:
        # ADX
        adx_obj = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14, fillna=True)
        out["adx"] = float(adx_obj.adx().iloc[-1])
        out["+di"] = float(adx_obj.adx_pos().iloc[-1])
        out["-di"] = float(adx_obj.adx_neg().iloc[-1])

        # Bollinger Bands breakout
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2, fillna=True)
        out["bb_high"] = float(bb.bollinger_hband().iloc[-1])
        out["bb_low"] = float(bb.bollinger_lband().iloc[-1])
        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        # breakout up/down
        if last_close > out["bb_high"] and last_close > prev_close:
            out["bb_break"] = "UP"
        elif last_close < out["bb_low"] and last_close < prev_close:
            out["bb_break"] = "DOWN"
        else:
            out["bb_break"] = None

        # StochRSI
        stoch = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
        k = stoch.stochrsi_k().iloc[-1]
        d = stoch.stochrsi_d().iloc[-1]
        out["stoch_k"] = float(k)
        out["stoch_d"] = float(d)
        # signal rules
        if k > d and k < 30:
            out["stoch_signal"] = "BUY"
        elif k < d and k > 70:
            out["stoch_signal"] = "SELL"
        else:
            out["stoch_signal"] = None

        # Simple candlestick patterns (basic heuristics)
        # Bullish engulfing
        o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
        o2, c2 = df["open"].iloc[-1], df["close"].iloc[-1]
        if (c2 > o2) and (o2 < c1) and (c2 > o1):
            out["bullish_engulfing"] = True
        if (c2 < o2) and (o2 > c1) and (c2 < o1):
            out["bearish_engulfing"] = True

        # Hammer / Shooting star (body small, long wick)
        high = df["high"].iloc[-1]; low = df["low"].iloc[-1]
        body = abs(c2 - o2)
        upper_wick = high - max(c2, o2)
        lower_wick = min(c2, o2) - low
        if body > 0 and (lower_wick / body) > 2 and upper_wick < body:
            out["hammer"] = True
        if body > 0 and (upper_wick / body) > 2 and lower_wick < body:
            out["shooting_star"] = True

    except Exception as e:
        logging.debug(f"detect_extra_signals error: {e}")

    return out

# -------------------------
# Existing detect_signal for 1H (keaslian tidak diubah)
# -------------------------
def detect_signal(df, interval="1h"):
    if df.shape[0] < 50:
        return None
    try:
        df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
    except Exception as e:
        logging.error(f"Indicator calc error (1h): {e}")
        return None

    last = df.iloc[-1]

    if last["ema50"] > last["ema200"] and last["macd"] > last["macd_signal"] and last["rsi"] > 55:
        return "BUY", "Trend Confirm", "Strong"
    elif last["ema50"] < last["ema200"] and last["macd"] < last["macd_signal"] and last["rsi"] < 45:
        return "SELL", "Trend Confirm", "Strong"

    return None

# -------------------------
# detect_signal_15m (core logic preserved) + new hybrid scoring
# -------------------------
def detect_signal_15m(df):
    """
    Core original checks remain (RSI, MACD, VWAP, OBV, volume ratio).
    Additional indicators computed and returned as meta (hybrid score + flags).
    By default we DO NOT block the original signal path; set APPLY_HYBRID_FILTER True
    if you want to require a minimum hybrid score to accept a signal.
    """
    if df.shape[0] < 60:
        return None

    try:
        # RSI Optimized (14 smoothed + fast 7)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_fast"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
        df["rsi_smooth"] = df["rsi"].rolling(5).mean()

        # MACD standard
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()

        # VWAP
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
        ).volume_weighted_average_price()

        # OBV and volume ratio
        obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"])
        df["obv"] = obv.on_balance_volume()
        df["obv_ma"] = df["obv"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # VWAP divergence measure (pct change cross-check)
        df["price_change"] = df["close"].pct_change()
        df["vwap_change"] = df["vwap"].pct_change()
        # more negative means stronger opposite move
        df["vwap_div"] = df["price_change"] * df["vwap_change"]

        # Extra signals
        extra = detect_extra_signals(df)

    except Exception as e:
        logging.error(f"Indicator error 15m: {e}")
        return None

    last = df.iloc[-1]

    # Original "baseline" conditions (kept unchanged)
    base_buy = (
        (last["rsi"] > 50)
        and (last["macd"] > last["macd_signal"])
        and (last["close"] > last["vwap"])
        and (last["obv"] > last["obv_ma"])
        and (last["volume_ratio"] > 1.0)
    )

    base_sell = (
        (last["rsi"] < 50)
        and (last["macd"] < last["macd_signal"])
        and (last["close"] < last["vwap"])
        and (last["obv"] < last["obv_ma"])
        and (last["volume_ratio"] > 1.0)
    )

    # Compute hybrid score (additive)
    score = 0
    reasons = []

    # base signals count as points (but do not block/override)
    if last["rsi"] > 50:
        score += 1; reasons.append("RSI>50")
    else:
        reasons.append("RSI<=50")

    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD>Signal")
    else:
        reasons.append("MACD<Signal")

    if last["close"] > last["vwap"]:
        score += 1; reasons.append("Price>VWAP")
    else:
        reasons.append("Price<VWAP")

    if last["obv"] > last["obv_ma"]:
        score += 1; reasons.append("OBV>MA")
    else:
        reasons.append("OBV<=MA")

    if last["volume_ratio"] > 1.2:
        score += 1; reasons.append("Vol>1.2x")
    elif last["volume_ratio"] > 1.0:
        score += 0.5; reasons.append("Vol>1.0x")
    else:
        reasons.append("LowVol")

    # VWAP divergence: negative indicates opposite short-term movement -> potentially meaningful
    # stronger threshold for higher weight
    if last["vwap_div"] < -0.0015:
        score += 1.5; reasons.append("VWAPDivStrong")
    elif last["vwap_div"] < -0.0005:
        score += 0.7; reasons.append("VWAPDivWeak")

    # RSI optimized stronger check
    if last["rsi_smooth"] > 55 and last["rsi_fast"] > 57:
        score += 1; reasons.append("RSIOptStrong")
    elif last["rsi_smooth"] > 52:
        score += 0.5; reasons.append("RSIOptWeak")

    # extra signals
    if extra.get("adx") is not None:
        if extra["adx"] > 25 and extra["+di"] > extra["-di"]:
            score += 1; reasons.append("ADXUp")
        elif extra["adx"] > 25 and extra["-di"] > extra["+di"]:
            score -= 0.5; reasons.append("ADXDown")

    if extra.get("bb_break") == "UP":
        score += 1; reasons.append("BBBreakUp")
    if extra.get("bb_break") == "DOWN":
        score -= 0.8; reasons.append("BBBreakDown")

    if extra.get("stoch_signal") == "BUY":
        score += 0.8; reasons.append("StochBuy")
    if extra.get("stoch_signal") == "SELL":
        score -= 0.8; reasons.append("StochSell")

    if extra.get("bullish_engulfing"):
        score += 0.6; reasons.append("BullEngulf")
    if extra.get("bearish_engulfing"):
        score -= 0.6; reasons.append("BearEngulf")
    if extra.get("hammer"):
        score += 0.5; reasons.append("Hammer")
    if extra.get("shooting_star"):
        score -= 0.5; reasons.append("ShootingStar")

    # clamp score >= -5 .. 10 for sanity
    score = float(max(-5.0, min(10.0, score)))

    # determine final label: we keep original detection behavior, but add hybrid meta
    if base_buy:
        # label strength by score
        if score >= 6:
            strength = "Strong"
        elif score >= 4:
            strength = "Confirmed"
        else:
            strength = "Weak"
        signal = "BUY"
    elif base_sell:
        if score >= 6:
            strength = "Strong"
        elif score >= 4:
            strength = "Confirmed"
        else:
            strength = "Weak"
        signal = "SELL"
    else:
        # original conditions not met: return None but still provide extras for debugging
        return None

    # If user chose to enforce hybrid filter, block low-score signals
    if APPLY_HYBRID_FILTER and score < HYBRID_THRESHOLD:
        logging.info(f"Signal blocked by hybrid filter (score {score:.2f} < {HYBRID_THRESHOLD})")
        return None

    # return tuple and meta
    mode = "RSI+MACD+VWAP+OBV"
    meta = {
        "score": score,
        "reasons": reasons,
        "extra": extra,
        "rsi": float(last["rsi"]),
        "rsi_smooth": float(last["rsi_smooth"]) if not np.isnan(last["rsi_smooth"]) else None,
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "vwap_div": float(last["vwap_div"]) if not np.isnan(last["vwap_div"]) else None,
        "volume_ratio": float(last["volume_ratio"])
    }

    return signal, mode, strength, meta

# -------------------------
# Multi-timeframe confirm (keaslian logika dipertahankan)
# -------------------------
def confirm_signal(signal_small_tf, signal_big_tf):
    if not signal_small_tf or not signal_big_tf:
        return None
    # small tf: (signal, mode, strength, meta)
    s_signal, s_mode, s_strength, s_meta = signal_small_tf
    b_signal, b_mode, b_strength = signal_big_tf
    # original behavior: require same direction
    if s_signal == b_signal:
        combined_strength = "Strong Confirmed" if ("Strong" in [s_strength, b_strength] or s_meta.get("score",0) >= 6) else "Confirmed"
        mode = f"{s_mode}+{b_mode}"
        tf_info = f"[15m + 1h]"
        return s_signal, combined_strength, f"{mode} MTF {tf_info}", s_meta
    return None

# -------------------------
# SCAN ONCE and messaging (keaslian pesan dipertahankan, meta ditambahkan)
# -------------------------
def scan_once():
    total_signals = 0
    for symbol in SYMBOLS:
        try:
            df15 = get_klines(symbol, "15m")
            df1h = get_klines(symbol, "1h")
            if df15.empty or df1h.empty:
                continue

            res15 = detect_signal_15m(df15)   # returns (signal, mode, strength, meta) or None
            res1h = detect_signal(df1h, "1h") # returns (signal, mode, strength) or None

            if not res15 or not res1h:
                continue

            # adapt res1h to tuple length for confirm_signal
            b_signal, b_mode, b_strength = res1h
            result = confirm_signal(res15, (b_signal, b_mode, b_strength))
            if not result:
                continue

            signal, strength_label, mode_str, meta = result
            # meta comes from res15
            score = meta.get("score", 0.0)
            reasons = meta.get("reasons", [])
            details = meta

            # avoid duplicate same-direction spam
            if last_signals.get(symbol) == signal:
                logging.debug(f"{symbol} same as last signal; skip")
                continue

            last_row = df15.iloc[-1]
            entry = float(last_row["close"])
            if signal == "BUY":
                tp = entry * (1 + TP_PERCENT)
                sl = entry * (1 - SL_PERCENT)
                emoji = "🟢"
            else:
                tp = entry * (1 - TP_PERCENT)
                sl = entry * (1 + SL_PERCENT)
                emoji = "🔴"

            # Build message — keep previous fields and add hybrid meta
            msg = (
                f"{emoji} *{signal} Signal*\n"
                f"Strength: *{strength_label}* (hybrid score: {score:.2f})\n"
                f"Mode: `{mode_str}`\n"
                f"Pair: `{symbol}` | TF: `15m & 1h`\n"
                f"Entry: `{entry:.4f}`\n"
                f"TP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                f"RSI: {details.get('rsi'):.2f} | RSI(smooth): {details.get('rsi_smooth'):.2f}\n"
                f"MACD: {details.get('macd'):.4f} | Signal: {details.get('macd_signal'):.4f}\n"
                f"VWAP Div: {details.get('vwap_div'):.6f}\n"
                f"Volume Ratio: {details.get('volume_ratio'):.2f}x\n"
                f"Extra Flags: ADX={details['extra'].get('adx')}, BB={details['extra'].get('bb_break')}, Stoch={details['extra'].get('stoch_signal')}\n"
                f"Reasons: {', '.join(reasons)}\n"
                f"Time: {last_row['close_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                "_Info only — no auto order._"
            )

            send_message(msg)
            logging.info(f"{symbol} {signal} score={score:.2f} mode={mode_str}")

            # persist last signal and csv log
            last_signals[symbol] = signal
            total_signals += 1

            # log row
            row = {
                "time_utc": last_row["close_time"].strftime('%Y-%m-%d %H:%M:%S'),
                "pair": symbol,
                "signal": signal,
                "score": score,
                "mode": mode_str,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "rsi": details.get("rsi"),
                "rsi_smooth": details.get("rsi_smooth"),
                "macd": details.get("macd"),
                "macd_signal": details.get("macd_signal"),
                "vwap_div": details.get("vwap_div"),
                "volume_ratio": details.get("volume_ratio"),
                "extra_adx": details["extra"].get("adx"),
                "extra_bb": details["extra"].get("bb_break"),
                "extra_stoch": details["extra"].get("stoch_signal")
            }
            try:
                log_signal_csv(row)
            except Exception as e:
                logging.debug(f"CSV logging failed: {e}")

        except Exception as e:
            logging.error(f"Error scanning {symbol}: {e}")

    return total_signals

# -------------------------
# MAIN LOOP
# -------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    load_last_signals()

    send_message(
        f"🚀 MrT Combo+Booster Ultra+Hybrid aktif\n"
        f"Pairs: {len(SYMBOLS)} | TF: {', '.join(TIMEFRAMES)}\n"
        f"TP: +{TP_PERCENT*100:.1f}% | SL: -{SL_PERCENT*100:.1f}%\n"
        f"Hybrid Filter Applied: {APPLY_HYBRID_FILTER} (threshold {HYBRID_THRESHOLD})"
    )

    while True:
        start = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        send_message(f"🔍 Mulai scan otomatis ({start} UTC) — {len(SYMBOLS)} pairs")
        total = scan_once()
        save_last_signals()
        if total > 0:
            send_message(f"✅ Scan selesai. {total} sinyal baru ditemukan.")
        else:
            send_message("✅ Scan selesai. 0 sinyal baru ditemukan.")
        time.sleep(SCAN_INTERVAL * 60)

if __name__ == "__main__":
    main()
