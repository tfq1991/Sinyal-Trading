# =====================================
# === MrT Scalper Combo + Booster ===
# === Swing Short-Term Conservative ===
# === VWAP Divergence + RSI Opt ===
# =====================================
import logging, os, sys, time, json, requests, numpy as np, pandas as pd, ta
from datetime import datetime

TIMEFRAMES = ["15m", "1h"]

SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT",
    "TRXUSDT","OKBUSDT","LTCUSDT","SHIBUSDT","UNIUSDT","ATOMUSDT","BCHUSDT","FILUSDT","ETCUSDT","APTUSDT",
    "NEARUSDT","ARBUSDT","OPUSDT","ICPUSDT","TONUSDT","AAVEUSDT","SANDUSDT","THETAUSDT","EGLDUSDT","FLOWUSDT",
    "COREUSDT","XTZUSDT","MANAUSDT","GALAUSDT","PEPEUSDT","SNXUSDT","CRVUSDT","INJUSDT","XLMUSDT","CFXUSDT",
    "CHZUSDT","NEOUSDT","COMPUSDT","IMXUSDT","ZROUSDT","WLDUSDT","SUIUSDT","PYTHUSDT"
]

TP_PERCENT = 0.05
SL_PERCENT = 0.025
SCAN_INTERVAL = 15

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
if not TELEGRAM_TOKEN or not CHAT_ID:
    sys.exit("❌ Missing TELEGRAM_TOKEN atau CHAT_ID")

LAST_SIGNALS_FILE = "last_signals.json"
last_signals = {}

def send_message(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except Exception: pass

def load_last_signals():
    global last_signals
    if os.path.exists(LAST_SIGNALS_FILE):
        with open(LAST_SIGNALS_FILE, "r") as f: last_signals.update(json.load(f))

def save_last_signals():
    with open(LAST_SIGNALS_FILE, "w") as f: json.dump(last_signals, f, indent=2)

def get_klines(symbol, interval="15m", limit=200):
    tf_map = {"15m":"15m","1h":"1H"}
    interval = tf_map.get(interval, "15m")
    if "-" not in symbol: symbol = symbol.replace("USDT", "-USDT")
    for base in ["https://www.okx.com","https://www.okx.cab","https://aws.okx.com"]:
        try:
            url = f"{base}/api/v5/market/candles"
            r = requests.get(url, params={"instId":symbol,"bar":interval,"limit":limit}, timeout=10)
            data = r.json()["data"][::-1]
            df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
            for c in ["open","high","low","close","volume"]: df[c]=df[c].astype(float)
            df["open_time"]=pd.to_datetime(df["open_time"],unit="ms"); df["close_time"]=df["open_time"]
            return df
        except Exception: continue
    return pd.DataFrame()

def detect_signal_1h(df):
    if len(df)<50: return None
    df["ema50"]=ta.trend.EMAIndicator(df["close"],50).ema_indicator()
    df["ema200"]=ta.trend.EMAIndicator(df["close"],200).ema_indicator()
    df["rsi"]=ta.momentum.RSIIndicator(df["close"],14).rsi()
    macd=ta.trend.MACD(df["close"],26,12,9)
    df["macd"],df["macd_signal"]=macd.macd(),macd.macd_signal()
    last=df.iloc[-1]
    if last["ema50"]>last["ema200"] and last["macd"]>last["macd_signal"] and last["rsi"]>55: return "BUY","1H Trend","Strong"
    if last["ema50"]<last["ema200"] and last["macd"]<last["macd_signal"] and last["rsi"]<45: return "SELL","1H Trend","Strong"
    return None

def detect_signal_15m(df):
    if len(df)<50: return None
    df["rsi"]=ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["rsi_fast"]=ta.momentum.RSIIndicator(df["close"],7).rsi()
    df["rsi_smooth"]=df["rsi"].rolling(5).mean()
    macd=ta.trend.MACD(df["close"],26,12,9)
    df["macd"],df["macd_signal"]=macd.macd(),macd.macd_signal()
    df["vwap"]=ta.volume.VolumeWeightedAveragePrice(df["high"],df["low"],df["close"],df["volume"]).volume_weighted_average_price()
    df["obv"]=ta.volume.OnBalanceVolumeIndicator(df["close"],df["volume"]).on_balance_volume()
    df["obv_ma"]=df["obv"].rolling(20).mean()
    df["volume_ratio"]=df["volume"]/df["volume"].rolling(20).mean()
    df["price_change"]=df["close"].pct_change(); df["vwap_change"]=df["vwap"].pct_change()
    df["vwap_div"]=df["price_change"]*df["vwap_change"]
    last=df.iloc[-1]
    if (
        last["rsi_smooth"]>55 and last["rsi_fast"]>57 and
        last["macd"]>last["macd_signal"] and last["close"]>last["vwap"] and
        last["obv"]>last["obv_ma"] and last["vwap_div"]<-0.0015 and
        last["volume_ratio"]>1.2
    ): return "BUY","RSIOpt+VWAP","Confirmed"
    if (
        last["rsi_smooth"]<45 and last["rsi_fast"]<43 and
        last["macd"]<last["macd_signal"] and last["close"]<last["vwap"] and
        last["obv"]<last["obv_ma"] and last["vwap_div"]<-0.0015 and
        last["volume_ratio"]>1.2
    ): return "SELL","RSIOpt+VWAP","Confirmed"
    return None

def confirm_signal(sig_small, sig_big):
    if not sig_small or not sig_big: return None
    s,b=sig_small,sig_big
    if s[0]==b[0]: return s[0],"Strong Confirmed",f"{s[1]}+{b[1]}"
    return None

def scan_once():
    total=0
    for sym in SYMBOLS:
        try:
            df15=get_klines(sym,"15m"); df1h=get_klines(sym,"1h")
            if df15.empty or df1h.empty: continue
            s15=detect_signal_15m(df15); s1h=detect_signal_1h(df1h)
            res=confirm_signal(s15,s1h)
            if not res: continue
            sig,strength,mode=res
            if last_signals.get(sym)==sig: continue
            last=df15.iloc[-1]; entry=last["close"]
            tp=entry*(1+TP_PERCENT if sig=="BUY" else 1-TP_PERCENT)
            sl=entry*(1-SL_PERCENT if sig=="BUY" else 1+SL_PERCENT)
            emoji="🟢" if sig=="BUY" else "🔴"
            msg=(f"{emoji} *{sig} Signal*\nStrength: {strength}\nMode: `{mode}`\n"
                 f"Pair: `{sym}` | TF: 15m+1h\nEntry: `{entry:.4f}`\n"
                 f"TP: `{tp:.4f}` | SL: `{sl:.4f}`\n"
                 f"RSI(opt): {last['rsi_smooth']:.2f} | VWAP_Div: {last['vwap_div']:.4f}\n"
                 f"VolRatio: {last['volume_ratio']:.2f}\nTime: {last['close_time']}")
            send_message(msg); last_signals[sym]=sig; total+=1
        except Exception as e: logging.error(f"{sym}: {e}")
    return total

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    load_last_signals()
    send_message("🚀 MrT Combo+Booster Conservative Mode aktif\n🎯 Swing pendek | VWAP & RSI Optimized\n⏱ Scan tiap 15 menit")
    while True:
        total=scan_once(); save_last_signals()
        send_message(f"✅ Scan selesai. {total} sinyal baru.")
        time.sleep(SCAN_INTERVAL*60)

if __name__=="__main__":
    main()
