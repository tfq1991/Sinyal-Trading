# 🚀 Combo+Booster Trading Signal Bot (OKX + Telegram)

Bot ini melakukan **pemindaian multi-timeframe (MTF)** pada pair utama di **OKX** dan mengirimkan sinyal ke **Telegram** dengan label kekuatan sinyal 💪⚡⚪ berdasarkan kekuatan tren dan indikator konfirmasi.

## 🧠 Fitur Utama
✅ Analisis **50 pair utama OKX** (BTC, ETH, BNB, SOL, dll)  
✅ Multi Timeframe (default: `5m + 1h`)  
✅ Indikator Teknis:
- EMA50, EMA200, EMA9, EMA21  
- RSI & StochRSI  
- MACD & VWAP  
- Bollinger Bands  
- Volume Ratio & ATR  
✅ Deteksi sinyal otomatis & multi-timeframe  
✅ Label kekuatan sinyal:
  - 💪 **Kuat**
  - ⚡ **Sedang**
  - ⚪ **Lemah**
✅ Auto filter duplikasi sinyal  
✅ Kirim notifikasi lengkap ke Telegram  

## ⚙️ Instalasi

### 1. Clone atau Unduh
```bash
git clone https://github.com/yourusername/combo-booster-bot.git
cd combo-booster-bot
```

### 2. Instal dependensi
```bash
pip install -r requirements.txt
```

### 3. Siapkan Environment
Tambahkan variabel environment:
```bash
export TELEGRAM_TOKEN="your_bot_token_here"
export CHAT_ID="your_chat_id_here"
```

### 4. Jalankan Bot
```bash
python bot.py
```

## 📲 Contoh Pesan Telegram

```
🟢 BUY Signal
Strength: 💪 Kuat
Mode: `VWAP-MACD+Trend-MACD-RSI MTF [5m + 1h]`
Pair: `BTCUSDT` | TF: `5m & 1h`
Entry: `68250.50`
TP: `68725.25` | SL: `67900.25`
ATR: 50.00
RSI-Kernel: 61.20
MACD: 0.0023 | Signal: 0.0019
Volume: 1.45x rata-rata
EMA50: 68100.25
Time: 2025-10-29 07:30:00 UTC

_Info only — no auto order._
```

## 🕒 Konfigurasi
Ubah interval scan di dalam `bot.py`:
```python
SCAN_INTERVAL = 15  # dalam menit
```

## ☁️ Deploy ke VPS / Render / Railway

**Render / Railway:**
- Upload `bot.py` dan `requirements.txt`
- Tambahkan variabel env (`TELEGRAM_TOKEN`, `CHAT_ID`)
- Jalankan `python bot.py`

## 🧾 Lisensi
MIT License © 2025 — bebas digunakan & dimodifikasi.
