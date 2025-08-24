import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("환경변수 BOT_TOKEN / CHAT_ID 필요")

SYMBOL       = "BTCUSD"
CONTRACT_ID  = "10000001"
TIMEFRAME    = "5m"
POLL_SEC     = 5

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_PERIOD, RSI_FILTER            = 14, 50.0
STOCH_RSI_PERIOD                  = 14
STOCH_K_SMOOTH, STOCH_D_SMOOTH    = 3, 3
K_OVERSOLD, K_OVERBOUGHT          = 20.0, 80.0

HTTP_BASE = "https://pro.edgex.exchange"
TF_MAP = {
    "1m":"MINUTE_1","3m":"MINUTE_3","5m":"MINUTE_5","15m":"MINUTE_15","30m":"MINUTE_30",
    "1h":"HOUR_1","2h":"HOUR_2","4h":"HOUR_4","6h":"HOUR_6","8h":"HOUR_8","12h":"HOUR_12","1d":"DAY_1",
}

def ts_local(ms:int) -> str:
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def tg_send(text:str):
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10).raise_for_status()
    except Exception as e:
        print("텔레그램 오류:", e)

def fetch_kline(contract_id:str, kline_type:str, size:int=400) -> pd.DataFrame:
    try:
        r = requests.get(f"{HTTP_BASE}/api/v1/public/quote/getKline",
                         params={"contractId": contract_id, "priceType":"LAST_PRICE",
                                 "klineType": kline_type, "size": str(size)}, timeout=15)
        j = r.json() or {}
        rows = (j.get("data") or {}).get("dataList") or []
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ["open","high","low","close","size","klineTime"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("klineTime").reset_index(drop=True)
        return pd.DataFrame({
            "timestamp": df["klineTime"],
            "open": df["open"], "high": df["high"], "low": df["low"],
            "close": df["close"], "volume": df["size"]
        })
    except Exception as e:
        print("fetch_kline 오류:", e)
        return pd.DataFrame()

def ema(s: pd.Series, span:int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, period:int=14) -> pd.Series:
    d   = s.diff()
    up  = d.where(d>0, 0.0)
    dn  = (-d).where(d<0, 0.0)
    avg_up = up.rolling(period).mean()
    avg_dn = dn.rolling(period).mean()
    rs  = avg_up / avg_dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.ffill().fillna(50.0)

def macd(s: pd.Series, fast:int, slow:int, signal:int):
    m  = ema(s, fast) - ema(s, slow)
    sg = ema(m, signal)
    return m, sg, (m - sg)

def stoch_rsi_from_rsi(rsi_s: pd.Series, period:int, k_s:int, d_s:int):
    min_r = rsi_s.rolling(period).min()
    max_r = rsi_s.rolling(period).max()
    denom = (max_r - min_r).replace(0, np.nan)
    stoch = ((rsi_s - min_r) / denom) * 100.0
    k = stoch.rolling(k_s).mean().ffill().fillna(50.0)
    d = k.rolling(d_s).mean().ffill().fillna(50.0)
    return k, d

def check_signal(df: pd.DataFrame):
    if df.empty or len(df) < max(MACD_SLOW, RSI_PERIOD) + 5:
        return None, None
    close = df["close"]
    rsi14 = rsi(close, RSI_PERIOD)
    macd_line, sig_line, _ = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    k, d = stoch_rsi_from_rsi(rsi14, STOCH_RSI_PERIOD, STOCH_K_SMOOTH, STOCH_D_SMOOTH)

    di = df.copy()
    di["rsi"] = rsi14; di["macd"] = macd_line; di["sig"] = sig_line
    di["k"] = k; di["d"] = d

    prev = di.iloc[-2]
    curr = di.iloc[-1]

    macd_up   = (prev["macd"] <= prev["sig"]) and (curr["macd"] >  curr["sig"])
    macd_down = (prev["macd"] >= prev["sig"]) and (curr["macd"] <  curr["sig"])
    k_up      = (prev["k"]   <= prev["d"])    and (curr["k"]   >   curr["d"])
    k_down    = (prev["k"]   >= prev["d"])    and (curr["k"]   <   curr["d"])
    from_os   = (prev["k"] < K_OVERSOLD)
    from_ob   = (prev["k"] > K_OVERBOUGHT)

    long_ok  = macd_up   and k_up   and from_os and (curr["rsi"] >= RSI_FILTER)
    short_ok = macd_down and k_down and from_ob and (curr["rsi"] <= RSI_FILTER)

    info = {
        "ts": int(curr["timestamp"]),
        "px": float(curr["close"]),
        "macd": float(curr["macd"]), "sig": float(curr["sig"]),
        "k": float(curr["k"]), "d": float(curr["d"]),
        "rsi": float(curr["rsi"])
    }

    if long_ok:  return "LONG", info
    if short_ok: return "SHORT", info
    return None, info

def main():
    kline_type = TF_MAP[TIMEFRAME]
    last_alerted_candle = set()
    while True:
        df = fetch_kline(CONTRACT_ID, kline_type, size=400)
        if df.empty:
            time.sleep(POLL_SEC); continue
        signal, info = check_signal(df)
        if signal and info and info["ts"] not in last_alerted_candle:
            side = "롱" if signal == "LONG" else "숏"
            msg = (f"[{side} 신호] {SYMBOL} {TIMEFRAME}\n"
                   f"- 가격: {info['px']}\n"
                   f"- MACD: {info['macd']:.4f} vs {info['sig']:.4f}\n"
                   f"- StochRSI K/D: {info['k']:.1f}/{info['d']:.1f}\n"
                   f"- RSI(14): {info['rsi']:.1f}\n"
                   f"- 시간: {ts_local(info['ts'])}")
            print("[ALERT]", msg.replace("\n", " | "))
            tg_send(msg)
            last_alerted_candle.add(info["ts"])
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
