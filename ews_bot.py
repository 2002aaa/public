# =========================
# DAILY MONITORING VERSION
# =========================

import os
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

TOKEN = os.environ.get("BOT_TOKEN", "ՔՈ_TOKEN_Ը")
CHAT_ID = os.environ.get("CHAT_ID", "@ewsarmenia")

END_DATE = datetime.today()
START_DATE = END_DATE - pd.DateOffset(years=5)

print("Daily monitoring period:")
print(START_DATE.date(), "→", END_DATE.date())

# =========================
# LOAD DATA
# =========================

tickers = {
    "SP500": "^GSPC",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "EEM": "EEM",
    "RUB_USD": "RUB=X"
}

yf_raw = yf.download(
    list(tickers.values()),
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)

yf_data = yf_raw["Close"].copy()
yf_data.columns = list(tickers.keys())

print("Yahoo shape:", yf_data.shape)

fred_series = ["DGS2", "DGS5", "DGS10", "VIXCLS"]
fred_data = pd.DataFrame()

for s in fred_series:
    try:
        fred_data[s] = pdr.DataReader(s, "fred", START_DATE, END_DATE)
    except Exception as e:
        print(f"{s} error:", e)

print("FRED shape:", fred_data.shape)

# =========================
# MERGE
# =========================

df = yf_data.join(fred_data, how="outer")
df = df.sort_index().ffill().bfill()

print("Merged shape:", df.shape)

# =========================
# FEATURE ENGINEERING
# =========================

df["SP500_ret"] = df["SP500"].pct_change()
df["BRENT_ret"] = df["BRENT"].pct_change()

df["FX_vol_30"] = df["RUB_USD"].pct_change().rolling(30).std()
df["SP500_vol_30"] = df["SP500_ret"].rolling(30).std()

df["VIX_z"] = (df["VIXCLS"] - df["VIXCLS"].rolling(120).mean()) / df["VIXCLS"].rolling(120).std()
df["FX_vol_z"] = (df["FX_vol_30"] - df["FX_vol_30"].rolling(120).mean()) / df["FX_vol_30"].rolling(120).std()

df["Yield_spread"] = df["DGS10"] - df["DGS2"]
df["Yield_inversion"] = (-df["Yield_spread"]).clip(lower=0)

# =========================
# NORMALIZATION
# =========================

def expanding_minmax(series):
    min_s = series.expanding().min()
    max_s = series.expanding().max()
    denom = (max_s - min_s).replace(0, np.nan)
    return (series - min_s) / denom

df["VIX_norm"] = expanding_minmax(df["VIX_z"])
df["FX_norm"] = expanding_minmax(df["FX_vol_z"])
df["Yield_norm"] = expanding_minmax(df["Yield_inversion"])
df["SP500_norm"] = expanding_minmax(-df["SP500_ret"])
df["BRENT_norm"] = expanding_minmax(-df["BRENT_ret"])

# =========================
# RISK SCORE
# =========================

df["RiskScore"] = (
    0.30 * df["VIX_norm"] +
    0.25 * df["FX_norm"] +
    0.20 * df["Yield_norm"] +
    0.15 * df["SP500_norm"] +
    0.10 * df["BRENT_norm"]
)

df = df.dropna()

print("Final df:", df.shape)
print(df[["RiskScore"]].tail())

# =========================
# LATEST RESULT
# =========================

latest = df.iloc[-1]

report_date = latest.name.strftime("%Y-%m-%d")
risk = latest["RiskScore"] * 100
status = "STRESS 🔴" if latest["RiskScore"] > 0.5 else "NORMAL 🟢"

print("Latest:", report_date, f"{risk:.1f}%")

# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["RiskScore"], label="Risk Score")
plt.axhline(df["RiskScore"].mean(), linestyle="--", label="Mean")
plt.title("EWS Armenia Risk Score")
plt.legend()
plt.tight_layout()
plt.savefig("risk_chart.png", dpi=200)
plt.close()

# =========================
# TELEGRAM SEND
# =========================

caption = f"""
📊 EWS Armenia — Daily Risk Assessment

📅 Date: {report_date}
📈 Risk Level: {risk:.1f}%
🚨 Status: {status}
⏱ Forecast Horizon: t+2 days
"""

url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

with open("risk_chart.png", "rb") as photo:
    response = requests.post(
        url_photo,
        files={"photo": photo},
        data={
            "chat_id": CHAT_ID,
            "caption": caption
        }
    )

print("Telegram response:", response.status_code, response.text)
response.raise_for_status()
