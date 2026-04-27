# =========================
# DAILY MONITORING VERSION
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

TOKEN = "8785731071:AAGBTF-jvQtaj4RzOOqPpMHV1YHnIuVnfZY"
CHAT_ID = "@ewsarmenia"

END_DATE = datetime.today()
START_DATE = END_DATE - pd.DateOffset(years=5)

print("Daily monitoring period:")
print(START_DATE.date(), "→", END_DATE.date())# =========================
# LOAD DAILY DATA
# =========================

import yfinance as yf
from pandas_datareader import data as pdr

# Yahoo Finance
tickers = {
    "SP500": "^GSPC",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "EEM": "EEM",
    "RUB_USD": "RUB=X"
}

yf_data = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE)

# rename columns
yf_data = yf_data["Close"]
yf_data.columns = tickers.keys()

print("Yahoo shape:", yf_data.shape)


# FRED տվյալներ
fred_series = ["DGS2", "DGS5", "DGS10", "VIXCLS"]

fred_data = pd.DataFrame()

for s in fred_series:
    try:
        fred_data[s] = pdr.DataReader(s, "fred", START_DATE, END_DATE)
    except:
        print(f"{s} error")

print("FRED shape:", fred_data.shape)


# =========================
# MERGE
# =========================

df = yf_data.join(fred_data, how="outer")

# fill missing
df = df.sort_index().ffill().bfill()

print("Merged shape:", df.shape)
df.tail()# =========================
# FEATURE ENGINEERING
# =========================

# returns
df["SP500_ret"] = df["SP500"].pct_change()
df["BRENT_ret"] = df["BRENT"].pct_change()

# FX volatility (RUB/USD որպես proxy)
df["FX_vol_30"] = df["RUB_USD"].pct_change().rolling(30).std()

# S&P volatility
df["SP500_vol_30"] = df["SP500_ret"].rolling(30).std()

# VIX z-score
df["VIX_z"] = (df["VIXCLS"] - df["VIXCLS"].rolling(120).mean()) / df["VIXCLS"].rolling(120).std()

# FX volatility z-score
df["FX_vol_z"] = (df["FX_vol_30"] - df["FX_vol_30"].rolling(120).mean()) / df["FX_vol_30"].rolling(120).std()

# Yield spread
df["Yield_spread"] = df["DGS10"] - df["DGS2"]

# inversion pressure (եթե բացասական է)
df["Yield_inversion"] = (-df["Yield_spread"]).clip(lower=0)

# =========================
# NORMALIZATION (expanding)
# =========================

def expanding_minmax(series):
    return (series - series.expanding().min()) / (series.expanding().max() - series.expanding().min())

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
df[["RiskScore"]].tail()import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(df.index, df["RiskScore"], label="Risk Score")
plt.axhline(df["RiskScore"].mean(), linestyle="--", label="Mean")

plt.title("EWS Armenia Risk Score")
plt.legend()

# save
plt.savefig("risk_chart.png")
plt.close()url_photo = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

with open("risk_chart.png", "rb") as photo:
    requests.post(url_photo, files={"photo": photo}, data={
        "chat_id": CHAT_ID
    })caption = f"""
📊 EWS Armenia — Daily Risk Assessment

📅 Date: {report_date}
📈 Risk Level: {risk:.1f}%
🚨 Status: {status}
⏱ Forecast Horizon: t+2 days
"""

with open("risk_chart.png", "rb") as photo:
    requests.post(url_photo, files={"photo": photo}, data={
        "chat_id": CHAT_ID,
        "caption": caption
    })
