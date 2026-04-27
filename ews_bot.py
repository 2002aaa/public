import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from pandas_datareader import data as pdr

TOKEN = "ՔՈ_TOKEN_Ը"
CHAT_ID = "@ewsarmenia"

END_DATE = datetime.today()
START_DATE = END_DATE - pd.DateOffset(years=5)

# DATA
tickers = {
    "SP500": "^GSPC",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "EEM": "EEM",
    "RUB_USD": "RUB=X"
}

yf_data = yf.download(list(tickers.values()), start=START_DATE, end=END_DATE)
yf_data = yf_data["Close"]
yf_data.columns = tickers.keys()

fred_series = ["DGS2", "DGS10", "VIXCLS"]
fred_data = pd.DataFrame()

for s in fred_series:
    try:
        fred_data[s] = pdr.DataReader(s, "fred", START_DATE, END_DATE)
    except:
        pass

df = yf_data.join(fred_data, how="outer")
df = df.sort_index().ffill().bfill()

# FEATURES
df["SP500_ret"] = df["SP500"].pct_change()
df["BRENT_ret"] = df["BRENT"].pct_change()

df["VIX_z"] = (df["VIXCLS"] - df["VIXCLS"].rolling(120).mean()) / df["VIXCLS"].rolling(120).std()
df["Yield_spread"] = df["DGS10"] - df["DGS2"]
df["Yield_inversion"] = (-df["Yield_spread"]).clip(lower=0)

def norm(x):
    return (x - x.expanding().min()) / (x.expanding().max() - x.expanding().min())

df["RiskScore"] = (
    0.4 * norm(df["VIX_z"]) +
    0.3 * norm(df["Yield_inversion"]) +
    0.3 * norm(-df["SP500_ret"])
)

df = df.dropna()

# LATEST
latest = df.iloc[-1]
date = latest.name.strftime("%Y-%m-%d")
risk = latest["RiskScore"] * 100
status = "STRESS 🔴" if latest["RiskScore"] > 0.5 else "NORMAL 🟢"

# PLOT
plt.figure(figsize=(10,5))
plt.plot(df.index, df["RiskScore"])
plt.axhline(df["RiskScore"].mean(), linestyle="--")
plt.savefig("risk.png")
plt.close()

# SEND
url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

caption = f"""
📊 EWS Armenia

📅 {date}
📈 Risk: {risk:.1f}%
🚨 {status}
"""

with open("risk.png", "rb") as photo:
    requests.post(url, files={"photo": photo}, data={
        "chat_id": CHAT_ID,
        "caption": caption
    })

print("DONE")
