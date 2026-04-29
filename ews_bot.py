import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import requests

TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

END_DATE = datetime.today()
START_DATE = END_DATE - pd.DateOffset(years=3)

# DATA
tickers = {
    "SP500":  "^GSPC",
    "GOLD":   "GC=F",
    "BRENT":  "BZ=F",
    "EEM":    "EEM",
    "VIX":    "^VIX",
    "DXY":    "DX-Y.NYB",
}

print("Downloading data...")
raw = yf.download(
    list(tickers.values()),
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

raw.columns = list(tickers.keys())
df = raw.copy()
df = df.sort_index().ffill().bfill()
df = df.dropna()

if len(df) < 30:
    raise ValueError(f"Not enough data: {len(df)} rows")

# FEATURES
df["SP500_ret"]  = df["SP500"].pct_change()
df["BRENT_ret"]  = df["BRENT"].pct_change()

df["VIX_z"] = (
    (df["VIX"] - df["VIX"].rolling(120).mean()) /
    df["VIX"].rolling(120).std()
)

df["DXY_z"] = (
    (df["DXY"] - df["DXY"].rolling(120).mean()) /
    df["DXY"].rolling(120).std()
)

df["GOLD_z"] = (
    (df["GOLD"] - df["GOLD"].rolling(60).mean()) /
    df["GOLD"].rolling(60).std()
)

def expanding_norm(x):
    return (x - x.expanding().min()) / (
        x.expanding().max() - x.expanding().min()
    ).replace(0, np.nan)

df["RiskScore"] = (
    0.40 * expanding_norm(df["VIX_z"].clip(lower=0)) +
    0.25 * expanding_norm(df["DXY_z"].clip(lower=0)) +
    0.20 * expanding_norm(-df["SP500_ret"]) +
    0.15 * expanding_norm(df["GOLD_z"].clip(lower=0))
)

df = df.dropna(subset=["RiskScore"])

if len(df) == 0:
    raise ValueError("RiskScore is all NaN")

# LATEST
latest = df.iloc[-1]
date   = df.index[-1].strftime("%Y-%m-%d")
risk   = float(latest["RiskScore"]) * 100

if risk >= 65:
    status = "🔴 ԲԱՐՁՐ ՌԻՍԿ"
    color  = "#e74c3c"
elif risk >= 40:
    status = "🟡 ՄԻՋԻՆ ՌԻՍԿ"
    color  = "#f39c12"
else:
    status = "🟢 ՆՈՐՄԱԼ"
    color  = "#2ecc71"

print(f"Date: {date} | Risk: {risk:.1f}% | {status}")

# PLOT
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

plot_df = df["RiskScore"].tail(365)

ax.fill_between(plot_df.index, plot_df.values, alpha=0.3, color=color)
ax.plot(plot_df.index, plot_df.values, color=color, linewidth=1.5)

ax.axhline(0.65, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1)
ax.axhline(0.40, color="#f39c12", linestyle="--", alpha=0.6, linewidth=1)

ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45, color="white", fontsize=9)
plt.yticks(color="white", fontsize=9)

for spine in ax.spines.values():
    spine.set_edgecolor("#333")

ax.set_title("EWS Armenia — Composite Risk Index", color="white", fontsize=13, pad=12)
ax.set_ylabel("Risk Score", color="white", fontsize=10)

plt.tight_layout()
plt.savefig("risk.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Plot saved.")

# SEND
caption = (
    f"📊 *EWS Armenia — Daily Update*\n\n"
    f"📅 Ամսաթիվ՝ `{date}`\n"
    f"📈 Ռիսկի մակարդակ՝ `{risk:.1f}%`\n"
    f"⚡ Կարգավիճակ՝ {status}\n\n"
    f"_Աղբյուրներ՝ S\\&P500, VIX, Brent, Gold, DXY_"
)

url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

with open("risk.png", "rb") as photo:
    resp = requests.post(
        url,
        files={"photo": photo},
        data={
            "chat_id": CHAT_ID,
            "caption": caption,
            "parse_mode": "MarkdownV2"
        },
        timeout=30
    )

if resp.status_code == 200:
    print("DONE — message sent.")
else:
    print(f"ERROR: {resp.status_code} — {resp.text}")
    exit(1)
