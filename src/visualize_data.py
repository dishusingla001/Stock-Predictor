import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "data/raw/RELIANCE.csv"
out_path = "data/processed/reliance_stock.png"

data = pd.read_csv(csv_path, header=None)

# The raw file has 3 metadata rows before the actual dataset starts:
# 1) column names, 2) ticker row, 3) blank Date row
# We keep the actual stock table and set real column names.

data = data.iloc[3:].reset_index(drop=True)
data.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

data = data.dropna(subset=["Date"]).copy()
data["Date"] = pd.to_datetime(data["Date"])

plt.figure(figsize=(12, 6))
plt.plot(data["Date"], data["Close"], color="tab:blue")

plt.title("Reliance Stock Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"Chart saved to: {out_path}")