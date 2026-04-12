import pandas as pd
import matplotlib.pyplot as plt

from hmm_strategy import calculate_rolling_hurst
from data_loader import load_all_data
import config
import os
import numpy as np
import seaborn as sns

def plot_yearly_returns():
    data = {
        "Year": list(range(2012, 2026)),

        "Benchmark": [
            49.52,45.46,86.46,57.74,65.99,81.61,54.73,93.21,91.49,
            73.54,69.55,79.19,114.51,113.98
        ],

        "HMM_Standard": [
            80.87,60.79,49.80,93.12,65.73,74.65,50.61,104.11,97.69,
            89.04,88.06,92.61,108.25,98.78
        ],

        "HMM_Elder": [
            49.72,79.39,81.58,67.92,97.22,77.43,69.19,111.62,95.73,
            98.41,94.59,92.73,111.83,103.17
        ],

        "HMM_VaR": [
            62.35,89.79,60.82,81.71,72.26,75.28,61.50,115.88,100.44,
            106.95,102.40,99.06,111.40,104.19
        ]
    }

    df = pd.DataFrame(data)
    for col in ["Benchmark","HMM_Standard","HMM_Elder","HMM_VaR"]:
        df[col] = df[col] / 100 - 1
    
    x = np.arange(len(df["Year"]))
    width = 0.2

    plt.figure(figsize=(14,6))

    plt.bar(x - 1.5*width, df["Benchmark"], width, label="Benchmark")
    plt.bar(x - 0.5*width, df["HMM_Standard"], width, label="HMM_Standard")
    plt.bar(x + 0.5*width, df["HMM_Elder"], width, label="HMM_Elder")
    plt.bar(x + 1.5*width, df["HMM_VaR"], width, label="HMM_VaR")

    plt.xticks(x, df["Year"], rotation=45)
    plt.axhline(0, linestyle='--')

    plt.title("Annual Returns by Strategy (Bar Chart)")
    plt.xlabel("Year")
    plt.ylabel("Return")

    plt.legend()
    plt.grid(axis='y')
    plt.show()


def plot_backtest_graphs():
    data = {
        "Start": [2012,2013,2014,2015,2016,2017,2018,2019,2020],

        "Benchmark_Final": [121.76,121.44,143.62,101.92,115.99,115.84,99.80,128.06,104.25],
        "HMM_Standard_Final": [65.98,68.31,79.73,64.09,77.45,75.92,76.79,120.77,106.73],
        "HMM_Elder_Final": [84.04,87.61,82.72,84.32,88.73,99.36,103.02,129.67,110.49],
        "HMM_VaR_Final": [144.05,130.53,106.36,102.13,118.01,113.32,116.14,168.37,140.81],

        "Benchmark_Sharpe": [0.09,0.11,0.14,0.01,0.08,0.07,0.04,0.18,-0.03],
        "HMM_Standard_Sharpe": [-0.06,-0.04,0.01,-0.25,-0.10,-0.19,-0.12,0.17,-0.14],
        "HMM_Elder_Sharpe": [0.00,-0.09,-0.06,-0.17,-0.15,-0.08,-0.02,0.30,-0.01],
        "HMM_VaR_Sharpe": [0.14,0.08,0.02,-0.02,0.07,0.05,0.10,0.86,0.59],

        "Benchmark_MaxDD": [56.90,54.46,51.24,48.69,42.32,35.03,42.53,32.57,33.08],
        "HMM_Standard_MaxDD": [78.50,78.47,78.51,79.61,73.15,73.31,73.25,63.99,63.99],
        "HMM_Elder_MaxDD": [66.84,56.69,53.64,58.97,43.65,43.19,33.78,19.33,18.72],
        "HMM_VaR_MaxDD": [65.50,65.26,65.49,74.29,53.12,51.41,42.82,25.32,25.32],
    }

    df = pd.DataFrame(data)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10,6))

    for col in ["Benchmark_Final","HMM_Standard_Final","HMM_Elder_Final","HMM_VaR_Final"]:
        plt.plot(df["Start"], df[col], marker='o', label=col.replace("_Final",""))

    plt.title("Final Portfolio Value vs Start Year")
    plt.xlabel("Start Year")
    plt.ylabel("Final Value (Million)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./backtest_result/FinalValue_over_time.png")
    plt.close()
    
    plt.figure(figsize=(10,6))
    for col in ["Benchmark_Sharpe","HMM_Standard_Sharpe","HMM_Elder_Sharpe","HMM_VaR_Sharpe"]:
        plt.plot(df["Start"], df[col], marker='o', label=col.replace("_Sharpe",""))

    plt.axhline(0, linestyle='--')
    plt.title("Sharpe Ratio vs Start Year")
    plt.xlabel("Start Year")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig("./backtest_result/SharpeRatio_over_time.png")
    plt.close()

    plt.figure(figsize=(10,6))
    for col in ["Benchmark_MaxDD","HMM_Standard_MaxDD","HMM_Elder_MaxDD","HMM_VaR_MaxDD"]:
        plt.plot(df["Start"], df[col], marker='o', label=col.replace("_MaxDD",""))

    plt.title("Maximum Drawdown vs Start Year")
    plt.xlabel("Start Year")
    plt.ylabel("Max Drawdown (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("./backtest_result/MaxDD_over_time.png")
    plt.close()

    sharpe_df = df.set_index("Start")[[
        "Benchmark_Sharpe","HMM_Standard_Sharpe","HMM_Elder_Sharpe","HMM_VaR_Sharpe"
    ]]

    plt.figure(figsize=(10, 6))
    sns.heatmap(sharpe_df, annot=True, cmap="coolwarm", center=0)

    plt.title("Sharpe Ratio Heatmap")
    plt.ylabel("Start Year")
    plt.savefig("Sharpe_heatmap.png")
    plt.close()


def plot_hurst_windows(ticker):
    data = load_all_data(config.DATA_DIR)
    df_prices = data["adjusted"]
    df_prices = df_prices[df_prices.index >= pd.to_datetime("2020-01-01")]
    tickers_csv_path = os.path.join(config.DATA_DIR, "tickers.csv")

    tickers = pd.read_csv(tickers_csv_path, header=None)
    tickers.columns = ["ticker", "gics"]
    tickers["sector"] = tickers["gics"].apply(
        lambda x: str(x)[:2] if x is not None else None
    )
    tickers["industry"] = tickers["gics"].apply(
        lambda x: str(x)[:6] if x is not None else None
    )

    gics_map = tickers.set_index("ticker")["sector"]

    log_return = np.log(df_prices / df_prices.shift(1))
    log_return_T = log_return.T
    log_return_T["sector"] = gics_map

    log_return_neutral_T = log_return_T.groupby("sector").transform(
        lambda x: x - x.mean()
    )

    df_log_return_neutral = log_return_neutral_T.T

    price_series = df_prices[ticker]
    log_return_series = df_log_return_neutral[ticker]

    valid_price = price_series.dropna()
    valid_log_return = log_return_series.dropna()

    df = pd.DataFrame({"Log_Return": valid_log_return, "Close": valid_price})
    print(df)
    df.fillna(0, inplace=True)

    # Micro: Hurst Exponent
    df["Hurst"] = calculate_rolling_hurst(df["Log_Return"], window=200, q=2)
    
    # =========================
    # 📊 PLOTTING (3 PANELS)
    # =========================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # --- 1. Price ---
    axes[0].plot(df.index, df["Close"])
    axes[0].set_title(f"{ticker} Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(True)

    # --- 2. Log Returns ---
    axes[1].plot(df.index, df["Log_Return"])
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Log Returns (Sector Neutral)")
    axes[1].set_ylabel("Return")
    axes[1].grid(True)

    # --- 3. Hurst ---
    axes[2].plot(df.index, df["Hurst"], label="Hurst (200d)")
    axes[2].axhline(config.HURST_UPPER, linestyle="--", color='red', label="Upper")
    axes[2].axhline(config.HURST_LOWER, linestyle="--", color='green', label="Lower")
    axes[2].axhline(0.5, linestyle=":", color='magenta', label="Random Walk")
    axes[2].set_title("Hurst Exponent")
    axes[2].set_ylabel("H")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True)

    axes[2].fill_between(
    df.index,
        config.HURST_UPPER,
        1,
        where=df["Hurst"] > config.HURST_UPPER,
        alpha=0.2,
        label="Trending Regime"
    )

    axes[2].fill_between(
        df.index,
        0,
        config.HURST_LOWER,
        where=df["Hurst"] < config.HURST_LOWER,
        alpha=0.2,
        label="Mean-Reverting Regime"
    )

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"hurst_full_{ticker}.png", dpi=300)

    
if __name__ == "__main__":
    # plot_hurst_windows("000031")
    plot_backtest_graphs()
    # plot_yearly_returns()