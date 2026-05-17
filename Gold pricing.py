"""Generated from Jupyter notebook: Gold pricing

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    print(f"{name} is {('stationary' if result[1] <= 0.05 else 'NOT stationary')}.")


def get_data():
    start_date = datetime.now() - pd.DateOffset(years=5)
    end_date = datetime.now()
    nem = yf.download("NEM", start=start_date, end=end_date, auto_adjust=False)[
        ["Close"]
    ]
    gold = yf.download("GC=F", start=start_date, end=end_date, auto_adjust=False)[
        ["Close"]
    ]
    gdx = yf.download("GDX", start=start_date, end=end_date, auto_adjust=False)[
        ["Close"]
    ]
    nem.columns = ["NEM"]
    gold.columns = ["Gold"]
    gdx.columns = ["GDX"]
    df = pd.concat([nem, gold, gdx], axis=1).dropna()
    return df


def plot_fevd_stacked(fevd, target_var, filename):
    idx = fitted_model.names.index(target_var)
    decomp = fevd.decomp[:, idx, :]
    months = np.arange(1, decomp.shape[0] + 1)
    plt.figure(figsize=(10, 6))
    plt.stackplot(months, decomp.T, labels=fitted_model.names)
    plt.title(f"FEVD - Stacked Area Chart: {target_var}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Fraction of Variance")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main() -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(data.index, np.log(data["GDX"]), label="Newmont (NEM)")

    plt.plot(data.index, np.log(data["Gold"]), label="Gold (GC=F)")

    plt.title("Newmont vs. Gold Prices")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.grid()

    plt.show()

    print("\nStationarity Tests:")

    check_stationarity(data["GDX"], "Newmont")

    check_stationarity(data["Gold"], "Gold")

    data_diff = np.log(data).diff().dropna()

    plt.figure(figsize=(12, 6))

    data_diff.plot(title="Log Returns: NEM and Gold")

    plt.xlabel("Date")

    plt.ylabel("Log Returns")

    plt.grid()

    plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )

    df = get_data()

    monthly_data = df.resample("ME").mean()

    monthly_log_returns = np.log(monthly_data).diff().dropna()

    print("\nMonthly Stationarity Tests:")

    for col in monthly_log_returns.columns:
        check_stationarity(monthly_log_returns[col], col)

    print("\nGranger Causality Test (Gold → GDX):")

    grangercausalitytests(monthly_log_returns[["GDX", "Gold"]], maxlag=3, verbose=True)

    model = VAR(monthly_log_returns)

    lag_selection = model.select_order(12)

    selected_lag = lag_selection.aic

    fitted_model = model.fit(selected_lag)

    print("\nVAR Model Summary:")

    print(fitted_model.summary())

    irf = fitted_model.irf(12)

    fig = irf.plot(impulse="Gold", response="GDX")

    plt.suptitle("Impulse Response: Monthly Gold Shock → GDX", fontsize=14)

    plt.tight_layout()

    plt.savefig("irf_gold_to_gdx.png")

    plt.close()

    fevd = fitted_model.fevd(12)

    plot_fevd_stacked(fevd, "GDX", "fevd_gdx.png")

    plot_fevd_stacked(fevd, "Gold", "fevd_gold.png")

    plot_fevd_stacked(fevd, "NEM", "fevd_nem.png")

    orth_irfs = irf.orth_irfs

    cumulative_irfs = np.cumsum(orth_irfs, axis=0)

    months = np.arange(1, cumulative_irfs.shape[0] + 1)

    impulse_idx = fitted_model.names.index("Gold")

    for i, response_var in enumerate(fitted_model.names):
        plt.figure(figsize=(10, 6))
        plt.plot(
            months,
            cumulative_irfs[:, i, impulse_idx],
            label=f"Gold → {response_var}",
            color="blue",
        )
        plt.axhline(0, linestyle="--", linewidth=0.7, color="black")
        plt.title(f"Cumulative IRF: Shock to Gold → {response_var}")
        plt.xlabel("Months Ahead")
        plt.ylabel("Cumulative Response")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"cumulative_irf_gold_to_{response_var.lower()}.png")
        plt.close()


if __name__ == "__main__":
    main()
