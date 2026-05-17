"""Generated from Jupyter notebook: Cointegration_Analysis_Gold_Complete

Magics and shell lines are commented out. Run with a normal Python interpreter."""

# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# Download data
symbols = ["HO=F", "CL=F"]  # Gold Futures and Newmont
df = yf.download(symbols, start="2015-01-01", end="2024-12-31")["Close"]
df.dropna(inplace=True)
df.columns = ["Gold", "NEM"]

# Plot
df.plot(title="Gold vs Newmont")
plt.ylabel("Price")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("cointegration_series.png")
plt.show()

def adf_test(series, name):
    result = adfuller(series)
    print(f"{name} ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")

adf_test(df["Gold"], "Gold")
adf_test(df["NEM"], "NEM")

# Now first difference
adf_test(df["Gold"].diff().dropna(), "Gold Δ")
adf_test(df["NEM"].diff().dropna(), "NEM Δ")

jres = coint_johansen(df, det_order=0, k_ar_diff=1)
trace_stat = jres.lr1
crit_values = jres.cvt

print("Trace Statistic:", trace_stat)
print("Critical Values (90%, 95%, 99%):\n", crit_values)

vecm_model = VECM(df, k_ar_diff=1, coint_rank=1)
vecm_res = vecm_model.fit()
print(vecm_res.summary())

forecast = vecm_res.predict(steps=12)
forecast_df = pd.DataFrame(
    forecast,
    columns=["Gold", "NEM"],
    index=pd.date_range(df.index[-1], periods=12, freq="M"),
)

# Plot forecasts
df[-100:].plot(figsize=(10, 5), label="Historical")
forecast_df.plot(ax=plt.gca(), style="--")
plt.title("12-Month VECM Forecast")
plt.savefig("vecm_forecast.png")
plt.show()

# First difference to make series stationary
diff_df = df.diff().dropna()

from statsmodels.tsa.api import VAR

# Fit VAR model
model = VAR(diff_df)
var_res = model.fit(maxlags=1)
print(var_res.summary())

# IRF up to 12 steps
irf = var_res.irf(12)

# Plot standard IRFs
irf.plot(orth=False)
plt.suptitle("VAR IRFs (Standard)")
plt.savefig("var_irf_standard.png")
plt.show()

# Orthogonalized IRFs
irf.plot(orth=True)
plt.suptitle("VAR IRFs (Orthogonalized)")
plt.savefig("var_irf_orthogonalized.png")
plt.show()

# Forecast Error Variance Decomposition
fevd = var_res.fevd(12)
fevd.plot()
plt.suptitle("Forecast Error Variance Decomposition (FEVD)")
plt.savefig("var_fevd.png")
plt.show()

# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

def fetch_data(symbols, start="2015-01-01", end="2024-12-31"):
    df = yf.download(symbols, start=start, end=end)["Close"].dropna()
    df.columns = ["HeatingOil", "CrudeOil"]
    return df

def plot_series(df, filename="cointegration_series.png"):
    df.plot(title="Heating Oil vs Crude Oil Prices")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def adf_summary(df):
    for col in df.columns:
        result = adfuller(df[col])
        print(f"{col} ADF: {result[0]:.4f}, p={result[1]:.4f}")
        result_diff = adfuller(df[col].diff().dropna())
        print(f"{col} ΔADF: {result_diff[0]:.4f}, p={result_diff[1]:.4f}")

def johansen_test(df):
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    print("Trace Statistic:", result.lr1)
    print("Critical Values (90%, 95%, 99%):\n", result.cvt)
    return result

def fit_vecm(df, lags=1, rank=1):
    model = VECM(df, k_ar_diff=lags, coint_rank=rank)
    res = model.fit()
    print(res.summary())
    return res

def forecast_vecm(res, df, steps=12):
    forecast = res.predict(steps=steps)
    future_idx = pd.date_range(df.index[-1], periods=steps, freq="ME")
    forecast_df = pd.DataFrame(forecast, columns=df.columns, index=future_idx)

    df[-100:].plot(figsize=(10, 5), label="Historical")
    forecast_df.plot(ax=plt.gca(), style="--")
    plt.title("12-Month VECM Forecast")
    plt.savefig("vecm_forecast.png")
    plt.show()

def run_var_irf_fevd(df_diff, lags=1, horizon=12):
    model = VAR(df_diff)
    res = model.fit(maxlags=lags)
    print(res.summary())

    irf = res.irf(horizon)
    irf.plot(orth=False)
    plt.suptitle("Impulse Response (Standard)")
    plt.savefig("var_irf_standard.png")
    plt.show()

    irf.plot(orth=True)
    plt.suptitle("Impulse Response (Orthogonalized)")
    plt.savefig("var_irf_orthogonalized.png")
    plt.show()

    fevd = res.fevd(horizon)
    fevd.plot()
    plt.suptitle("Forecast Error Variance Decomposition (FEVD)")
    plt.savefig("var_fevd.png")
    plt.show()

def main():
    symbols = ["HO=F", "CL=F"]  # Heating Oil and Crude Oil
    df = fetch_data(symbols)
    plot_series(df)
    adf_summary(df)

    johansen_test(df)
    vecm_res = fit_vecm(df, lags=1, rank=1)
    forecast_vecm(vecm_res, df)

    df_diff = df.diff().dropna()
    run_var_irf_fevd(df_diff, lags=1, horizon=12)

if __name__ == "__main__":
    main()

# --- code cell ---

# --- code cell ---

forecast_vecm(vecm_res, df)

# --- code cell ---

forecast_vecm(vecm_res, df)
