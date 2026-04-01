import logging

logger = logging.getLogger(__name__)

# Extracted code from '12_Multivariate-Time-Series-VAR.md'
# Blocks appear in the same order as in the markdown article.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Load multiple datasets from the local data folder
use_df = pd.read_csv(BASE_DIR / "data" / "use_OK.csv")
pr_df = pd.read_csv(BASE_DIR / "data" / "pr_OK.csv")
co2_df = pd.read_csv(BASE_DIR / "data" / "co2_OK.csv")


def aggregate_years(df: pd.DataFrame) -> pd.Series:
    """Aggregate SEDS-style table across all MSN codes into a yearly series."""
    year_cols = [col for col in df.columns if col.isdigit()]
    year_totals = df[year_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    return pd.Series(
        data=year_totals.values,
        index=pd.to_datetime(year_totals.index, format="%Y"),
    ).sort_index()


# Prepare all series
use_series = aggregate_years(use_df)
pr_series = aggregate_years(pr_df)
co2_series = aggregate_years(co2_df)

# Align indices
common_years = use_series.index.intersection(pr_series.index).intersection(co2_series.index)
use_series = use_series.loc[common_years]
pr_series = pr_series.loc[common_years]
co2_series = co2_series.loc[common_years]

# Create multivariate dataframe
var_data = pd.DataFrame({
    'consumption': use_series,
    'production': pr_series,
    'emissions': co2_series
})

logger.info(f"Multivariate series length: {len(var_data)}")
logger.info(f"Date range: {var_data.index.min()} to {var_data.index.max()}")
logger.info(f"\nSeries statistics:")
logger.info(var_data.describe())

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

# Check stationarity (VAR requires stationary data)
def check_stationarity(series, name):
    result = adfuller(series.dropna())
    logger.info(f"{name}: ADF statistic={result[0]:.4f}, p-value={result[1]:.4f}")
    return result[1] < 0.05

logger.info("Stationarity tests:")
for col in var_data.columns:
    is_stationary = check_stationarity(var_data[col], col)

# If not stationary, difference
var_data_diff = var_data.diff().dropna()

# Fit VAR model
var_model = VAR(var_data_diff)
var_fitted = var_model.fit(maxlags=4, ic='aic')

logger.info(f"\nVAR Model Summary:")
logger.info(f"Selected lag order: {var_fitted.k_ar}")
logger.info(f"\n{var_fitted.summary()}")

# Forecast
forecast_steps = 10
forecast = var_fitted.forecast(var_data_diff.values[-var_fitted.k_ar:], steps=forecast_steps)

# Convert back to levels (cumulative sum)
forecast_levels = var_data.iloc[-1:].values + np.cumsum(forecast, axis=0)

logger.info(f"\nForecast shape: {forecast_levels.shape}")

from statsmodels.tsa.stattools import coint

# Test cointegration between pairs
logger.info("Cointegration tests:")
pairs = [
    ('consumption', 'production'),
    ('consumption', 'emissions'),
    ('production', 'emissions')
]

for var1, var2 in pairs:
    score, pvalue, _ = coint(var_data[var1], var_data[var2])
    logger.info(f"{var1} vs {var2}: p-value={pvalue:.4f} {'(cointegrated)' if pvalue < 0.05 else '(not cointegrated)'}")

# Impulse response function
irf = var_fitted.irf(10)  # 10 periods ahead

# Plot impulse responses
irf.plot(figsize=(12, 10))
plt.tight_layout()
plt.savefig('var_impulse_response.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze specific responses
logger.info("\nImpulse Response Analysis:")
logger.info("Shock in consumption affects:")
logger.info(f"  Production after 1 period: {irf.irfs[1, 1, 0]:.4f}")
logger.info(f"  Emissions after 1 period: {irf.irfs[1, 2, 0]:.4f}")

# Granger causality tests
gc = var_fitted.test_causality('consumption', 'production', kind='f')
logger.info(f"\nGranger Causality: consumption -> production")
logger.info(f"F-statistic: {gc.test_statistic:.4f}, p-value: {gc.pvalue:.4f}")
logger.info(f"Causal: {'Yes' if gc.pvalue < 0.05 else 'No'}")

gc2 = var_fitted.test_causality('production', 'emissions', kind='f')
logger.info(f"\nGranger Causality: production -> emissions")
logger.info(f"F-statistic: {gc2.test_statistic:.4f}, p-value: {gc2.pvalue:.4f}")
logger.info(f"Causal: {'Yes' if gc2.pvalue < 0.05 else 'No'}")

# Create forecast dataframe
forecast_dates = pd.date_range(start=var_data.index[-1] + pd.DateOffset(years=1), 
                               periods=forecast_steps, freq='YS')
forecast_df = pd.DataFrame(forecast_levels, 
                          index=forecast_dates,
                          columns=var_data.columns)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for i, col in enumerate(var_data.columns):
    # Historical
    axes[i].plot(var_data.index[-20:], var_data[col].values[-20:], 
                'b-', linewidth=2, label='Historical', alpha=0.7)
    
    # Forecast
    axes[i].plot(forecast_df.index, forecast_df[col], 
                'r--', linewidth=2, label='VAR Forecast', marker='o')
    
    axes[i].axvline(var_data.index[-1], color='gray', linestyle=':', linewidth=1)
    axes[i].set_title(f'{col.capitalize()} Forecast', fontweight='bold')
    axes[i].set_ylabel(col.capitalize(), fontsize=11)
    axes[i].legend()
    axes[-1].set_xlabel('Year', fontsize=11)
plt.tight_layout()
plt.savefig('var_forecast.png', dpi=300, bbox_inches='tight')
plt.show()
