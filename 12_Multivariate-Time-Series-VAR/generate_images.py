#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""
import logging

logger = logging.getLogger(__name__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    tf = None
except:
    pass

# Tufte-style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
import matplotlib.pyplot as plt
original_savefig = plt.savefig

def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith('/') and not str(filename).startswith('images/'):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")

plt.savefig = savefig_tufte

# Code blocks from article

# Code block 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

use_df = pd.read_csv("../../geospatial/datasets/use_OK.csv")
pr_df = pd.read_csv("../../geospatial/datasets/pr_OK.csv")
co2_df = pd.read_csv("../../geospatial/datasets/co2_OK.csv")

# Extract and prepare each series
def prepare_series(df, col_name):
    year_cols = [col for col in df.columns if col.isdigit()]
    df_long = df.melt(
        id_vars=['State', 'MSN'],
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    df_long['Year'] = pd.to_datetime(df_long['Year'], format='%Y')
    df_long = df_long.sort_values('Year')
    
    total = df_long[df_long['MSN'].str.contains('TOT|TCR', na=False)].copy()
    total = total.groupby('Year')['Value'].sum().reset_index()
    total = total[total['Value'].notna() & (total['Value'] > 0)]
    
    return total.set_index('Year')['Value'].interpolate(method='linear').sort_index()

use_series = prepare_series(use_df, 'consumption')
pr_series = prepare_series(pr_df, 'production')
co2_series = prepare_series(co2_df, 'emissions')

# Align indices
common_years = use_series.index.intersection(pr_series.index).intersection(co2_series.index)
use_series = use_series.loc[common_years]
pr_series = pr_series.loc[common_years]
co2_series = co2_series.loc[common_years]

var_data = pd.DataFrame({
    'consumption': use_series,
    'production': pr_series,
    'emissions': co2_series
})

logger.info(f"Multivariate series length: {len(var_data)}")
logger.info(f"Date range: {var_data.index.min()} to {var_data.index.max()}")
logger.info(f"\nSeries statistics:")
logger.info(var_data.describe())



# Code block 2
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



# Code block 3
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



# Code block 4
# Impulse response function
irf = var_fitted.irf(10)  # 10 periods ahead

irf.plot(figsize=(12, 10))
plt.tight_layout()
plt.savefig('var_impulse_response.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze specific responses
logger.info("\nImpulse Response Analysis:")
logger.info("Shock in consumption affects:")
logger.info(f"  Production after 1 period: {irf.irfs[1, 1, 0]:.4f}")
logger.info(f"  Emissions after 1 period: {irf.irfs[1, 2, 0]:.4f}")



# Code block 5
# Granger causality tests
gc = var_fitted.test_causality('consumption', 'production', kind='f')
logger.info(f"\nGranger Causality: consumption -> production")
logger.info(f"F-statistic: {gc.test_statistic:.4f}, p-value: {gc.pvalue:.4f}")
logger.info(f"Causal: {'Yes' if gc.pvalue < 0.05 else 'No'}")

gc2 = var_fitted.test_causality('production', 'emissions', kind='f')
logger.info(f"\nGranger Causality: production -> emissions")
logger.info(f"F-statistic: {gc2.test_statistic:.4f}, p-value: {gc2.pvalue:.4f}")
logger.info(f"Causal: {'Yes' if gc2.pvalue < 0.05 else 'No'}")



# Code block 6
forecast_dates = pd.date_range(start=var_data.index[-1] + pd.DateOffset(years=1), 
                               periods=forecast_steps, freq='YS')
forecast_df = pd.DataFrame(forecast_levels, 
                          index=forecast_dates,
                          columns=var_data.columns)

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
plt.tight_layout()
plt.savefig('var_forecast.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 7
# Complete code for reproducibility
# See individual code blocks above for full implementation



logger.info("All images generated successfully!")
