---
author: "Kyle Jones"
date_published: "April 1, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/dynamic-links-between-gold-newmont-and-gdx-using-var-irf-and-fevd-in-python-a08658d8a074"
---

# Dynamic Links Between Gold, Newmont, and GDX Using VAR, IRF, and FEVD in Python

This project uses vector autoregressive models and Granger causality to examine the relationship between the price of gold futures (GC=F)...

### **Dynamic Links Between Gold, Newmont, and GDX Using VAR, IRF, and** FEVD **in Python**
This project uses vector autoregressive models and Granger causality to examine the relationship between the price of gold futures (GC=F), Newmont Mining Corporation (NEM), and the VanEck Gold Miners ETF (GDX). My hypothesis is that the price of gold causes changes in the stock price for NEM and GDX.

I used daily data for the five years from Yahoo Finance.

The daily data was too volatile to show anything interesting. So I resampled the daily data into a monthly format by taking mean closing price. This resampling reduces noise and focuses the analysis on medium-term dynamics. After resampling, I compute the log returns of each time series because they stabilize the variance and allow for easier interpretation of percentage changes over time.

I tested each series for stationarity using the Augmented Dickey-Fuller (ADF) test. This test checks whether a time series has a stable mean and variance over time. Stationarity is a key assumption of vector autoregressive (VAR) models. All three series --- NEM, gold, and GDX --- pass the test, confirming they are stationary and suitable for modeling with VAR.

```python
# --- Monthly Log Returns ---
monthly_data = df.resample('ME').mean()
monthly_log_returns = np.log(monthly_data).diff().dropna()

# --- ADF Stationarity Tests ---
def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    print(f"{name} is {'stationary' if result[1] <= 0.05 else 'NOT stationary'}.")

print("\nMonthly Stationarity Tests:")
for col in monthly_log_returns.columns:
    check_stationarity(monthly_log_returns[col], col)
```

All three monthly log return series --- NEM, Gold, and GDX --- are stationary. The Augmented Dickey-Fuller test returns p-values well below 0.05, which means we can safely use them in a VAR model. Stationarity ensures the model is estimating stable relationships over time rather than chasing trends.

We can explore potential causal relationships using Granger causality tests. This test determines whether one time series can be used to forecast another. In our case, we test whether gold returns help forecast GDX returns.

``` 
# --- Granger Causality Test: Gold → GDX ---
print("\nGranger Causality Test (Gold → GDX):")
grangercausalitytests(monthly_log_returns[['GDX', 'Gold']], maxlag=3, verbose=True)
```

Granger causality tests whether past values of gold help forecast GDX. The p-values for all three lag lengths (1, 2, and 3) are greater than 0.05, which means we fail to reject the null hypothesis. In practical terms, this means gold returns do not significantly improve the forecast of GDX returns beyond what is already contained in GDX's own history. This does not mean gold has no influence --- it means there is no predictive lead-lag structure between them at the monthly level in this model.

#### VAR model
Next, we estimate a VAR model. A VAR model captures the interdependence of multiple time series by allowing each variable to depend on past values of itself and the other variables. We determine the appropriate number of lags using the Akaike Information Criterion (AIC), which balances model fit with complexity. Once the model is fit, we examine the parameter estimates and residual correlations.

``` 
# --- Fit VAR Model ---
model = VAR(monthly_log_returns)
lag_selection = model.select_order(12)
selected_lag = lag_selection.aic
fitted_model = model.fit(selected_lag)

print("\nVAR Model Summary:")
print(fitted_model.summary())
```

The model includes 3 equations --- one for each variable --- with 12 lags. The AIC score suggests the model fits the data well, but interpretation focuses on the coefficients and their significance.

The response of NEM to its own past values and those of Gold and GDX shows occasional significant coefficients. L3.NEM, L5.NEM, L7.NEM, and L9.NEM have statistically significant coefficients. NEM is highly autocorrelated, indicating its own history helps predict its future. L7.Gold and L3.Gold have p-values below 0.05. This suggests some response of NEM to gold shocks with a lag, especially around the third and seventh months.

L7.GDX is also significant. This suggests interactions between NEM and GDX beyond just shared exposure to gold.

None of the lagged predictors are statistically significant at the 5% level. Even though the coefficients vary in sign and magnitude, the standard errors are large and p-values are all above 0.1. This suggests gold behaves more independently and is less responsive to shocks in NEM or GDX. This fits with gold's role as a macroeconomic asset, driven by broader factors like interest rates and global risk sentiment.

Like NEM, the GDX equation contains few significant coefficients. Most lags have p-values above 0.1. L9. Gold has a borderline p-value (0.142), suggesting a weak delayed response of GDX to gold. L3.Gold also approaches significance but still exceeds common thresholds. The high correlation between GDX and both NEM and gold suggests shared influences but not strong predictive lag structures.

The residuals from all three equations are highly correlated:

- NEM and GDX: 0.91
- Gold and GDX: 0.95
- Gold and NEM: 0.82

This shows that these assets move together due to common shocks, even if their past values do not predict each other strongly in the Granger sense. High residual correlation also suggests that exogenous macroeconomic factors are affecting all three simultaneously.

#### Impulse Response Functions
I looked at the system's dynamic response using impulse response functions (IRFs). These functions trace the effect of a one-time shock to one variable on the current and future values of all variables in the system. Specifically, we analyze how a shock to gold affects GDX over a twelve-month horizon. The IRF shows the direction, magnitude, and persistence of this influence over time.

Gold → GDX

The cumulative IRF shows a moderate and sustained increase in GDX following a shock to Gold. The response grows over time, peaking around month 12. This suggests that while the Granger causality test found no significant short-term predictive power, the VAR model still captures a longer-term, accumulated response of the GDX index to changes in Gold prices. The influence is positive and gradually builds, indicating a delayed but persistent effect.


Gold → Gold

Gold's own cumulative response rises steadily. This confirms that shocks to Gold exhibit momentum or persistence. In practical terms, when Gold experiences a shock, it tends to continue moving in the same direction over the next year, rather than reverting quickly.

Gold → NEM

The response of Newmont (NEM) to Gold shocks is the strongest among the three. It rises sharply from month 5 onward, peaking around month 12. This makes sense: NEM is a large gold mining company, so its performance closely tracks the commodity price. The sharp cumulative effect confirms that Gold price shocks significantly influence NEM returns.

#### Forecast Error Variance Decomposition
To summarize the long-run influence of each variable, we turn to forecast error variance decomposition (FEVD). FEVD quantifies how much of the forecast error variance of a variable, at each horizon, is attributable to shocks from each of the variables in the model. This gives us a breakdown of the drivers behind GDX, gold, and NEM over time. We visualize the results using stacked area charts, which provide a clear view of the cumulative contribution from each source of variation.

In the short term, NEM explains the majority of the variation in GDX. This highlights the strong influence of company-specific returns (NEM being a major component of GDX). Gold contributes some share, but much less than NEM. GDX's own history explains very little of its forward variance.


Gold is mostly driven by itself. Over the 3-month horizon, most of the variance in Gold returns remains self-driven, with NEM contributing a minor share. This matches expectations, as commodity prices are largely governed by macroeconomic and geopolitical factors, not equity prices.


In the first month, NEM is nearly entirely self-driven. By month 2 and 3, the share of variance explained by Gold increases significantly. This supports what the IRFs showed: NEM gradually absorbs the impact of Gold, even if the initial response is muted. GDX has little direct influence on NEM.


#### So What?
GDX reacts to Gold shocks gradually, but the effect is small compared to its exposure to NEM.


NEM is highly responsive to Gold, with a strong cumulative effect.


Gold drives itself, and while it influences NEM and GDX, the effect is stronger and more persistent on NEM.


FEVD supports the idea that NEM drives GDX more than Gold does, at least over short horizons.

Shocks to Gold ripple through the system with lag and diminishing strength. NEM absorbs these shocks more directly. GDX, as an ETF of miners, reflects both --- but is dominated by the performance of its top holdings like NEM.

In the end, there is evidence that the price of gold and NEM/GDX are correlated. There is not evidence of a Granger causality. But there is evidence that shocks to gold affect NEM about 4 months later.
