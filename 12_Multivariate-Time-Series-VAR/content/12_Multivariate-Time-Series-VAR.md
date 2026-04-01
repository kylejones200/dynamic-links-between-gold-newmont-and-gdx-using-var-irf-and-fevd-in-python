# Multivariate Time Series Forecasting with Vector Autoregression (VAR) in Python

Multivariate time series capture relationships between multiple variables. We use Vector Autoregression (VAR) to model interactions between energy consumption, production, and CO2 emissions, including cointegration testing and impulse response analysis.
### Multivariate Time Series Forecasting with Vector Autoregression (VAR) in Python
Most time series don't exist in isolation. Energy consumption affects production, which affects emissions. Prices influence demand, which influences supply. Multivariate models capture these relationships, providing richer forecasts than univariate methods.

Vector Autoregression (VAR) models multiple time series simultaneously, capturing cross-variable dependencies. We use VAR to model Oklahoma's energy system: consumption, production, and CO2 emissions interact in complex ways that univariate models miss.

### Dataset: Multiple Energy Metrics
We combine consumption, production, and emissions data into a multivariate series.


The multivariate dataset captures relationships between consumption, production, and emissions. After aligning the three series, we obtain **54 annual observations from 1970–2023**, with summary statistics:

- **Consumption**: mean ≈ 20.4M, ranging from 14.2M to 24.8M units  
- **Production**: mean ≈ 7.1M, ranging from 4.51M to 8.65M units  
- **Emissions**: mean ≈ 946, ranging from 410.7 to 1,569.7 million metric tons  

The three series are also strongly correlated at the residual level: consumption and production residuals have a correlation of about **0.95**, while emissions are more weakly correlated with the other two.

### VAR Model
VAR models each variable as a function of its own lags and lags of other variables.


VAR captures cross-variable dependencies through lagged relationships. Before fitting, we difference the series to address non-stationarity; Augmented Dickey–Fuller tests on the original levels yield **p-values above 0.24** for all three variables, indicating that a simple VAR in levels would be inappropriate.

On the differenced data, the information criteria select a **lag order of 0** for this particular specification. That means most of the structure is captured by level shifts rather than additional lags, and the model effectively behaves like a multivariate mean model with a rich residual covariance structure.

### Cointegration Testing
Cointegration tests whether variables have long-run relationships.


If cointegrated, use Vector Error Correction Model (VECM) instead of VAR.

### Impulse Response Analysis
Impulse response shows how shocks in one variable affect others.


Impulse response reveals dynamic relationships between variables. Even with a low selected lag order, we can still examine how a one-time shock to consumption or production would propagate to the other variables over a short horizon. The figure `var_impulse_response.png` visualizes these responses for a 10-period window, highlighting, for example, how a consumption shock primarily affects production with only modest spillover to emissions.

### Granger Causality
Granger causality tests whether one variable helps predict another.


Granger causality identifies predictive relationships.

### Forecast Visualization
We visualize VAR forecasts for all variables.


VAR provides forecasts for all variables simultaneously, capturing their interactions. The chart `var_forecast.png` overlays the last decades of historical consumption, production, and emissions with their corresponding VAR-based forecasts, illustrating how the model extrapolates recent trends while respecting the joint dynamics estimated from the multivariate system.

### Key Advantages of VAR
- Captures cross-variable dependencies Models interactions between multiple series
- Interpretable Coefficients show how variables affect each other
- Impulse response analysis Understands dynamic effects of shocks
- Granger causality Identifies predictive relationships

### When to Use VAR
Use VAR when:
- You have multiple related time series
- Variables influence each other
- You need to understand cross-variable relationships
- Interpretability is important

Consider alternatives when:
- Series are not cointegrated (use VECM)
- Non-linear relationships exist (use neural VAR)
- High-dimensional systems (use factor models)

### Conclusion
VAR provides a powerful framework for multivariate time series forecasting. It captures cross-variable dependencies, provides interpretable results, and enables rich analysis through impulse response and Granger causality tests.


