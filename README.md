# VAR, IRF, and FEVD Analysis: Gold, Newmont, and GDX

This project demonstrates Vector Autoregression (VAR) modeling with Impulse Response Functions (IRF) and Forecast Error Variance Decomposition (FEVD) for analyzing dynamic relationships between financial time series.

## Business context

This project uses vector autoregressive models and Granger causality to examine the relationship between the price of gold futures (GC=F)...

This project uses vector autoregressive models and Granger causality to examine the relationship between the price of gold futures (GC=F), Newmont Mining Corporation (NEM), and the VanEck Gold Miners ETF (GDX). My hypothesis is that the price of gold causes changes in the stock price for NEM and GDX.

The daily data was too volatile to show anything interesting. So I resampled the daily data into a monthly format by taking mean closing price. This resampling reduces noise and focuses the analysis on medium-term dynamics. After resampling, I compute the log returns of each time series because they stabilize the variance and allow for easier interpretation of percentage changes over time.

## Article

Medium article: [Dynamic Links Between Gold, Newmont, and GDX Using VAR, IRF, and FEVD](https://medium.com/@kylejones_47003/dynamic-links-between-gold-newmont-and-gdx-using-var-irf-and-fevd-in-python-a08658d8a074)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # VAR, IRF, FEVD functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (tickers, date ranges)
- VAR model parameters (max lag, lag selection criterion)
- IRF and FEVD parameters (periods, orthogonalization)
- Granger causality tests
- Output settings

## Methods

### Vector Autoregression (VAR)
- Models multiple time series simultaneously
- Captures dynamic interdependencies
- Automatic lag selection via information criteria

### Impulse Response Functions (IRF)
- Shows response of each variable to shocks in other variables
- Cumulative IRFs show long-term effects
- Useful for understanding transmission mechanisms

### Forecast Error Variance Decomposition (FEVD)
- Decomposes forecast error variance by source
- Identifies relative importance of shocks
- Measures contribution of each variable to forecast uncertainty

## Caveats

- Data is fetched from Yahoo Finance. Ensure internet connection.
- All series must be stationary (log returns are used by default).
- VAR models require sufficient data for reliable estimation.
- Results are sensitive to lag selection and model specification.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).