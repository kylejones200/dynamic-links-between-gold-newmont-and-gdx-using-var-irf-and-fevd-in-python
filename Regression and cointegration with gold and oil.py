"""Generated from Jupyter notebook: Regression and cointegration with gold and oil

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller, coint


def adf_test(series, name):
    result = adfuller(series)
    print(f"ADF Test for {name}:")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-Value: {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} has a unit root (non-stationary).\n")
    else:
        print(f"{name} is stationary.\n")


def simulate_time_series_data() -> None:
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    y1 = np.cumsum(np.random.normal(size=n))
    y2 = 0.5 * np.cumsum(np.random.normal(size=n)) + 10
    y3 = np.sin(t / 10) + np.random.normal(scale=0.5, size=n)
    data = pd.DataFrame({"y1": y1, "y2": y2, "y3": y3})
    data.plot(subplots=True, figsize=(10, 8), title="Simulated Time Series")
    plt.show()
    adf_test(data["y1"], "y1")
    adf_test(data["y2"], "y2")
    adf_test(data["y3"], "y3")
    X = add_constant(data["y2"])
    model = OLS(data["y1"], X).fit()
    residuals = model.resid
    adf_test(residuals, "Residuals of y1 ~ y2")
    coint_stat, p_value, critical_values = coint(data["y1"], data["y2"])
    print("Engle-Granger Cointegration Test:")
    print(f"Test Statistic: {coint_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Critical Values: {critical_values}")
    if p_value < 0.05:
        print("y1 and y2 are cointegrated.\n")
    else:
        print("y1 and y2 are not cointegrated.\n")

    adf_test(residuals, "Residuals of y1 ~ y2")
    coint_stat, p_value, critical_values = coint(data["y1"], data["y2"])
    print("Engle-Granger Cointegration Test:")
    print(f"Test Statistic: {coint_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Critical Values: {critical_values}")
    if p_value < 0.05:
        print("y1 and y2 are cointegrated.\n")
    else:
        print("y1 and y2 are not cointegrated.\n")

    plt.figure(figsize=(10, 6))
    plt.plot(residuals, label="Residuals", color="blue")
    plt.axhline(0, linestyle="--", color="red", label="Zero Line")
    plt.title("Residuals of Linear Regression (y1 ~ y2)")
    plt.xlabel("Time")
    plt.ylabel("Residual Value")
    plt.legend()
    plt.grid()
    plt.show()


def use_the_correct_ticker_symbol_for_gold_for_examp() -> None:
    import matplotlib.pyplot as plt
    import yfinance as yf

    # Use the correct ticker symbol for gold, for example GC=F (Gold Futures)
    gold_ticker = "GC=F"
    # Fetch historical data
    gold_data = yf.download(gold_ticker, start="2022-01-01", end="2023-01-01")
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(gold_data["Close"], label="Gold Price", color="gold")
    plt.title("Gold Price Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


def use_the_correct_ticker_symbol_for_gold_for_examp_2() -> None:
    import matplotlib.pyplot as plt
    import yfinance as yf

    # Use the correct ticker symbol for gold, for example GC=F (Gold Futures)
    gold_ticker = "CL=F"
    # Fetch historical data
    gold_data = yf.download(gold_ticker, start="2022-01-01", end="2023-01-01")
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(gold_data["Close"], label="Gold Price", color="gold")
    plt.title("Gold Price Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


def download_data() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import yfinance as yf
    from statsmodels.api import OLS, add_constant
    from statsmodels.tsa.stattools import adfuller, coint

    # Download data
    def get_data(start_date="2020-01-01", end_date="2024-02-04"):
        # Download data
        print("Downloading Gold data...")
        gold = yf.download("GC=F", start=start_date, end=end_date)
        print("Downloading Oil data...")
        oil = yf.download("CL=F", start=start_date, end=end_date)
        print("\nGold data shape:", gold.shape)
        print("Oil data shape:", oil.shape)
        # Check if data is empty
        if gold.empty or oil.empty:
            raise ValueError("No data retrieved for one or both symbols")

        # Create DataFrame with close prices
        merged_data = pd.merge(
            gold["Close"],
            oil["Close"],
            left_index=True,
            right_index=True,
            how="inner",
            suffixes=("_Gold", "_Oil"),
        )
        # Rename columns
        merged_data.columns = ["Gold", "Oil"]
        print("\nMerged data shape:", merged_data.shape)
        print("\nFirst few rows of merged data:")
        print(merged_data.head())
        return merged_data

    # Function to perform ADF test
    def adf_test(series, name):
        result = adfuller(series)
        print(f"ADF Test for {name}:")
        print(f"Test Statistic: {result[0]:.4f}")
        print(f"P-Value: {result[1]:.4f}")
        if result[1] > 0.05:
            print(f"{name} has a unit root (non-stationary).\n")
        else:
            print(f"{name} is stationary.\n")

    try:
        # Get the data
        data = get_data()
        # Plot the series
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Gold"], label="Gold", color="gold")
        plt.plot(data.index, data["Oil"], label="Oil", color="black")
        plt.title("Gold and Oil Futures Prices")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()
        # Test for stationarity
        adf_test(data["Gold"], "Gold")
        adf_test(data["Oil"], "Oil")
        # Perform linear regression with constant
        X = add_constant(data["Oil"])
        model = OLS(data["Gold"], X).fit()
        residuals = model.resid
        # Print regression results
        print("\nRegression Results:")
        print(model.summary().tables[1])
        # Test residuals for stationarity (cointegration)
        adf_test(residuals, "Residuals of Gold ~ Oil")
        # Use coint function for direct testing
        coint_stat, p_value, critical_values = coint(data["Gold"], data["Oil"])
        print("\nEngle-Granger Cointegration Test:")
        print(f"Test Statistic: {coint_stat:.4f}")
        print(f"P-Value: {p_value:.4f}")
        print("Critical Values:")
        print(f"1%: {critical_values[0]:.4f}")
        print(f"5%: {critical_values[1]:.4f}")
        print(f"10%: {critical_values[2]:.4f}")
        if p_value < 0.05:
            print("Gold and Oil are cointegrated.\n")
        else:
            print("Gold and Oil are not cointegrated.\n")

        # Plot the residuals
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, residuals, label="Residuals", color="blue")
        plt.axhline(0, linestyle="--", color="red", label="Zero Line")
        plt.title("Residuals of Linear Regression (Gold ~ Oil)")
        plt.xlabel("Date")
        plt.ylabel("Residual Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Print more detailed error information
        # import traceback

        traceback.print_exc()


def download_data_2() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from statsmodels.api import OLS, add_constant
    from statsmodels.tsa.stattools import adfuller, coint

    def get_data(
        start_date="2020-01-01", end_date="2025-02-04", scaling_method="standard"
    ):
        # Download data
        print("Downloading Gold data...")
        gold = yf.download("GC=F", start=start_date, end=end_date)
        print("Downloading Oil data...")
        oil = yf.download("CL=F", start=start_date, end=end_date)
        # Create DataFrame with close prices
        merged_data = pd.merge(
            gold["Close"],
            oil["Close"],
            left_index=True,
            right_index=True,
            how="inner",
            suffixes=("_Gold", "_Oil"),
        )
        # Rename columns
        merged_data.columns = ["Gold", "Oil"]
        # Apply scaling
        if scaling_method == "standard":
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(merged_data),
                columns=merged_data.columns,
                index=merged_data.index,
            )
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(merged_data),
                columns=merged_data.columns,
                index=merged_data.index,
            )
        else:
            raise ValueError("Invalid scaling method. Use 'standard' or 'minmax'")

        print("\nOriginal data first few rows:")
        print(merged_data.head())
        print("\nScaled data first few rows:")
        print(scaled_data.head())
        return scaled_data, merged_data

    def adf_test(series, name):
        result = adfuller(series)
        print(f"ADF Test for {name}:")
        print(f"Test Statistic: {result[0]:.4f}")
        print(f"P-Value: {result[1]:.4f}")
        if result[1] > 0.05:
            print(f"{name} has a unit root (non-stationary).\n")
        else:
            print(f"{name} is stationary.\n")

    try:
        # Get both original and scaled data
        scaled_data, original_data = get_data(scaling_method="standard")
        # Plot both original and scaled series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # Original data
        ax1.plot(original_data.index, original_data["Gold"], label="Gold", color="gold")
        ax1.plot(original_data.index, original_data["Oil"], label="Oil", color="black")
        ax1.set_title("Original Prices")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)")
        ax1.legend()
        ax1.grid(True)
        # Scaled data
        ax2.plot(
            scaled_data.index, scaled_data["Gold"], label="Gold (scaled)", color="gold"
        )
        ax2.plot(
            scaled_data.index, scaled_data["Oil"], label="Oil (scaled)", color="black"
        )
        ax2.set_title("Scaled Prices")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Standardized Price")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
        # Test for stationarity
        adf_test(scaled_data["Gold"], "Scaled Gold")
        adf_test(scaled_data["Oil"], "Scaled Oil")
        # Perform linear regression with constant
        X = add_constant(scaled_data["Oil"])
        model = OLS(scaled_data["Gold"], X).fit()
        residuals = model.resid
        # Print regression results
        print("\nRegression Results:")
        print(model.summary().tables[1])
        # Test residuals for stationarity (cointegration)
        adf_test(residuals, "Residuals of Gold ~ Oil")
        # Use coint function for direct testing
        coint_stat, p_value, critical_values = coint(
            scaled_data["Gold"], scaled_data["Oil"]
        )
        print("\nEngle-Granger Cointegration Test:")
        print(f"Test Statistic: {coint_stat:.4f}")
        print(f"P-Value: {p_value:.4f}")
        print("Critical Values:")
        print(f"1%: {critical_values[0]:.4f}")
        print(f"5%: {critical_values[1]:.4f}")
        print(f"10%: {critical_values[2]:.4f}")
        if p_value < 0.05:
            print("Gold and Oil are cointegrated.\n")
        else:
            print("Gold and Oil are not cointegrated.\n")

        # Plot the residuals
        plt.figure(figsize=(12, 6))
        plt.plot(scaled_data.index, residuals, label="Residuals", color="blue")
        plt.axhline(0, linestyle="--", color="red", label="Zero Line")
        plt.title("Residuals of Linear Regression (Scaled Gold ~ Scaled Oil)")
        plt.xlabel("Date")
        plt.ylabel("Residual Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()


def main() -> None:
    simulate_time_series_data()
    use_the_correct_ticker_symbol_for_gold_for_examp()
    use_the_correct_ticker_symbol_for_gold_for_examp_2()
    download_data()
    download_data_2()


if __name__ == "__main__":
    main()
