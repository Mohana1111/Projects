## TIME SERIES ANALYSIS ON AIR QUALITY DATA

This script/notebook demonstrates **time series forecasting** using **air quality data** from the `AirQualityUCI.xlsx` file.  
It performs **data cleaning**, **datetime handling**, **missing value treatment**, and fits an **ARIMA model** to forecast future pollutant levels.

---

### 🔹 Main Steps

**1️⃣ Load Data**

- Reads an Excel file (`AirQualityUCI.xlsx`) into a Pandas DataFrame.
- Ensures `Date` and `Time` columns are strings to prevent parsing errors.
- Merges `Date` and `Time` into a single `Datetime` column with proper format (`MM/DD/YY HH:MM:SS`).
- Drops rows where datetime parsing failed.
- Sets the new `Datetime` as the DataFrame index.
- Drops the original `Date` and `Time` columns to keep only clean timestamps.

---

**2️⃣ Clean & Interpolate**

- Replaces all `-200` values with `NaN` — the dataset uses `-200` as a missing value marker.
- Performs linear interpolation to fill missing measurements.
- Final dataset has continuous numeric time series for pollutant levels.

---

**3️⃣ ARIMA Forecasting**

- Fits an **ARIMA** model `(p=1, d=1, q=1)` on the `CO(GT)` pollutant time series.
  - `p`: autoregressive term.
  - `d`: differencing.
  - `q`: moving average term.
- Prints a summary of the ARIMA model fit.
- Plots model residuals to check assumptions.

---

**4️⃣ Forecast Plot**

- Forecasts the next 48 hours of `CO(GT)` values.
- Creates a future timestamp index to align the forecast.
- Plots historical `CO(GT)` vs. predicted future values.

---

### 🔹 Advanced Data Cleaning Function

The second block defines a **`load_and_preprocess`** function for robust preprocessing:
- Loads the Excel file and explicitly treats `Date` and `Time` as strings.
- Cleans any unwanted characters from `Date` and `Time` fields.
- Uses flexible `pd.to_datetime` parsing to handle mixed or non-standard formats.
- Validates and drops invalid datetime rows if the percentage is reasonable.
- Forward-fills and backward-fills any remaining missing values.
- Converts all columns to numeric, dropping any leftover non-numeric columns.
- Defines lists of **target** variables (pollutant columns) and **exogenous** variables (temperature, humidity, etc.) for future modeling.

---

### ⚙️ Purpose

This script shows how to:
- Prepare and validate real-world time series data.
- Handle messy timestamps and unusual missing value encodings.
- Train an ARIMA model for forecasting.
- Extend to more advanced models (SARIMAX, Prophet) with defined targets and exogenous features.

---

## Exploratory Data Analysis (EDA)

This script also includes a comprehensive **EDA pipeline** for the air quality dataset to understand its structure, missing data, trends, distributions, and correlations.

The `perform_eda(df)` function does the following:

---

### 1️⃣ Basic Statistics

- Prints descriptive statistics (`count`, `mean`, `std`, `min`, `max`, quartiles) for all numeric columns.
- Helps identify the scale, central tendency, and possible outliers.

---

### 2️⃣ Missing Values Heatmap

- Uses a heatmap (`sns.heatmap`) to visualize the location and extent of missing values across the entire DataFrame.
- Highlights if missingness is random or systematic (e.g., sensor dropouts).

---

### 3️⃣ Time Series Plots

- Selects target columns that contain `"(GT)"` (Gas Turbine measurements).
- Creates individual time series plots for each pollutant.
- Shows trends, seasonality, and sudden drops/spikes.

---

### 4️⃣ Distribution Analysis

- Plots histograms with KDE curves for each target pollutant.
- Reveals the skewness, normality, and possible multi-modality of the data.

---

### 5️⃣ Correlation Matrix

- Filters DataFrame to numeric columns only.
- Calculates pairwise correlation coefficients between features.
- Displays a heatmap (`sns.heatmap`) to identify strong positive/negative relationships.
- Helps spot multicollinearity or redundant features.

---

### 6️⃣ Boxplots for Outlier Detection

- Draws horizontal boxplots for all target pollutants.
- Visualizes the spread and highlights outliers.
- Useful for deciding whether to treat outliers before modeling.

---

### 7️⃣ Seasonal Decomposition

- Performs seasonal decomposition (`seasonal_decompose`) for `CO(GT)` as an example:
  - Splits the time series into **trend**, **seasonal**, and **residual** components.
  - Uses a daily cycle (`period=24`).
  - Shows whether there’s daily seasonality in carbon monoxide levels.

---

### Integration

At the end, an `if __name__ == "__main__":` block:
- Loads the data file safely.
- Validates that the file exists.
- Runs the `perform_eda` function only if the data loads successfully.

---

**Purpose:**  
This EDA block provides a solid **visual and statistical understanding** of the raw air quality data.  
It helps guide:
- Data cleaning (missing values, outliers)
- Feature selection (based on correlations)
- Choice of models (e.g., seasonal vs. non-seasonal)

Use it to validate data integrity **before fitting time series models** like ARIMA, SARIMAX, or Prophet.

---

## Forecasting with SARIMAX and Prophet

This final part of the script extends the project from EDA and ARIMA to **multi-model forecasting** for multiple air pollutant time series, with exogenous variables, and export of results to Excel.

---

### 🔹 `forecast_models` function

This function automates **forecast generation** for all defined pollutants (`targets`).

**How it works:**

- ✅ **Input:**  
  - `df` — the cleaned and preprocessed dataset.  
  - `targets` — list of pollutant columns to forecast.  
  - `exog_features` — exogenous features used to improve predictions (e.g., temperature, humidity).

- ✅ **Loop over each pollutant:**
  - Splits data into train/test: the last `steps` hours are reserved for testing (`steps=48` by default).
  - Fits a **SARIMAX** model (`auto_arima` from `pmdarima`) for each pollutant:
    - Uses exogenous variables for better prediction.
    - Seasonal settings are enabled (`seasonal=True, m=6`) to capture periodic patterns.
    - Keeps the search space constrained for speed (`p` and `q` max at 2).
  - Fits a **Prophet** model for each pollutant:
    - Adds exogenous regressors for temperature, humidity, and absolute humidity.
    - Generates hourly future DataFrames for the forecast horizon.
  - Handles errors gracefully — if a model fails, fills forecast with NaNs.
  - Stores both forecast results plus actual test values for comparison.

- ✅ **Returns:**  
  A nested dictionary with forecasts from both models for each target pollutant.

---

### 🔹 `make_future_dataframe`

Creates a **future time index** for the next `n` hours after the last timestamp in the dataset:
- Fills the `Date` and `Time` columns for output.
- Fills exogenous variables (`T`, `RH`, `AH`) with the last known value — a simple hold-last-value strategy.

---

### 🔹 `export_forecasts`

- Takes the `forecasts` output plus the generated future time DataFrame.
- Prepares two DataFrames: one for SARIMAX results, one for Prophet.
- Each includes:
  - Date and Time columns.
  - Forecasted pollutant values for all targets.
  - Exogenous features.
- Reorders columns to match the original dataset’s structure.
- Exports both forecasts into an **Excel file** with separate sheets (`SARIMAX_Forecast` and `Prophet_Forecast`).

---

### 🗂️ Purpose

This block demonstrates a **full multi-model time series forecasting pipeline**:
- Compare multiple models (SARIMAX, Prophet) side by side.
- Use exogenous weather features for better predictions.
- Automate generation and export of future pollutant levels.
- Supports real use cases like air quality reporting or environmental dashboards.

---

This completes the end-to-end flow:  
**✅ Clean raw sensor data → ✅ EDA → ✅ Time Series Modeling → ✅ Multi-step Forecast → ✅ Export for real-world use.**
