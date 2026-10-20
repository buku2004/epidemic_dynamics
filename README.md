
# Epidemic Dynamics & Forecasting

### A Comparative Study of Mechanistic, Statistical, and Machine Learning Models for COVID-19 Forecasting in India

## 1. Project Overview

Forecasting epidemic trends is challenging because disease transmission changes over time due to population immunity, public health interventions, emerging variants, testing patterns, and human behaviour.

**Epidemic Dynamics & Forecasting** compares four modelling approaches for forecasting COVID-19 cases in India:

- **SIR:** Mechanistic epidemiological modelling.
- **ARIMA:** Statistical time-series forecasting.
- **XGBoost:** Gradient-boosted machine learning.
- **LSTM:** Deep learning for sequential time-series forecasting.

The project investigates predictive accuracy, generalization to a later epidemic wave, interpretability, and the limitations of different modelling assumptions.

## 2. Project Objectives

The primary objective is to evaluate how mechanistic, statistical, and machine-learning approaches forecast COVID-19 cases under changing epidemic conditions.

Specific objectives:

1. Model transmission and recovery dynamics using epidemiological differential equations.
2. Capture temporal dependencies using statistical forecasting.
3. Learn nonlinear relationships from historical case counts using machine learning.
4. Learn sequential temporal patterns in historical case counts using an LSTM neural network.
5. Compare models using MAE, RMSE, and R² score on a common test period.
6. Analyze forecasting errors and the ability of models to capture epidemic surges.

### Research Question

**How do different modelling approaches compare when forecasting COVID-19 cases during a later epidemic wave that differs from the training period?**

## 3. Dataset and Data Source

**Source:** Our World in Data (OWID) COVID-19 dataset

- Dataset: https://github.com/owid/covid-19-data
- Documentation: https://docs.owid.io/projects/covid-19-data/en/latest/

The project uses India's COVID-19 time-series data.

### Key Variables

| Variable | Description |
|---|---|
| `date` | Date of observation |
| `new_cases` | Newly reported COVID-19 cases |
| `new_cases_smoothed` | Smoothed daily reported cases |
| `total_cases` | Cumulative reported cases |
| `new_deaths` | Newly reported deaths |
| `reproduction_rate` | Estimated effective reproduction rate |
| `positive_rate` | Test positivity rate |
| `stringency_index` | Government response stringency indicator |

The primary forecasting target is `new_cases_smoothed`, which reduces short-term fluctuations associated with reporting irregularities.

### Data Preprocessing

- Parse dates and sort observations chronologically.
- Filter the dataset to the study period.
- Inspect and handle missing values.
- Create lagged and rolling features for XGBoost.
- Prepare chronological sequences for the LSTM model.
- Avoid data leakage by fitting models and preprocessing steps using training data only.

## 4. Experimental Design

### Train-Test Split

| Period | Dates | Purpose |
|---|---|---|
| Training | January 2020 – September 30, 2021 | Model fitting |
| Testing | October 1, 2021 – March 31, 2022 | Out-of-sample evaluation |

The test period includes the Omicron-associated surge in India, providing a challenging evaluation setting.

Hyperparameter selection should use a chronological validation set within the training period. The final test set must remain untouched during model selection.

All models should be evaluated on the same test dates and observed target values.

### Evaluation Metrics

#### Mean Absolute Error (MAE)

Measures the average absolute difference between observed and predicted case counts. Lower values indicate smaller average errors.

#### Root Mean Squared Error (RMSE)

Penalizes large errors more heavily, making it useful for evaluating missed epidemic peaks. Lower values indicate better performance.

#### R² Score (Coefficient of Determination)

Measures predictive performance relative to a baseline that predicts the mean of the observed test values.

- **R² = 1:** Perfect predictions.
- **R² = 0:** Equivalent to predicting the test-set mean.
- **R² < 0:** Worse than predicting the test-set mean.

Higher R² generally indicates a better fit. It should be interpreted alongside MAE and RMSE because a model can explain overall variation while still missing important epidemic peaks.

## 5. Models and Methodology

### 5.1 SIR — Susceptible-Infected-Recovered Model

**Category:** Mechanistic epidemiological model

The SIR model divides the population into susceptible, infectious, and recovered or removed compartments.

The governing equations are:

dS/dt = -βSI/N

dI/dt = βSI/N - γI

dR/dt = γI

Where:

- S: Susceptible population.
- I: Infectious population.
- R: Recovered or removed population.
- β: Transmission rate.
- γ: Recovery rate.
- N: Total population.

The basic reproduction number under standard model assumptions is:

R₀ = β/γ

#### Methodology

1. Define compartments and initial conditions.
2. Estimate transmission and recovery parameters using training observations.
3. Solve the differential equations numerically.
4. Derive predicted incidence from transmission dynamics.
5. Compare predictions with observed case counts.
6. Calculate MAE, RMSE, and R².

#### Why SIR?

SIR provides an interpretable epidemiological baseline and describes transmission through explicit mechanisms.

#### Limitations

The standard model does not explicitly represent an exposed stage, waning immunity, or time-varying transmission. These simplifying assumptions can limit its ability to reproduce real-world epidemic waves.

### 5.2 ARIMA — Autoregressive Integrated Moving Average

**Category:** Statistical time-series model

ARIMA models temporal dependencies using autoregressive terms, differencing, and moving-average terms.

ARIMA(p, d, q)

- p: Number of autoregressive terms.
- d: Degree of differencing.
- q: Number of moving-average terms.

#### Methodology

1. Prepare and inspect the daily case-count series.
2. Assess stationarity using statistical tests and time-series diagnostics.
3. Select appropriate differencing and model orders.
4. Fit ARIMA using the training period.
5. Generate forecasts for the test period.
6. Calculate MAE, RMSE, and R².

#### Why ARIMA?

ARIMA establishes a statistical baseline that captures temporal patterns without requiring explicit epidemiological equations or extensive feature engineering.

#### Limitations

It relies heavily on historical temporal relationships. Sudden changes in transmission or reporting can make those relationships unreliable.

### 5.3 XGBoost — Extreme Gradient Boosting

**Category:** Supervised machine learning

XGBoost learns nonlinear relationships between historical features and the target variable, `new_cases_smoothed`.

#### Feature Engineering

| Feature | Purpose |
|---|---|
| `lag_1` | Cases one day earlier |
| `lag_2` | Cases two days earlier |
| `lag_3` | Cases three days earlier |
| `lag_7` | Cases seven days earlier |
| `lag_14` | Cases fourteen days earlier |
| `rolling_7` | Mean of the preceding seven days |
| `rolling_14` | Mean of the preceding fourteen days |
| `sin_year`, `cos_year` | Cyclical day-of-year features |

Rolling features must use only observations available before the prediction date.

#### Methodology

1. Sort observations chronologically.
2. Construct lagged, rolling, and calendar features.
3. Remove rows with unavailable historical features.
4. Fit the XGBoost regressor on training data.
5. Select hyperparameters using chronological validation.
6. Generate predictions using a defined forecasting strategy.
7. Calculate MAE, RMSE, and R².

Two forecasting strategies may be examined:

- **Rolling-origin forecasting:** Forecast a short horizon, such as fourteen days, then update with newly observed data.
- **Recursive forecasting:** Feed previous predictions into subsequent forecasts, allowing prediction errors to accumulate.

#### Why XGBoost?

It can learn nonlinear relationships between recent case counts and temporal features. Feature importance can also help explain which inputs contribute to its predictions.

#### Limitations

It does not inherently enforce epidemiological laws. Its performance depends on feature quality and how closely future conditions resemble the training data.

### 5.4 LSTM — Long Short-Term Memory Network

**Category:** Deep learning for time-series forecasting

LSTM is a recurrent neural-network architecture designed to learn temporal dependencies in sequential data. Its gated memory mechanism helps it retain useful information across multiple time steps.

For this project, the model uses the previous 28 days of `new_cases_smoothed` to predict the next day's case count. The target is scaled using a scaler fitted only on the training period.

#### Methodology

1. Sort India's OWID observations chronologically and select data through March 31, 2022.
2. Use `new_cases_smoothed` as the primary target, handling missing observations carefully.
3. Split the data at September 30, 2021: observations up to this date are used for training, and October 1, 2021–March 31, 2022 is reserved for testing.
4. Fit the feature scaler using training observations only.
5. Create supervised sequences using the previous 28 days to predict the next day.
6. Train a stacked LSTM network with dropout and a dense output layer; use a chronological validation subset and early stopping.
7. Generate predictions for the test period and inverse-transform them to the original case-count scale.
8. Evaluate predictions using MAE, RMSE, and R², and inspect forecast plots and residuals.

#### Forecasting strategy

The current implementation uses **rolling one-step-ahead evaluation**: each test prediction uses the preceding 28 observed values, which may include observations from earlier dates in the test period. This evaluates daily predictions as new observations become available; it is not a single recursive six-month forecast made on September 30, 2021.

#### Why LSTM?

LSTM can learn nonlinear temporal patterns from sequential case-count data without requiring explicit epidemiological differential equations. It provides a deep-learning comparison with the mechanistic SIR model, statistical ARIMA, and feature-based XGBoost.

#### Limitations

LSTM requires careful scaling, sequence construction, and validation to avoid data leakage. It can overfit, is sensitive to architecture and hyperparameters, and may struggle when epidemic dynamics change sharply from the training period. A univariate LSTM also does not explicitly model transmission, immunity, vaccination, or other epidemiological mechanisms.

## 6. Results

### Preliminary Experiments

Earlier SIR and ARIMA experiments used a different train-test split. Their results are not included as final scores because they are not directly comparable with the planned October 2021–March 2022 evaluation.

All four models must be evaluated on the common test period before drawing final conclusions.

### Final Evaluation Results

| Model   | MAE         | RMSE        | R² Score |
|---------|------------:|------------:|---------:|
| SIR     | 107,841.86  | 140,854.31  | -1.7       |
| ARIMA   | 50,012.00   | 98,731.12  | -0.364        |
| XGBoost | 18,474.40   | 41,436.40   | 0.76     |
| LSTM    | 18,890.62   | 36,858.31   | 0.81     |

*TBD indicates that the final metric has not yet been confirmed.*

### Results Interpretation

The final analysis will investigate:

- Which model achieves the lowest MAE and RMSE?
- Which model achieves the highest R²?
- Which models capture the timing and magnitude of major surges?
- How does performance change with forecast horizon?
- How well does LSTM capture the timing and magnitude of major surges?
- How much error accumulates during recursive forecasting?

Rankings must be based on measured results, not assumptions about a model's capabilities.

## 7. Final Model Comparison

| Criterion | SIR | ARIMA | XGBoost | LSTM |
|---|---|---|---|---|
| Approach | Mechanistic | Statistical | Machine learning | Deep learning (sequence model) |
| Learns from historical data | Parameter fitting | Yes | Yes | Yes |
| Encodes epidemic dynamics | Explicitly | No | No, unless added externally | No, unless epidemiological features or constraints are added |
| Captures nonlinear relationships | Through equations | Limited by specification | Yes | Yes |
| Feature engineering | Low | Low to moderate | Moderate to high | Sequence-window design and scaling |
| Interpretability | Epidemiological parameters | Statistical structure | Feature importance and tree behaviour | Depends on states, parameters, and loss |
| Main strength | Mechanistic interpretation | Time-series baseline | Flexible nonlinear prediction | Learns temporal dependencies from sequential observations |
| Main limitation | Simplified assumptions | Vulnerable to structural changes | Feature and distribution dependence | Data-hungry, tuning-sensitive, and vulnerable to distribution shifts |
| Predictive performance | To be measured | To be measured | To be measured | To be measured |

This table compares methodological characteristics. Empirical rankings will be added after the common evaluation is complete.

## 8. Discussion

The four models represent different perspectives on epidemic forecasting.

SIR provides an interpretable explanation of transmission and recovery but relies on simplified assumptions. ARIMA captures historical temporal patterns but can struggle when the underlying process changes. XGBoost learns nonlinear relationships from engineered features but remains dependent on the information available in its training data.

LSTM investigates whether a recurrent neural network can learn useful temporal dependencies from historical case counts and improve predictions over the test period.

The Omicron-associated period provides a challenging test because the relationship between past cases and future outcomes may differ substantially from earlier periods. Changes in variants, immunity, testing, interventions, and reporting can all affect model performance.

**Forecast accuracy and epidemiological plausibility are not the same thing.** A model may accurately predict reported cases without recovering the true infectious population, while a mechanistic model may produce plausible compartment dynamics but poor reported-case forecasts.

Results should therefore be interpreted using MAE, RMSE, R², forecast plots, residual analysis, and peak timing.

The study also has limitations. Reported cases are an imperfect measure of infections, latent states may be difficult to identify, and one test period cannot establish that a model will generalize to every future outbreak.

## 9. Conclusion

Epidemic Dynamics & Forecasting provides a comparative framework for studying COVID-19 forecasting through mechanistic epidemiology, statistical time-series analysis, and machine learning, including recurrent neural networks.

By evaluating the four approaches on a later epidemic period, the project aims to determine how modelling assumptions affect predictive accuracy, generalization, and interpretability.

The main contribution is not simply identifying the model with the lowest error. It is understanding the trade-offs between data-driven flexibility and epidemiological structure, and identifying the conditions under which each approach succeeds or fails.

The final conclusion will be based on measured results from the common test set, including whether LSTM provides meaningful benefits over the simpler baselines.

## 10. Future Improvements

- Evaluate multiple rolling-origin test periods.
- Incorporate vaccination, testing, or intervention indicators where reliable data are available.
- Estimate uncertainty intervals alongside point forecasts.
- Compare short-term and longer-term forecasting horizons.
- Test LSTM sensitivity to sequence length, architecture, dropout, and training settings.
- Investigate improved observation models connecting latent epidemic states to reported cases.

**Project Status:** SIR and ARIMA have been implemented in earlier experiments, and XGBoost has been implemented. The LSTM notebook is prepared for the fourth model; its final metrics should be added after running it and checking that all models use the same target, dates, and evaluation protocol.
