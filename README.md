# Global GDP Forecasting and Crisis Surveillance

## A Hybrid Data Science Framework for Country-Level Macroeconomic Risk Analysis

This project is an end-to-end data science system for forecasting country-level GDP growth and monitoring macroeconomic crisis risk. It combines econometric modelling, machine learning, deep learning, an Early Warning System classifier, and a deployed interactive dashboard.

The system uses IMF/WEO-style macroeconomic panel data to generate GDP growth forecasts, crisis probabilities, risk levels, model comparison results, and country-level surveillance insights.

---

## Project Overview

Macroeconomic forecasting is difficult because GDP growth is influenced by inflation, trade, debt, fiscal balance, investment, savings, current account conditions, and unexpected global shocks.

This project addresses two connected problems:

1. **GDP Growth Forecasting**  
   Predict future country-level GDP growth using historical macroeconomic indicators.

2. **Crisis Risk Surveillance**  
   Estimate crisis probability and classify countries into risk categories using an Early Warning System.

The final output is deployed as an interactive dashboard where users can select a country and forecast year to view GDP forecasts, IMF benchmark comparisons, crisis probability, risk level, and high-risk or low-risk country rankings.

---

## Key Features

- Country-level GDP growth forecasting
- Composite Macroeconomic Instability Index
- Econometric baseline modelling
- Machine Learning forecasting models
- LSTM deep learning benchmark
- Early Warning System classifier
- Crisis probability and risk-level classification
- 2024-2026 IMF benchmark comparison
- 2027-2030 scenario-based forecasting
- Interactive dashboard with country-year selection
- Backend API deployment using FastAPI and Render
- Frontend deployment using Next.js and Vercel

---

## Dataset

The project uses IMF/WEO-style macroeconomic panel data.

### Dataset Structure

- **Countries:** 175
- **Observations:** 5,507 country-year records
- **Core indicators:** 12 macroeconomic indicators
- **Unit of analysis:** Country-year observation
- **Target variable:** GDP Growth

### Main Indicators

- GDP Growth
- Inflation
- Exports
- Imports
- Debt
- Revenue
- Expenditure
- Savings
- Investment
- Current Account
- Fiscal Balance
- Macroeconomic Instability Index

---

## Time Horizon

The dataset and forecasts are divided into three periods:

| Period | Purpose |
|---|---|
| 1995-2023 | Historical observed data used for training and testing |
| 2024-2026 | IMF benchmark period used for model comparison |
| 2027-2030 | Scenario-based recursive forecast period |

The 2027-2030 forecasts are model-based scenario projections and should not be interpreted as guaranteed real-world outcomes.

---

## Project Architecture

```text
IMF / WEO-style Data
        |
        v
Data Cleaning and Preprocessing
        |
        v
Feature Engineering
        |
        |-- Lag Features
        |-- Rolling Features
        |-- Volatility Features
        |-- Difference Features
        |-- Instability Index
        |
        v
Model Development
        |
        |-- Econometric Models
        |-- Machine Learning Models
        |-- LSTM Deep Learning Model
        |-- Early Warning System Classifier
        |
        v
Model Comparison and Scenario Forecasting
        |
        v
FastAPI Backend
        |
        v
Next.js Frontend Dashboard
```
## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · Statsmodels
Linearmodels · SHAP · TensorFlow · Streamlit
## Methodology

This project followed a structured end-to-end data science methodology that covered data collection, preprocessing, feature engineering, instability measurement, forecasting model development, early warning classification, model comparison, scenario forecasting, and dashboard deployment. The purpose of the methodology was not only to build a predictive model, but to create a complete macroeconomic surveillance framework that can forecast GDP growth and identify country-level risk conditions.

---

### 1. Data Collection and Source

The project used IMF/WEO-style macroeconomic panel data containing country-year observations across multiple economies. Each row in the dataset represents one country in one specific year, while each column represents a macroeconomic indicator.

The main target variable for the forecasting models was **GDP Growth**. Other macroeconomic indicators were used as explanatory variables and risk signals.

The dataset included indicators such as:

- GDP growth
- Inflation
- Exports
- Imports
- Debt
- Revenue
- Expenditure
- Savings
- Investment
- Current account
- Fiscal balance
- Macroeconomic instability indicators

The dataset contained approximately **175 countries** and **5,507 country-year observations**, making it suitable for panel-data modelling and country-level macroeconomic analysis.

---

### 2. Time Period Division

The dataset was divided into three separate time periods to maintain logical consistency between observed data, benchmark comparison, and future scenario forecasting.

| Period | Purpose |
|---|---|
| 1995-2023 | Historical observed period used for training, testing, and validation |
| 2024-2026 | IMF benchmark projection period used for comparing model forecasts |
| 2027-2030 | Scenario-based forecast period generated using recursive forecasting |

The period **1995-2023** was treated as the historical observed dataset. This period was used for learning past country-level macroeconomic patterns.

The period **2024-2026** was treated as the IMF benchmark period. Since IMF projection values were available for these years, the model-generated forecasts could be compared against these benchmark values.

The period **2027-2030** was treated as a future scenario period. Since actual or benchmark values were not available for these years in the project, these forecasts were interpreted as model-based scenario projections rather than guaranteed real-world outcomes.

---

### 3. Data Preprocessing

Data preprocessing was performed to prepare the raw macroeconomic panel data for modelling. Since the dataset contained multiple countries and years, it was important to ensure that the data was clean, consistent, and properly ordered.

The major preprocessing steps included:

- Removing irrelevant or duplicate records
- Standardizing country and year columns
- Converting macroeconomic indicators into numeric format
- Handling missing values
- Sorting the data by country and year
- Separating observed, benchmark, and scenario periods
- Preparing country-year panel structure for feature engineering

Missing-value handling was an important step because macroeconomic datasets often contain incomplete records for some countries and years. Care was taken to avoid introducing future information into past observations during preprocessing.

---

### 4. Feature Engineering

Feature engineering was one of the most important stages of the methodology. Raw macroeconomic indicators alone may not fully capture historical momentum, instability, or delayed effects. Therefore, several time-aware features were created.

The main feature groups were:

#### 4.1 Lag Features

Lag features were created to use previous-year values as predictors for current or future GDP growth.

Formula:

```text
X_lag1(i,t) = X(i,t-1)
```

where:

- `i` represents the country
- `t` represents the year
- `X(i,t-1)` represents the previous-year value of variable `X`

Examples:

```text
GDP_Growth_lag1
Exports_lag1
Imports_lag1
Inflation_lag1
Debt_lag1
Instability_Index_lag1
```

Lag features were important because macroeconomic variables often show persistence over time.

---

#### 4.2 Rolling Features

Rolling features were created to capture short-term historical trends. A three-year rolling mean was used to summarize recent macroeconomic behavior.

Formula:

```text
RollingMean3(i,t) = [X(i,t-1) + X(i,t-2) + X(i,t-3)] / 3
```

Example:

```text
GDP_Growth_rollmean3
```

Rolling features helped smooth year-to-year fluctuations and capture recent economic momentum.

---

#### 4.3 Volatility Features

Volatility features were created to measure instability or fluctuation in macroeconomic indicators over time. These features were mainly based on rolling standard deviation.

Formula:

```text
RollingStd3(i,t) = standard deviation of X over previous 3 years
```

Volatility features helped capture whether a country’s macroeconomic conditions were stable or unstable.

---

#### 4.4 Difference Features

Difference features were created to measure year-to-year changes in macroeconomic indicators.

Formula:

```text
Delta X(i,t) = X(i,t) - X(i,t-1)
```

Examples:

```text
Debt_diff_lag1
Revenue_diff_lag1
Expenditure_diff_lag1
Savings_diff_lag1
Investment_diff_lag1
```

These features helped identify whether a country’s fiscal or macroeconomic condition was improving or worsening over time.

---

### 5. Target Leakage Prevention

A major methodological concern in forecasting projects is **target leakage**. Target leakage occurs when a model uses information that would not be available at the time of prediction.

In this project, special attention was given to avoid leakage. For example, the composite instability index was originally related to current-year macroeconomic conditions, including GDP growth. If the current-year instability index were directly used to predict current-year GDP growth, it could create leakage.

To prevent this, the modelling process used lagged versions of such variables, especially:

```text
Instability_Index_lag1
GDP_Growth_lag1
GDP_Growth_rollmean3
```

This ensured that the model used historical information instead of current-year target-derived information.

---

### 6. Composite Macroeconomic Instability Index

A Composite Macroeconomic Instability Index was constructed to summarize macroeconomic vulnerability for each country-year observation. The purpose of the index was to capture hidden economic stress that may not be visible through GDP growth alone.

The index combined multiple macroeconomic stress components, such as:

- Abnormal deviations from historical patterns
- Rolling volatility
- Sudden shocks in macroeconomic indicators
- Fiscal and external-sector stress

A general mathematical representation of the index is:

```text
I(i,t) = Sum w_j * Z_j(i,t)
```

where:

- `I(i,t)` is the instability index for country `i` in year `t`
- `Z_j(i,t)` is the standardized value of indicator `j`
- `w_j` is the weight assigned to indicator `j`

Before combining indicators, variables were standardized so that indicators with different units and scales could be compared.

Standardization formula:

```text
Z = (X - mean) / standard deviation
```

The lagged version of the instability index was used in later forecasting and classification models.

---

### 7. Econometric Baseline Modelling

Econometric models were used as the baseline layer because they are interpretable and commonly used in macroeconomic analysis. This layer helped establish a statistical benchmark against which machine learning and deep learning models could be compared.

The econometric models included:

- Pooled Ordinary Least Squares
- Fixed Effects model

The general regression form was:

```text
GDP_Growth(i,t) = beta0 + beta1X1(i,t-1) + beta2X2(i,t-1) + ... + epsilon(i,t)
```

where:

- `GDP_Growth(i,t)` is the target variable
- `X1, X2, ...` are lagged macroeconomic predictors
- `beta` values are estimated coefficients
- `epsilon(i,t)` is the error term

The econometric layer provided interpretability, but it was limited in capturing nonlinear relationships among macroeconomic indicators.

---

### 8. Machine Learning Modelling

Machine learning models were developed to capture nonlinear relationships and interactions between macroeconomic indicators. Several supervised regression models were trained and evaluated.

The machine learning models included:

- Linear Regression
- Elastic Net
- Support Vector Regression
- Random Forest
- XGBoost

The models were evaluated using regression metrics such as:

```text
RMSE
MAE
R2
```

Among the tested models, **Random Forest** achieved the best GDP forecasting performance. Random Forest was effective because it can model nonlinear relationships and interactions among lagged, rolling, fiscal, trade, and volatility features.

The Random Forest prediction can be represented as:

```text
y_hat = (1 / B) * Sum T_b(X)
```

where:

- `B` is the number of decision trees
- `T_b(X)` is the prediction from tree `b`
- `y_hat` is the final averaged prediction

---

### 9. Model Explainability

Model explainability was included to understand which features contributed most to GDP growth prediction. Feature importance and SHAP-based interpretation were used to identify the most influential predictors.

Important features included:

- GDP_Growth_lag1
- GDP_Growth_rollmean3
- Exports_lag1
- Imports_lag1
- Inflation_lag1_log
- Fiscal and volatility indicators

SHAP explains predictions using the idea:

```text
Prediction = Base Value + Sum of Feature Contributions
```

This helped interpret the behaviour of machine learning models and provided transparency for the forecasting layer.

---

### 10. LSTM Deep Learning Model

An LSTM model was developed as a deep learning benchmark for sequential GDP growth forecasting. LSTM models are suitable for time-series data because they can learn temporal dependencies using memory cells and gates.

The LSTM architecture includes:

- Input gate
- Forget gate
- Output gate
- Memory cell

The purpose of using LSTM was to test whether a sequential deep learning approach could outperform traditional machine learning methods.

However, the LSTM model did not outperform Random Forest. This was mainly because annual macroeconomic panel data contains relatively short sequences, structural breaks, and noise. Therefore, Random Forest remained the best GDP forecasting model in this project.

---

### 11. Early Warning System Classifier

The Early Warning System was developed to classify country-year observations into crisis-risk categories. Unlike the GDP forecasting layer, the EWS layer was treated as a classification problem.

The target variable was a binary warning flag:

```text
Crisis_Flag = 1 means warning or high-risk condition
Crisis_Flag = 0 means non-warning or low-risk condition
```

The classification models included:

- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost

The EWS classifier estimated:

```text
P(Crisis = 1 | X)
```

This probability was then converted into risk categories:

```text
Low Risk
Moderate Risk
High Risk
```

The best-performing EWS model was **Extra Trees**, which achieved strong classification performance based on ROC-AUC.

---

### 12. Model Evaluation Metrics

Different evaluation metrics were used for regression and classification tasks.

#### Regression Metrics

##### RMSE

```text
RMSE = sqrt(mean((y - y_hat)^2))
```

RMSE penalizes larger errors more strongly.

##### MAE

```text
MAE = mean(|y - y_hat|)
```

MAE measures the average absolute prediction error.

##### R2

```text
R2 = 1 - (Residual Sum of Squares / Total Sum of Squares)
```

R2 measures the proportion of variance explained by the model.

---

#### Classification Metrics

##### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

##### Precision

```text
Precision = TP / (TP + FP)
```

##### Recall

```text
Recall = TP / (TP + FN)
```

##### F1-Score

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

##### ROC-AUC

ROC-AUC measures the ability of the classifier to distinguish between warning and non-warning cases across different thresholds.

---

### 13. Final Model Selection

After evaluating all models, the final selected models were:

| Task | Selected Model | Reason |
|---|---|---|
| GDP Growth Forecasting | Random Forest | Best regression performance |
| Crisis Risk Classification | Extra Trees | Best EWS classification performance |
| Econometric Benchmark | Pooled OLS | Best interpretable baseline |
| Deep Learning Benchmark | LSTM | Sequential modelling comparison |

Random Forest was selected for GDP forecasting because it achieved the lowest prediction error among the forecasting models. Extra Trees was selected for EWS because it achieved the strongest ROC-AUC performance.

---

### 14. IMF Benchmark Comparison: 2024-2026

For the years 2024-2026, IMF projection values were available. Therefore, model predictions were compared with IMF benchmark projections.

Prediction error was calculated as:

```text
Prediction_Error = IMF_Projection - Model_Prediction
```

Absolute error was calculated as:

```text
Absolute_Error = |IMF_Projection - Model_Prediction|
```

This comparison helped identify how close the model predictions were to IMF benchmark projections and which countries had the largest forecast deviations.

---

### 15. Scenario Forecasting: 2027-2030

For the years 2027-2030, recursive scenario forecasting was used. Since benchmark projections were not available for these years in the project, the forecasts were treated as model-based scenarios.

Recursive forecasting followed the logic:

```text
Use 2026 features to forecast 2027
Use 2027 predicted values to forecast 2028
Use 2028 predicted values to forecast 2029
Use 2029 predicted values to forecast 2030
```

This approach allowed the model to generate forward-looking estimates beyond the available benchmark horizon.

However, these forecasts were interpreted cautiously because recursive forecasting can accumulate errors over time.

---

### 16. Dashboard Development and Deployment

The final stage of the methodology was dashboard development and deployment. The purpose was to convert model outputs into an accessible decision-support interface.

The backend was developed using **FastAPI**. It reads processed CSV files and serves the outputs as JSON through API endpoints.

The backend was deployed on **Render**.

The frontend was developed using **Next.js**, **React**, and **TypeScript**. It allows users to interact with the model results using country and year selection.

The frontend was deployed on **Vercel**.

The deployment flow was:

```text
Model Outputs -> FastAPI Backend -> JSON API -> Next.js Dashboard -> User
```

The dashboard displays:

- Predicted GDP growth
- IMF benchmark values where available
- Crisis probability
- Risk level
- Early warning flag
- Country-level forecast trends
- High-risk and low-risk country rankings
- Model comparison results
- AI-style interpretation

---

### 17. Testing and Validation

Testing and validation were performed at multiple levels.

#### Model-Level Testing

The regression models were tested using RMSE, MAE, and R2. The classification models were tested using accuracy, precision, recall, F1-score, and ROC-AUC.

#### Benchmark Testing

For 2024-2026, model forecasts were compared with IMF benchmark projections using prediction error and absolute error.

#### Dashboard Testing

The dashboard was tested by selecting different countries and years and verifying whether the correct forecast, risk, and comparison values were displayed.

#### Backend Testing

The backend was tested using API endpoints such as health check and data retrieval endpoints.

---

### 18. Summary of Methodology

The methodology followed a complete data science workflow from raw macroeconomic data to deployed dashboard. The project began with IMF/WEO-style data collection, followed by preprocessing and feature engineering. A composite instability index was created to measure macroeconomic vulnerability. Econometric models, machine learning models, and LSTM were trained and compared for GDP forecasting. An Early Warning System classifier was developed to estimate crisis probability and assign risk levels. Finally, the results were deployed through a FastAPI backend and Next.js dashboard.

Overall, the methodology enabled the project to combine forecasting, instability measurement, crisis-risk classification, model comparison, scenario forecasting, and deployment into one integrated macroeconomic surveillance framework.
