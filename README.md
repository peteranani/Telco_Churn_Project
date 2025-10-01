# Telco Customer Churn Analysis

This repository contains the exploratory data analysis (EDA) and machine learning workflow from the `Telco_Customer_Churn.ipynb` notebook. The goal is to understand the drivers of customer churn for a telecommunications provider and to build predictive models that identify at-risk customers.

## Dataset
- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Observations:** 7,043 customers
- **Features:** 20 customer demographic, account, and service attributes
- **Target:** `Churn` (73.4% stayed, 26.6% churned)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Notebook Highlights
- **Data Loading & Cleaning**
  - Converts `TotalCharges` to numeric, drops rows with missing values, and sets `customerID` as the index.
  - Encodes binary categories with `LabelEncoder`, applies one-hot encoding to multi-class features, and scales numeric columns with `StandardScaler`.
- **Exploratory Data Analysis**
  - Visualizes churn by contract type, internet service, payment method, technical support, online security, and paperless billing.
  - Examines numeric distributions (tenure, monthly charges, total charges) versus churn and inspects feature correlations.
- **Modeling Workflow**
  - Splits the data into training and test sets, then balances the training data with SMOTE.
  - Trains Logistic Regression, Random Forest, and XGBoost classifiers inside a scikit-learn pipeline and evaluates them on the hold-out test set.

## Model Performance (Test Set)
| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.74 | 0.51 | 0.74 | 0.61 |
| Random Forest | 0.78 | 0.57 | 0.66 | 0.61 |
| XGBoost | 0.76 | 0.53 | 0.66 | 0.59 |

*Class 1 (`Churn = Yes`) metrics are shown to highlight how well each model captures churned customers.*

## Getting Started
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Telco_Churn_Project.git
   cd Telco_Churn_Project
   ```
2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```
   If you do not plan to create a `requirements.txt`, install the dependencies manually:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
   ```
4. **Launch Jupyter and explore the notebook**
   ```bash
   jupyter notebook Telco_Customer_Churn.ipynb
   ```

## Repository Structure
- `Telco_Customer_Churn.ipynb` — main notebook with the full analysis and modeling workflow.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — raw Telco customer churn dataset.

## Future Improvements
- Tune hyperparameters for each model to push recall and F1 on the churn class higher.
- Experiment with additional algorithms (e.g., gradient boosting, lightGBM) and calibration techniques.
- Build feature importance plots or SHAP values to better explain model predictions to business stakeholders.
- Package the final model into an API or dashboard for real-time churn monitoring.

## Acknowledgements
Dataset provided by IBM and distributed via Kaggle user **blastchar**.
