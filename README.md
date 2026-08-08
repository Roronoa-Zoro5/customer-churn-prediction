# Customer Churn Prediction

A notebook-based machine learning project for predicting customer churn using the Telco Customer Churn dataset. The repository demonstrates data exploration, preprocessing, feature engineering, model training, and evaluation for a real-world churn problem.

## Problem statement

This project solves the customer churn prediction problem: it identifies which customers are most likely to leave a subscription-based service. In a telecom business context, the model predicts whether a customer will churn based on demographic, service, and billing features.

### Why this matters

- helps retention teams reduce customer loss
- improves marketing targeting for churn prevention offers
- supports business decisions with data-driven risk scoring
- can be applied to telecom, digital subscription, insurance, or any recurring-service company with customer usage data

## What is included

- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` — raw Telco customer churn dataset.
- `notebooks/01_EDA.ipynb` — exploratory data analysis, data quality checks, and visualizations for churn drivers.
- `notebooks/03_modelling.ipynb` — model development workflow, including preprocessing, pipeline construction, and evaluation.
- `notebooks/x_train_scaled.csv`, `notebooks/x_test_scaled.csv`, `notebooks/y_train.csv`, `notebooks/y_test.csv` — saved processed datasets from the modeling notebook.
- `requirements.txt` — Python dependencies.

## Project approach

Key steps in the modeling workflow:

1. load the Telco churn dataset from `WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. prepare features by imputing missing `TotalCharges` values and encoding categorical fields
3. use a pipeline with `ColumnTransformer`, `SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder`, and `SelectKBest`
4. split the data into training and test sets with `train_test_split(test_size=0.2, random_state=42)`
5. train multiple classifiers and compare performance:
   - `RandomForestClassifier`
   - `XGBClassifier`
   - `CatBoostClassifier`
6. evaluate using accuracy and ROC AUC metrics

### Demonstrated model performance

The modeling notebook shows example results such as:

- Random Forest accuracy ~0.767 and AUC ROC ~0.708
- XGBoost accuracy ~0.811 and AUC ROC ~0.863    (after grid search)
- CatBoost accuracy ~0.806 and AUC ROC ~0.857

## Where it can be used

This project can be used by:

- data scientists studying churn prediction workflows
- analysts building churn-prevention models for telecom or subscription businesses
- students learning end-to-end machine learning in Python
- engineers preparing to convert notebooks into production-ready training or inference scripts

## How to run locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd customer_churn_prediction
```

### 2. Create and activate a Python environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebooks

```bash
jupyter notebook
```

Then open:

- `notebooks/01_EDA.ipynb`
- `notebooks/03_modelling.ipynb`

### 5. Run the notebooks

- execute the cells in `01_EDA.ipynb` to explore data distributions, churn counts, and feature relationships
- execute the cells in `03_modelling.ipynb` to run preprocessing, train models, and evaluate predictions

## Running on another local machine

To run this project on another local server or developer machine:

1. copy the repository and dataset to that machine
2. create a new Python virtual environment there
3. install packages with `pip install -r requirements.txt`
4. start Jupyter and open the notebook files

If you want to serve predictions from a local web service, convert the notebook pipeline into a Python script or API app with Flask/FastAPI and load the trained model or pipeline.

## Notes

- The repository is currently notebook-first and does not include a production web server.
- To make the project production-ready, add a script for training, model persistence, and an inference API.

## Suggested next steps

- extract preprocessing and model code into a reusable Python module
- save the best model to disk using `joblib` or `pickle`
- add a simple local Flask/FastAPI server for churn predictions
- implement more robust validation, hyperparameter tuning, and feature importance analysis
