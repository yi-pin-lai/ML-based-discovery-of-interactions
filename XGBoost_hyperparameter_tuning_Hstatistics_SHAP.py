#!/usr/bin/env python
# coding: utf-8

# ============================================================
# CAD PRS x Risk Factors Interaction Analysis
# ============================================================
#
# This script demonstrates:
# 1. Data preprocessing
# 2. Hyperparameter tuning
# 3. XGBoost model fitting
# 4. H-statistic interaction analysis
# 5. SHAP analysis
#
# ============================================================

# %% Load packages

import numpy as np
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import artemis as atm
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod, GreenwellMethod, SejongOhMethod
from artemis.interactions_methods.model_specific import ConditionalMinimalDepthMethod, SplitScoreMethod
import warnings
warnings.filterwarnings("ignore")
import sys
import utils
import random
from artemis.interactions_methods.model_agnostic import FriedmanHStatisticMethod
from sklearn.metrics import make_scorer

# %% Load data

data = pd.read_csv("/Users/yl503/Documents/ML_projects/CAD_PRS_train_multi3/ukbb_cadprs_pce_eur_phen_eur_select2_train60_3.tsv", sep='\t')


for c in data.columns:
    col_type = data[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        data[c] = data[c].astype('category')

data['hxDiabetes_Type_1'] = data['hxDiabetes_Type_1'].astype('category')
data['hxDiabetes_Type_2'] = data['hxDiabetes_Type_2'].astype('category')
data['hxHypercholesterolemia'] = data['hxHypercholesterolemia'].astype('category')
data['hxHypertension'] = data['hxHypertension'].astype('category')
data['Sex'] = data['Gender'].astype('category')
data['Smoking_status'] = data['Smoking_status'].astype('category')
data.info()


data['Log_ALP'] = np.log(data['ALP'])
data['Log_Creactiveprotein'] = np.log(data['Creactiveprotein'])
data['Log_CystatinC'] = np.log(data['CystatinC'])

categorical_columns_subset = [
    "Sex",
    "hxDiabetes_Type_1",
    "hxDiabetes_Type_2",
    "hxHypercholesterolemia",
    "hxHypertension",
    "Smoking_status",]

numerical_columns_subset = [
    'CAD_PRS', 'Age', 'BMI', 
    'HbA1c', 'LDL_mg_dl', 'HDL_mg_dl', 'Log_TG', 'SBP', 
    'eGFR_CysC', 'Log_lpa', 'Log_Creactiveprotein', 
    'pc1', 'pc2', 'pc3', 'pc4', 'pc5',
]


scaler = StandardScaler()
num_cols = ['Age', 'BMI',
    'HbA1c', 'LDL_mg_dl', 'HDL_mg_dl', 'SBP', 
    'eGFR_CysC', 
]
data[num_cols] = scaler.fit_transform(data[num_cols])
data = pd.DataFrame(data)


X = df_filtered[categorical_columns_subset + numerical_columns_subset]
y = df_filtered[["Myocardial_Infarctionfu", "days2Myocardial_Infarction"]]


# Define custom mapping
smoking_mapping = {
    'Never': 0,
    'Previous': 1,
    'Current': 2
}

# Apply mapping
X['Smoking_status'] = X['Smoking_status'].map(smoking_mapping)



# %% Split train/test sets
X_train, X_test, y_train_time, y_test_time, y_train_status, y_test_status = train_test_split(
    X,
    y["days2Myocardial_Infarction"],
    y["Myocardial_Infarctionfu"],
    test_size=0.3,
    random_state=42
)

# In the XGBRegressor API for Cox, positive values = event occurred, negative values = censored.
y_train_surv = np.where(y_train_status == 1, y_train_time, -y_train_time)
y_test_surv  = np.where(y_test_status == 1, y_test_time, -y_test_time)



# %% Hyperparameter Grid Search for XGBoost
# Define the hyperparameter grid for XGBoost
param_grid = {
    #'min_child_weight': [1, 5, 10],
    'max_depth': [2, 3, 5],
    #'learning_rate': [0.01, 0.1],
    #'n_estimators': [100, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000]      
    #'gamma': [0, 0.1, 1]
}



# %% Fit model
xgb = XGBRegressor(
    random_state=92,
    objective="survival:cox",
    eval_metric="cox-nloglik",
    tree_method="hist",
    n_jobs=-1,
    verbosity=0,
    enable_categorical=True,
    #subsample=0.7,
    #colsample_bytree=0.7
)


# %% Fixed custom c-index function for 1D arrays
def xgboost_c_index_lifelines(y_true, y_pred_risk):
    """
    y_true: 1D array where absolute values are times, and negative values are censored.
    y_pred_risk: Continuous risk scores outputted by XGBoost
    """
    # Extract times and indicators from the encoded format
    event_time = np.abs(y_true)
    event_indicator = np.where(y_true > 0, 1, 0)
    
    # Invert risk scores because lifelines expects higher values = longer survival time
    y_pred_survival_ranking = -y_pred_risk
    
    return concordance_index(event_time, y_pred_survival_ranking, event_indicator)



# %% Properly wrap the function into a Scikit-Learn Scorer
c_index_scorer = make_scorer(xgboost_c_index_lifelines, response_method='predict')


# %% Configuration of grid search
grid_search = GridSearchCV(
    estimator=xgb,              
    param_grid=param_grid,      
    cv=5,                       
    scoring=c_index_scorer,     
    n_jobs=1,
    error_score='raise'      
)

# %% Fit the grid search
grid_search.fit(
    X_train, y_train_surv,
    eval_set=[(X_test, y_test_surv)],
    verbose=False
)

# %% Print summary results
print("Best C-Index Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)

allscores = grid_search.cv_results_['mean_test_score']
allscores

# Get the best hyperparameters and their values
best_params = grid_search.best_params_
best_params


# %% Get the performance on the test set
# Retrieve the best tuned model from the grid search
best_cox_model = grid_search.best_estimator_

# Predict risk scores for the test set
# XGBoost outputs higher values for higher risk (earlier event)
test_risk_scores = best_cox_model.predict(X_test)

# Decode the test targets back to positive times and event indicators
test_times = np.abs(y_test_surv)
test_events = np.where(y_test_surv > 0, 1, 0)

# Invert risk scores for lifelines (higher value = longer survival)
test_survival_ranking = -test_risk_scores

# Calculate the final Test C-index
final_test_c_index = concordance_index(
    test_times, 
    test_survival_ranking, 
    test_events
)

print(f"Final validation C-index on test set: {final_test_c_index:.4f}")


# %% Fit the model using the best parameters
params_xgb = {
    "learning_rate": 0.01,
    "max_depth": 2,
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "subsample": 0.5,
    "min_child_weight": 1,
    "n_estimators": 1000,
    "tree_method": "hist",
    "n_jobs": -1,
    "enable_categorical":True
}

xgbmodel = XGBRegressor(**params_xgb)

xgbmodel.fit(X_train, y_train_surv)


# Predict risk scores
predictions = xgbmodel.predict(X_test)

# Compute C-index
c_index = concordance_index(
    y_test_time,
    -predictions,   # negative because higher risk = shorter survival
    y_test_status
)

print("C-index:", c_index)

# Get random selection of 3000 observations
random.seed(8)
X_exp = random.choices(X.to_numpy(), k=3000)
X_exp = pd.DataFrame(X_exp, columns=X.columns)

# %% Calculate Friedman H statistic
h_stat_xgb = FriedmanHStatisticMethod()
h_stat_xgb.fit(xgbmodel, X_exp_clean, predict_function=xgb_safe_predict, show_progress=True, n=1500)


# Overall interaction plot
fig, ax = plt.subplots(figsize=(10, 4))
h_stat_xgb.plot('bar_chart_ova', ax=ax, top_k=30)

# Pairwise interactions
fig, ax = plt.subplots(figsize=(10, 8))
h_stat_xgb.plot(vis_type='bar_chart', ax=ax, top_k=50)

# Heatmap of interaction plot
h_stat_xgb_heatmap = h_stat_xgb.plot(
    vis_type='heatmap', 
    figsize=(18, 16), 
    annot=True,         
    annot_fmt='.1g',    
    font_size=10
)
h_stat_xgb_heatmap


# %% SHAP interaction analysis
# SHAP based importance of variables
explainer_xgb = shap.TreeExplainer(xgbmodel)
shap_values_xgb = explainer_xgb(X)

# SHAP importance plot
shap.summary_plot(shap_values_xgb, X, plot_type="bar")

# SHAP based importance of interactions
shap_values_xgb_interactions = explainer_xgb.shap_interaction_values(X)

# SHAP summary plot to reveal global feature interactions
shap.summary_plot(shap_values_xgb_interactions, X, max_display=50, show=TRUE)

# SHAP beewarm plot 
shap.summary_plot(shap_values_xgb_interactions, X, plot_type="compact_dot", max_display=50, show=TRUE)

# SHAP dependence plot for CAD PRS and Age
shap.dependence_plot(
    ("CAD_PRS", "Age"),
    shap_values_xgb_interactions, X, show=TRUE
)

# SHAP dependence plot for CAD PRS and Sex
shap.dependence_plot(
    ("CAD_PRS", "Sex"),
    shap_values_xgb_interactions, X, show=TRUE
)





