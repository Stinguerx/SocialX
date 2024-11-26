import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
import json

# Load the data
data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Convert 'Income' to binary classification (Low = 1-4, High = 5-8)
data['Income_binary'] = data['Income'].apply(lambda x: 1 if x >= 5 else 0)

# Define sensitive features (Income_binary and Sex)
sensitive_features = ['Income_binary', 'Sex']

# Drop non-relevant columns
cols_to_drop = ['Diabetes_binary', 'Income']
X = data.drop(cols_to_drop, axis=1)
y = data['Diabetes_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and fit the Logistic Regression model
log_reg_model = LogisticRegression(solver='newton-cg',random_state=42, max_iter=1000)

# Fit the model on the training set (with no reweighing for now)
log_reg_model.fit(X_train, y_train)
y_pred = log_reg_model.predict(X_test)

# Define a function to calculate fairness and performance metrics
def calculate_metrics(y_true, y_pred, dataset_test, dataset_pred, privileged_groups, unprivileged_groups):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # roc = roc_auc_score(y_true, log_reg_model.predict_proba(X_test)[:, 1])
    
    # Fairness metrics
    metric = ClassificationMetric(dataset_test, dataset_pred, 
                                   unprivileged_groups=unprivileged_groups, 
                                   privileged_groups=privileged_groups)
    
    demographic_parity = metric.statistical_parity_difference()
    equalized_odds = metric.average_odds_difference()
    predictive_parity = metric.disparate_impact()
    
    return {
        "Accuracy": accuracy,
        "F1": f1,
        # "ROC AUC": roc,
        "Demographic Parity": demographic_parity,
        "Equalized Odds": equalized_odds,
        "Predictive Parity": predictive_parity
    }

# Prepare datasets for fairness calculations
dataset_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                   df=pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
                                   label_names=['Diabetes_binary'],
                                   protected_attribute_names=sensitive_features)

dataset_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                   df=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
                                   label_names=['Diabetes_binary'],
                                   protected_attribute_names=sensitive_features)

# Create dataset for predictions
dataset_pred = dataset_test.copy()
dataset_pred.labels = np.array(y_pred).reshape(-1, 1)

# Evaluate the model without reweighing (original model)
results = {}

# Fairness based on 'Income_binary' and 'Sex'
privileged_groups = [{'Income_binary': 1, 'Sex': 1}]  # High-income male group
unprivileged_groups = [{'Income_binary': 0, 'Sex': 0}]  # Low-income female group
results['Fairness - Logistic Regression with no mitigation'] = calculate_metrics(y_test, y_pred, dataset_test, dataset_pred,
                                                  privileged_groups, unprivileged_groups)

# Apply Reweighing for fairness with both sensitive attributes
reweigher = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
reweighed_dataset = reweigher.fit_transform(dataset_train)

# Extract reweighted features and labels
X_train_reweighed = reweighed_dataset.features
X_train_reweighed = pd.DataFrame(X_train_reweighed, columns=X_train.columns)
y_train_reweighed = reweighed_dataset.labels.ravel()

# Train the Logistic Regression model with reweighed dataset
log_reg_model.fit(X_train_reweighed, y_train_reweighed, sample_weight=reweighed_dataset.instance_weights)

# Predict on the test set after reweighing
y_pred_reweighed = log_reg_model.predict(X_test)

# Create dataset for reweighed predictions
dataset_pred_reweighed = dataset_test.copy()
dataset_pred_reweighed.labels = np.array(y_pred_reweighed).reshape(-1, 1)

# Evaluate the model with reweighing
# results['Fairness - Logistic Regression with only Reweighing'] = calculate_metrics(y_test, y_pred_reweighed, dataset_test, dataset_pred_reweighed,
                                                          # privileged_groups, unprivileged_groups)

## Apply Equalized Odds Post-Processing to the reweighed predictions
eqodds_postprocessing = EqOddsPostprocessing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

# Fit the EqOddsPostprocessing model to the reweighed dataset predictions
dataset_pred_reweighed_eqodds = dataset_pred_reweighed.copy()
dataset_pred_reweighed_eqodds.labels = np.array(y_pred_reweighed).reshape(-1, 1)

eqodds_postprocessing = eqodds_postprocessing.fit(dataset_test, dataset_pred_reweighed_eqodds)

# Apply the post-processing transformation
y_pred_eqodds = eqodds_postprocessing.predict(dataset_test).labels

# Create dataset for Equalized Odds predictions
dataset_pred_eqodds = dataset_test.copy()
dataset_pred_eqodds.labels = np.array(y_pred_eqodds).reshape(-1, 1)

# Evaluate the model after Equalized Odds Post-Processing
results['Fairness - Logistic Regression after Reweighing and Equalized Odds Post-Processing'] = calculate_metrics(
    y_test, y_pred_eqodds, dataset_test, dataset_pred_eqodds, privileged_groups, unprivileged_groups
)

# Output results as a JSON object
print(json.dumps(results, indent=2))