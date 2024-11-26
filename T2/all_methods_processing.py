import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
import tensorflow as tf
import json
tf.compat.v1.disable_eager_execution()

# Load the dataset
data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Convert 'Income' to binary classification (Low = 1-4, High = 5-8)
data['Income_binary'] = data['Income'].apply(lambda x: 1 if x >= 5 else 0)

sensitive_features = ['Income_binary', 'Sex']

# Drop non-relevant columns
cols_to_drop = ['Diabetes_binary', 'Income']
X = data.drop(cols_to_drop, axis=1)
y = data['Diabetes_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Prepare datasets for fairness calculations
dataset_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                   df=pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
                                   label_names=['Diabetes_binary'],
                                   protected_attribute_names=sensitive_features)

dataset_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                   df=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
                                   label_names=['Diabetes_binary'],
                                   protected_attribute_names=sensitive_features)

# Define privileged and unprivileged groups
privileged_groups = [{'Income_binary': 1, 'Sex': 1}]  # High-income male group
unprivileged_groups = [{'Income_binary': 0, 'Sex': 0}]  # Low-income female group

# --- Step 1: Apply Reweighing ---
reweigher = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
reweighed_dataset = reweigher.fit_transform(dataset_train)

# Extract reweighted features and labels
X_train_reweighed = reweighed_dataset.features
X_train_reweighed = pd.DataFrame(X_train_reweighed, columns=X_train.columns)
y_train_reweighed = reweighed_dataset.labels.ravel()

# --- Step 2: Train the Adversarial Debiasing Model ---
# Create a model for adversarial debiasing (requires TensorFlow)
# Adversarial debiasing uses neural networks for bias mitigation.
sess = tf.compat.v1.Session()

ad_model = AdversarialDebiasing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    scope_name='DebiasingModel',
    debias=True,
    num_epochs=50,
    sess=sess
)

# Train the model using the reweighed dataset
ad_model.fit(dataset_train)

# --- Step 3: Apply Equalized Odds Post-Processing to the Output ---
# Make predictions on the test set using the adversarial debiasing model
y_pred_ad = ad_model.predict(dataset_test).labels

# Apply Equalized Odds post-processing to the adversarial model's predictions
eqodds_postprocessing = EqOddsPostprocessing(
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

# Fit the EqOdds model and apply post-processing to the debiasing model's predictions
dataset_pred_ad = dataset_test.copy()
dataset_pred_ad.labels = np.array(y_pred_ad).reshape(-1, 1)

eqodds_postprocessing = eqodds_postprocessing.fit(dataset_test, dataset_pred_ad)
y_pred_ad_eqodds = eqodds_postprocessing.predict(dataset_test).labels

# --- Step 4: Train Logistic Regression Model Without Bias Mitigation ---
log_reg_model = LogisticRegression(solver='newton-cg', random_state=42, max_iter=1000)
log_reg_model.fit(X_train, y_train)
y_pred_lr = log_reg_model.predict(X_test)

# Create dataset for Logistic Regression predictions
dataset_pred_lr = dataset_test.copy()
dataset_pred_lr.labels = np.array(y_pred_lr).reshape(-1, 1)

# --- Step 5: Define a Function to Calculate Metrics ---
def calculate_metrics(y_true, y_pred, dataset_test, dataset_pred, privileged_groups, unprivileged_groups):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
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
        "Demographic Parity": demographic_parity,
        "Equalized Odds": equalized_odds,
        "Predictive Parity": predictive_parity
    }

# --- Step 6: Calculate Metrics for All Models ---
results = {}

# Metrics for Logistic Regression without bias mitigation
results['Fairness - Logistic Regression with no mitigation'] = calculate_metrics(y_test, y_pred_lr, dataset_test, dataset_pred_lr,
                                                  privileged_groups, unprivileged_groups)

# Metrics for Adversarial Debiasing
results['Fairness - Adversarial Debiasing with Reweighing'] = calculate_metrics(y_test, y_pred_ad, dataset_test, dataset_pred_ad,
                                                     privileged_groups, unprivileged_groups)

# Metrics for Adversarial Debiasing after Equalized Odds Post-Processing
dataset_pred_ad_eqodds = dataset_test.copy()
dataset_pred_ad_eqodds.labels = np.array(y_pred_ad_eqodds).reshape(-1, 1)

results['Fairness - Adversarial Debiasing with Reweighing and Equalized Odds'] = calculate_metrics(
    y_test, y_pred_ad_eqodds, dataset_test, dataset_pred_ad_eqodds, privileged_groups, unprivileged_groups
)

# Output results as a JSON object
print(json.dumps(results, indent=2))