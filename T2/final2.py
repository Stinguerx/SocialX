import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from imblearn.over_sampling import SVMSMOTE
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# Ignorar warnings
import warnings
# warnings.filterwarnings('ignore')

# Cargar y procesar datos
data = pd.read_csv("default_of_credit_card_clients.csv")
data.columns = ['ID', 'LIMITBAL', 'GENDER', 'EDUCATION', 'MARRIAGE', 'AGE'] + \
               [f'PAY{i}' for i in range(1, 7)] + \
               [f'BILLAMT{i}' for i in range(1, 7)] + \
               [f'PAYAMT{i}' for i in range(1, 7)] + ['DEFAULT']

# Variables sensibles
sensitive_features = ['GENDER', 'EDUCATION', 'MARRIAGE', 'AGE_GROUP']

# Crear nuevas características
data['PAYAVG'] = data[[f'PAY{i}' for i in range(1, 7)]].mean(axis=1)
data['AGE_GROUP'] = pd.cut(data['AGE'], bins=[20, 40, 100], labels=False)
data['PAY_AMT_AVG'] = data[[f'PAYAMT{i}' for i in range(1, 7)]].mean(axis=1)
data['BILL_AMT_AVG'] = data[[f'BILLAMT{i}' for i in range(1, 7)]].mean(axis=1)

# Definir columnas
cols_to_drop = ['ID', 'DEFAULT', 'AGE'] + [f'PAYAMT{i}' for i in range(1, 7)] + [f'BILLAMT{i}' for i in range(1, 7)]
X = data.drop(cols_to_drop, axis=1)
y = data['DEFAULT']
X['EDUCATION'] = X['EDUCATION'].map({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1})
X['GENDER'] = X['GENDER'].map({1: 0, 2: 1})
X['MARRIAGE'] = X['MARRIAGE'].map({0: 0, 1: 0, 2: 1, 3: 1})

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Balanceo con SVMSMOTE
smote = SVMSMOTE(k_neighbors=3, m_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

privileged_groups = [{'GENDER': 0}]
unprivileged_groups = [{'GENDER': 1}]

# Entrenar modelo original
rf_model = RandomForestClassifier(random_state=42, n_estimators=300, criterion='gini')
rf_model.fit(X_train_resampled, y_train_resampled)

# Predecir probabilidades
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1 (default)

# Establecer umbral A para la probabilidad
A = 0.25
y_pred = (y_pred_proba > A).astype(int)  # Convertir la probabilidad en 1 o 0 basándonos en el umbral

# Función para calcular métricas
def calculate_metrics(y_true, y_pred, dataset_test, dataset_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_proba)  # Usamos las probabilidades para ROC AUC
    
    # Métricas de equidad
    metric = ClassificationMetric(dataset_test, dataset_pred, 
                                   unprivileged_groups=unprivileged_groups, 
                                   privileged_groups=privileged_groups)
    demographic_parity = metric.statistical_parity_difference()
    equalized_odds = metric.average_odds_difference()
    predictive_parity = metric.disparate_impact()
    
    return {
        "Accuracy": accuracy,
        "F1": f1,
        "ROC AUC": roc,
        "Demographic Parity": demographic_parity,
        "Equalized Odds": equalized_odds,
        "Predictive Parity": predictive_parity
    }

# Dataset para AIF360
dataset_test = BinaryLabelDataset(favorable_label=0, unfavorable_label=1,
                                   df=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
                                   label_names=['DEFAULT'],
                                   protected_attribute_names=sensitive_features)

# Crear el dataset de predicción a partir del conjunto de test
dataset_pred = dataset_test.copy()
dataset_pred.labels = np.array(y_pred).reshape(-1, 1)  # Aquí asignamos las predicciones al dataset

# Evaluar el modelo original y calcular las métricas
results = {}
results['Original'] = calculate_metrics(y_test, y_pred, dataset_test, dataset_pred)
print("Resultados del modelo original:")
print(results)

# Preprocesamiento con Reweighing
reweighing = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_train = BinaryLabelDataset(favorable_label=0, unfavorable_label=1,
                                    df=pd.concat([X_train_resampled, y_train_resampled], axis=1),
                                    label_names=['DEFAULT'],
                                    protected_attribute_names=sensitive_features)

original_weights = dataset_train.instance_weights
print("Pesos originales (antes de Reweighing):")
print(original_weights)

# Aplicar Reweighing para transformar los pesos
reweighed_dataset = reweighing.fit_transform(dataset_train)

reweighed_weights = reweighed_dataset.instance_weights
print("Pesos después del Reweighing:")
print(reweighed_weights)

# Extraer las características y etiquetas transformadas
X_train_reweighed = reweighed_dataset.features
X_train_reweighed = pd.DataFrame(X_train_reweighed, columns=X_train.columns)
y_train_reweighed = reweighed_dataset.labels.ravel()

# Entrenar el modelo con el dataset reweighed
rf_model = RandomForestClassifier(random_state=42, n_estimators=300, criterion='gini')
rf_model.fit(X_train_reweighed, y_train_reweighed)

# Predecir probabilidades con el modelo reweighed
y_pred_proba_reweighing = rf_model.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1 (default)

# Aplicar el umbral A al modelo reweighed
y_pred_reweighing = (y_pred_proba_reweighing > A).astype(int)

# Actualizar etiquetas del dataset de predicción
dataset_pred.labels = y_pred_reweighing.reshape(-1, 1)

# Calcular métricas
results['Preprocessing (Reweighing)'] = calculate_metrics(y_test, y_pred_reweighing, dataset_test, dataset_pred)

print(results)