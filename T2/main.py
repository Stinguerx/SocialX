import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SVMSMOTE
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Cargar los datos
data = pd.read_csv("default_of_credit_card_clients.csv")

# Renombrar columnas para facilitar el trabajo
data.columns = ['ID', 'LIMIT_BAL', 'GENDER', 'EDUCATION', 'MARRIAGE', 'AGE'] + \
               [f'PAY_{i}' for i in range(1, 7)] + \
               [f'BILL_AMT_{i}' for i in range(1, 7)] + \
               [f'PAY_AMT_{i}' for i in range(1, 7)] + ['DEFAULT']

# Variables sensibles
sensitive_features = ['GENDER', 'MARRIAGE', 'AGE', 'EDUCATION']

# Ingeniería de características
data['PAY_AVG'] = data[[f'PAY_{i}' for i in range(1, 7)]].mean(axis=1)

data['AGE_GROUP'] = pd.cut(data['AGE'], bins=[20, 30, 40, 50, 60, 100], labels=False)

# Promedio de los pagos
data['PAY_AMT_AVG'] = data[[f'PAY_AMT_{i}' for i in range(1, 7)]].mean(axis=1)

data['BILL_AMT_AVG'] = data[[f'BILL_AMT_{i}' for i in range(1, 7)]].mean(axis=1)

cols_to_drop = ['ID', 'DEFAULT', 'AGE']  + [f'PAY_AMT_{i}' for i in range(1, 7)] + \
               [f'BILL_AMT_{i}' for i in range(1, 7)]
               

# Definir X y y
X = data.drop(cols_to_drop, axis=1)
y = data['DEFAULT']

print(X.head())
# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# categorical_features = ['GENDER', 'MARRIAGE', 'AGE_GROUP', 'EDUCATION'] + [f'PAY_{i}' for i in range(1, 7)]

# Normalización y balanceo
# best so far
# smote = SVMSMOTE(random_state=42)
smote = SVMSMOTE(k_neighbors=3, m_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Modelos y parámetros para GridSearchCV
# models = { 
    # 'Logistic Regression': {
    #     'model': LogisticRegression(random_state=42, max_iter=1000),
    #     'params': {
    #         'model__C': [0.01, 0.1, 1, 10, 100],
    #         'model__solver': ['liblinear', 'lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    #     }
    # },
    # 'Decision Tree': {
    #     'model': DecisionTreeClassifier(random_state=42),
    #     'params': {
    #         'model__criterion': ['gini', 'entropy', 'log_loss'],
    #         'model__max_depth': [None, 10, 20, 30, 40],
    #         'model__min_samples_split': [2, 5, 10],
    #         'model__min_samples_leaf': [1, 2, 4]
    #     }
    # },
    # 'SVM': {
    #     'model': SVC(random_state=42, probability=True),
    #     'params': {
    #         'model__C': [0.1, 1, 10, 100],
    #         'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #         'model__gamma': ['scale', 'auto']
    #     }
    # },
    # 'Random Forest': {
    #     'model': RandomForestClassifier(random_state=42),
    #     'params': {
    #         'model__criterion': ['gini', 'entropy', 'log_loss'],
    #         'model__n_estimators': [100, 200, 300, 400, 500],
    #         'model__max_depth': [None, 10, 20, 30],
    #         'model__min_samples_split': [2, 5, 10],
    #         'model__min_samples_leaf': [1, 2, 4]
    #     }
    # },
# }

# Evaluación de modelos
best_models = {}
results = {}

for model_name, model_info in models.items():
    print(f"Optimizing {model_name}...")
    
    # Crear pipeline
    pipeline = Pipeline([
        # ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
        ('model', model_info['model'])
    ])
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_info['params'],
        scoring='f1',
        cv=4,
        verbose=2,
        n_jobs=-1
    )
    
    # Entrenar
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Evaluación en el conjunto de prueba
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"
    
    results[model_name] = {'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': roc_auc}
    best_params = grid_search.best_params_  # Los mejores parámetros encontrados
    print(f"Mejores parámetros para {model_name}:")
    with open('results.txt', '+at') as r:
        r.write(f"Mejores parámetros para {model_name}:\n")
        r.write(f"{best_params}\n")
        r.write(f"Results:\n{classification_report(y_test, y_pred)}\n\n\n")
    print(best_params)
    print("-" * 50)
    print(f"\n{model_name} Results:")
    print(classification_report(y_test, y_pred))

# Convertir resultados a dataframe para analizar
results_df = pd.DataFrame(results).T
print("\nResumen de resultados:")
print(results_df)

optimal_params = {
    'criterion': 'gini',  # Found to be the best
    'n_estimators': 300,  # Best from GridSearchCV
    'max_depth': None,     # Best from GridSearchCV
    'min_samples_split': 2,  # Best from GridSearchCV
    'min_samples_leaf': 1  # Best from GridSearchCV
}

# Initialize and train the RandomForest model
rf_model = RandomForestClassifier(random_state=42, **optimal_params)
rf_model.fit(X_train_resampled, y_train_resampled)
