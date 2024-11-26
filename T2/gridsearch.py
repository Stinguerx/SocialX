import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Cargar los datos
data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

cols_to_drop = ['Diabetes_binary']
X = data.drop(cols_to_drop, axis=1)
y = data['Diabetes_binary']

# Convert 'Income' to binary classification (Low = 1-4, High = 5-8)
data['Income_binary'] = data['Income'].apply(lambda x: 1 if x >= 5 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelos y parámetros para GridSearchCV
print("ADVERTENCIA - GridSearchCV ocupa toda la CPU y se demora mucho en probar todos los parametros")
print("ADVERTENCIA - SI SVM se deja activo puede demorarse DIAS en terminar el gridsearch")
input("Desea continuar? ")

models = { 
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__solver': ['liblinear', 'lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'model__criterion': ['gini', 'entropy', 'log_loss'],
            'model__max_depth': [None, 10, 20, 30, 40],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'model__criterion': ['gini', 'entropy', 'log_loss'],
            'model__n_estimators': [100, 200, 300, 400, 500],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    },
}


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
    grid_search.fit(X_train, y_train)
    
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