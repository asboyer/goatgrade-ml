import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from joblib import dump




# Load the dataset
df = pd.read_json('/Users/joshuathomas/Projects/goatgrade-ml/updated_player_stats.json')

# Preprocessing
# Convert percentage strings to float and handle missing or malformed entries
for col in ['FT%', '3P%']:
    df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce')

# Replace NaN values with column means for the percentage features
df['FT%'] = df['FT%'].fillna(df['FT%'].mean())
df['3P%'] = df['3P%'].fillna(df['3P%'].mean())


# Feature Engineering
df['Efficiency'] = (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK']) / (df['FGA'] + df['FTA'])
df.fillna(0, inplace=True)

# Define features and target
features = ['PTS', 'AST', 'TRB', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'MP', 'PER', 'TS%', 'WS', 'BPM', 'Efficiency', 'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM']
X = df[features]
y = df['Is_MVP']

# Pipeline setup
pipeline = IMBPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Hyperparameters grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10],
    'classifier__gamma': [0, 0.1, 0.5],
    'classifier__min_child_weight': [1, 5, 10]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)

# Cross-validated predictions for the confusion matrix
y_pred = cross_val_predict(grid_search.best_estimator_, X, y, cv=cv)

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
print(classification_report(y, y_pred))

