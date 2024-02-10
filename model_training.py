import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from xgboost import XGBClassifier

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

# Define features
features = ['PTS', 'AST', 'TRB', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'MP', 'PER', 'TS%', 'WS', 'BPM', 'Efficiency', 'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM']

# Split data into features (X) and target (y)
X = df[features]
y = df['Is_MVP']

# Define a pipeline with preprocessing and the classifier
pipeline = IMBPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Define the hyperparameter grid for XGBClassifier
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10],
    'classifier__gamma': [0, 0.1, 0.5],
    'classifier__min_child_weight': [1, 5, 10]
}

# Stratified K-Fold for cross-validation
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with Stratified K-Fold
grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='roc_auc', cv=stratified_kfold, n_jobs=-1)
grid_search.fit(X, y)

# Best model and its evaluation
best_model = grid_search.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Precision-Recall Curve and F1 Score
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.nanargmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Precision-Recall AUC: {auc(recall, precision)}")
print(f"F1 Score at Optimal Threshold ({optimal_threshold}): {f1_scores[optimal_idx]}")
print("Best Parameters:", grid_search.best_params_)

# Adjusted predictions based on the optimal threshold
y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_custom))
