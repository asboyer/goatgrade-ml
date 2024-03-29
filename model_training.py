import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load the dataset
df = pd.read_json('/Users/joshuathomas/Projects/goatgrade-ml/updated_player_stats.json')

# Correct approach to handle non-numeric values and fill NaN for numeric columns only
for col in df.columns:
    # Convert to numeric where possible
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Now safely fill NaNs for numeric columns
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)

# Fill remaining NaNs with 0 (for any non-numeric columns that couldn't be converted)
df.fillna(0, inplace=True)

# Feature Engineering
df['Efficiency'] = (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK']) / (df['FGA'] + df['FTA'])

# Define features and target
features = ['PTS', 'AST', 'TRB', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'MP', 'PER', 'TS%', 'WS', 'BPM', 'Efficiency', 'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM']
X = df[features]
y = df['Is_MVP']

# Pipeline setup with RandomForest and SMOTE-ENN
pipeline = IMBPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote_enn', SMOTEENN(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Custom scorer focusing on F1 score for the minority class
f1_scorer = make_scorer(f1_score, pos_label=1)

# Simplified hyperparameters for quick setup
param_dist = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2]
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, scoring=f1_scorer, cv=cv, n_jobs=-1,random_state=42)
random_search.fit(X, y)

# Print the best parameters found
print("Best Parameters:", random_search.best_params_)

# Save the trained model
model_path = '/Users/joshuathomas/Projects/goatgrade-ml/best_model.joblib'
dump(random_search.best_estimator_, model_path)
print(f"Model saved to {model_path}")

# Performance overview
y_pred = random_search.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print(classification_report(y, y_pred))
