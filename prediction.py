import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # Ensure 'Player' column is treated separately and not modified
    player_data = df[['Player']].copy() if 'Player' in df.columns else pd.DataFrame(index=df.index)

    # Convert all expected numeric columns to float, handling NaN values, excluding 'Player'
    numeric_cols = [col for col in
                    ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FGA', 'FTA', 'FG%', 'FT%', '3P%', 'MP', 'PER', 'TS%', 'WS',
                     'BPM', 'Efficiency', 'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM'] if
                    col in df.columns and col != 'Player']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Reintroduce 'Player' column to df after numeric conversions
    df = pd.concat([player_data, df], axis=1)

    # Calculate 'Efficiency' if not already present
    if 'Efficiency' not in df.columns:
        df['Efficiency'] = (df['PTS'] + df['TRB'] + df['AST'] + df['STL'] + df['BLK']) / (
                    df['FGA'] + df['FTA']).replace({0: np.nan})
        df['Efficiency'].fillna(0, inplace=True)

    return df


# Load the saved model
model = load('/Users/joshuathomas/Projects/goatgrade-ml/best_model.joblib')

# Load the new dataset for 2024
df_2024 = pd.read_json('/Users/joshuathomas/Projects/goatgrade-ml/raw_data/raw_stats2024.json')
df_2024_preprocessed = preprocess_data(df_2024)

# Prepare features for prediction, ensuring no NaN values remain
features = ['PTS', 'AST', 'TRB', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'MP', 'PER', 'TS%', 'WS', 'BPM', 'Efficiency',
            'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM']
X_2024 = df_2024_preprocessed[features].fillna(0)

# Scale features
scaler = StandardScaler()
X_2024_scaled = scaler.fit_transform(X_2024)

# Predict MVP candidates
predictions_2024 = model.predict(X_2024_scaled)

# Add predictions to DataFrame and filter for MVP candidates
df_2024_preprocessed['Predicted_MVP'] = predictions_2024
predicted_mvp_candidates = df_2024_preprocessed[df_2024_preprocessed['Predicted_MVP'] == 1]

# Output predicted MVP candidates
print(predicted_mvp_candidates[['Player', 'Predicted_MVP']])
