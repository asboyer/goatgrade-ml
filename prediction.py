import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import json

# Function to preprocess the new dataset similar to the training data
def preprocess_data(raw_data):
    # Initialize a list to hold processed player data
    processed_data = []

    # Iterate over each player's stats in the raw data
    for player_stats in raw_data.values():
        # Handle team changes by selecting the first team listed if 'Tm' is an array
        if isinstance(player_stats.get('Tm'), list):
            player_stats['Tm'] = player_stats['Tm'][0]

        for pct_field in ['FG%', '3P%', 'FT%']:
            pct_value = player_stats.get(pct_field, '0').strip('%')
            if pct_value == '':  # Check if the percentage string is empty
                player_stats[pct_field] = 0.0
            else:
                try:
                    player_stats[pct_field] = float(pct_value) / 100.0
                except ValueError:
                    # Log or handle cases where conversion is not possible due to unexpected format
                    player_stats[pct_field] = 0.0
        # Calculate the 'Efficiency' metric
        pts, trb, ast, stl, blk = float(player_stats.get('PTS', 0)), float(player_stats.get('TRB', 0)), float(player_stats.get('AST', 0)), float(player_stats.get('STL', 0)), float(player_stats.get('BLK', 0))
        fga, fta = float(player_stats.get('FGA', 0)), float(player_stats.get('FTA', 0))
        efficiency = (pts + trb + ast + stl + blk) / (fga + fta) if (fga + fta) > 0 else 0
        player_stats['Efficiency'] = efficiency

        processed_data.append(player_stats)

    # Convert the processed data into a DataFrame
    df_processed = pd.DataFrame(processed_data)

    # Fill missing values with 0 for the entire DataFrame
    df_processed.fillna(0, inplace=True)

    # Ensure all columns needed for prediction are present, adding them with default values if missing
    required_columns = ['PTS', 'AST', 'TRB', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'MP', 'PER', 'TS%', 'WS', 'BPM', 'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM', 'Efficiency']
    for col in required_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed



# Load the saved model
model = load('/Users/joshuathomas/Projects/goatgrade-ml/best_model.joblib')

# Load the new dataset for 2024
with open('/Users/joshuathomas/Projects/goatgrade-ml/raw_data/raw_stats2024.json') as file:
    data_2024 = json.load(file)

df_2024_preprocessed = preprocess_data(data_2024)

# Prepare features for prediction
features = ['PTS', 'AST', 'TRB', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'MP', 'PER', 'TS%', 'WS', 'BPM', 'Efficiency', 'VORP', 'WS/48', 'USG%', 'FTr', 'OBPM']
X_2024 = df_2024_preprocessed[features]

# Predict using the loaded pipeline
predictions = model.predict_proba(X_2024)[:, 1]

# Print the range of prediction probabilities
print("Prediction probabilities range:", np.min(predictions), "to", np.max(predictions))

new_threshold = 0.3  # Adjust based on your observations
predicted_mvp = (predictions >= new_threshold).astype(int)
df_2024_preprocessed['Predicted_MVP'] = predicted_mvp
predicted_mvp_candidates = df_2024_preprocessed[df_2024_preprocessed['Predicted_MVP'] == 1]

# Display or save the predicted MVP candidates
print(predicted_mvp_candidates[['Player', 'Predicted_MVP']])
