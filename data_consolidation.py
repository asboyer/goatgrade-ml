import pandas as pd
import glob
import json

# Function to load JSON file to DataFrame
def load_json_to_df(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        players_data = []
        for player_name, stats in data.items():
            stats['Player'] = player_name
            players_data.append(stats)
        df = pd.DataFrame(players_data)
        return df

# Load JSON data into DataFrame
path_to_json_files = '/Users/joshuathomas/Projects/goatgrade-ml/raw_data/'
years = range(1970, 2024)
file_paths = [path_to_json_files + f'raw_stats{year}.json' for year in years]

all_seasons_data = pd.DataFrame()
for filepath in file_paths:
    df = load_json_to_df(filepath)
    season_year = filepath.split('/')[-1].split('_')[1].split('.')[0]
    df['Season'] = season_year
    all_seasons_data = pd.concat([all_seasons_data, df], ignore_index=True)

# Convert 'PER' to numeric, ignoring non-numeric values
all_seasons_data['PER'] = pd.to_numeric(all_seasons_data['PER'], errors='coerce')

# Define threshold for PER
threshold_per = 20

# Filter data based on PER
filtered_data = pd.DataFrame()
for season, group in all_seasons_data.groupby('Season'):
    high_per_players = group[group['PER'] >= threshold_per]
    filtered_data = pd.concat([filtered_data, high_per_players], ignore_index=True)

# Load MVP data
mvps_df = pd.read_csv('/Users/joshuathomas/Projects/goatgrade-ml/mvps.txt', sep='\s+', header=None, on_bad_lines='skip')
mvps_df['Player'] = mvps_df[1] + ' ' + mvps_df[2]
mvps_df['Year'] = mvps_df[0].astype(str)
mvps_df = mvps_df[['Year', 'Player']]

# Add 'Is_MVP' column
filtered_data['Is_MVP'] = 0
for index, mvp in mvps_df.iterrows():
    filtered_data.loc[(filtered_data['Player'] == mvp['Player']) & (filtered_data['Season'].str.contains(mvp['Year'])), 'Is_MVP'] = 1

# Save the updated DataFrame to a new JSON file
filtered_data.to_json('/Users/joshuathomas/Projects/goatgrade-ml/updated_player_stats.json')

# Print the head of the DataFrame to verify changes
print(filtered_data.head())

# Verify MVP players
mvp_players = filtered_data[filtered_data['Is_MVP'] == 1]
print(mvp_players)