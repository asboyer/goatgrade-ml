import pandas as pd
import glob
import json

# Define a function to load a JSON file and convert it to a DataFrame
def load_json_to_df(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        players_data = []
        for player_name, stats in data.items():
            stats['Player'] = player_name  # Ensure the player's name is included in their stats
            players_data.append(stats)
        df = pd.DataFrame(players_data)  # Convert to DataFrame
        return df

# Set the path to your JSON files and the range of years you're interested in
path_to_json_files = '/Users/joshuathomas/Projects/goatgrade-ml/raw_data/'
years = list(range(1990, 2024))  # Adjust the range as needed
file_paths = [path_to_json_files + f'raw_stats{year}.json' for year in years]

# Load all the JSON data into a single DataFrame
all_seasons_data = pd.DataFrame()
for filepath in file_paths:
    df = load_json_to_df(filepath)
    season_year = filepath.split('/')[-1].split('_')[1].split('.')[0]
    df['Season'] = season_year  # Add the season year as a new column
    all_seasons_data = pd.concat([all_seasons_data, df], ignore_index=True)

# Load the MVP data from the 'mvps.txt' file
mvps_df = pd.read_csv('/Users/joshuathomas/Projects/goatgrade-ml/mvps.txt', sep='\s+', header=None, on_bad_lines='skip')
mvps_df['Player'] = mvps_df[1] + ' ' + mvps_df[2]  # Concatenating the first and last name for MVP
mvps_df['Year'] = mvps_df[0].astype(str)
mvps_df = mvps_df[['Year', 'Player']]  # Simplifying the DataFrame to Year and Player

# Add an 'Is_MVP' column to the all_seasons_data DataFrame
all_seasons_data['Is_MVP'] = 0
for index, mvp in mvps_df.iterrows():
    all_seasons_data.loc[(all_seasons_data['Player'] == mvp['Player']) & (all_seasons_data['Season'].str.contains(mvp['Year'])), 'Is_MVP'] = 1

# Save the updated DataFrame to a new JSON file
all_seasons_data.to_json('/Users/joshuathomas/Projects/goatgrade-ml/updated_player_stats.json')

# Print the head of the DataFrame to verify the changes
print(all_seasons_data.head())

# Verify MVP players
mvp_players = all_seasons_data[all_seasons_data['Is_MVP'] == 1]
print(mvp_players)


# # Load your updated player stats JSON file
# df = pd.read_json('/Users/joshuathomas/Projects/goatgrade-ml/updated_player_stats.json')
#
# # Check if the 2023 season is present in the 'Season' column
# is_1990_present = 'stats1990' in all_seasons_data['Season'].values
#
# print(f"Is the 1990 season present in the dataset? {is_1990_present}")
