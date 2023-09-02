from tqdm import tqdm

from src.features.calculate_features import *

target_dir_locations = "./src/input_data/locations/"
target_dir_events = "./src/input_data/events/"


def calculate_locations_metrics(game_ids, target_dir_base):
    for i, game_id in enumerate(game_ids):
        events_df = pd.read_csv(target_dir_events + game_id + "_Event_cleaned.csv")
        events_df["Player"] = events_df["Player"].apply(lambda x: str(int(x))[0:5])
        locations_df = pd.read_csv(target_dir_locations + game_id + "_cleaned.csv")
        locations_df["Player_Name"] = locations_df["Player_Name"].apply(
            lambda x: str(int(x))[0:5]
        )

        # first we group the dataframe by the combination of player name and team name
        grouped_df = locations_df.groupby(["Player_Name", "Team"])

        # then we iterate over the groups (i.e. a combination of player and team)
        output_df = pd.DataFrame()
        print(f"Calculating features for game: {game_id} ...")
        for group_keys, group_df in tqdm(grouped_df):
            player_id, team_id = group_keys
            player_dict = {"Player": player_id, "Team": team_id}
            output_dict = calculate_features(group_df, player_dict)
            output_df = pd.concat((output_df, pd.DataFrame([output_dict])))
        output_df.to_csv(f"{target_dir_base}/{game_id}_features.csv", index=False)
