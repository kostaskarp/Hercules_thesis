import pandas as pd

import numpy as np

from tqdm import tqdm

from src.expected_goals.expected_goal import ExpectedGoals

target_dir_locations = "./src/input_data/locations/"
target_dir_events = "./src/input_data/events/"


def calculate_expected_goals(game_ids, target_dir_base):
    for i, game_id in tqdm(enumerate(game_ids)):
        events_df = pd.read_csv(target_dir_events + game_id + '_Event_cleaned.csv')
        events_df["Player"] = events_df["Player"].apply(lambda x: str(int(x))[0:5])
        locations_df = pd.read_csv(target_dir_locations + game_id + "_cleaned.csv")
        locations_df["Player_Name"] = locations_df["Player_Name"].apply(lambda x: str(int(x))[0:5])

        # # getting all unique player ids from the events and locations files and checking if they match
        # players_in_events = events_df["Player"].unique()
        # players_in_locations = locations_df["Player_Name"].unique()
        #
        # # we find which players in the locations data file are also found in the events file
        # players_locations = players_in_locations[np.isin(players_in_locations, players_in_events)]
        #
        # matching_players_in_events = np.isin(events_df["Player"].values, players_locations)
        #
        # clean_events_df = events_df.loc[matching_players_in_events, :]
        #
        # # first we group the dataframe by the combination of player name and team name
        # #grouped_df = clean_events_df.groupby(["Player_Name", "Team"])

        # detecting the id of the ball
        unique_players_per_team = locations_df.groupby(["Team"])["Player_Name"].nunique()
        ball_id = unique_players_per_team.where(unique_players_per_team == 1).dropna().index[0]

        # detecting the 2 goalkeepers
        df = locations_df.copy(deep=True)
        mean_x_df = df.groupby(["Player_Name", "Team"])["X"].mean().abs()
        goal_keepers = mean_x_df.groupby("Team").max().drop(index=ball_id).index.values
        #mean_x_df.reset_index().set_index("Team").drop(index=ball_id)

        indexed_locations_df = locations_df.set_index("Player_Name")
        goals = []
        print("Calculating expected goals...")
        for i, row in tqdm(events_df.iterrows()):
            #player_id = row["Player"]
            #players_team_id = indexed_locations_df.loc[player_id]["Team"][0]

            # print(row['Start_Ts (ms)'],row['End_Ts (ms)'],row['X'],row['Y'],row['Player'],row['Event'])
            goals.append(ExpectedGoals(row['X'], row['Y']))
        events_df['expGoal'] = goals  # adds an "expGoal" column
        clean_events_df = events_df[events_df['Event'] == 'Pass']  # keeps only the pass actions
        clean_events_df.to_csv(f"{target_dir_base}/{game_id}_expected_goals.csv", index=False)