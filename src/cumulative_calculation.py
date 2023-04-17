import os

import pandas as pd

from tqdm import tqdm

from src.expected_goals.expected_goal import ExpectedGoals
from src.features.calculate_features import *

target_dir_locations = "./src/input_data/locations/"
target_dir_events = "./src/input_data/events/"

"""
 Function that calculates and adds expected goal as
 a new column to given input data frame with events data 
"""


def add_expected_goal(events_df: pd.DataFrame) -> pd.DataFrame:
    goals = []
    for i, row in tqdm(events_df.iterrows()):
        goals.append(ExpectedGoals(row["X"], row["Y"]))
    events_df["expGoal"] = goals  # adds an "expGoal" column
    clean_events_df = events_df[
        events_df["Event"] == "Pass"
    ]  # keeps only the pass actions
    return clean_events_df


def calculate_cumulative_goals(game_ids, target_dir_agg, target_dir_cum):
    for i, game_id in enumerate(game_ids):
        events_df = pd.read_csv(target_dir_events + game_id + "_Event_cleaned.csv")
        events_df["Player"] = events_df["Player"].apply(lambda x: str(int(x))[0:5])
        locations_df = pd.read_csv(target_dir_locations + game_id + "_cleaned.csv")
        locations_df["Player_Name"] = locations_df["Player_Name"].apply(
            lambda x: str(int(x))[0:5]
        )

        # detecting and removing the id of the ball
        unique_players_per_team = locations_df.groupby(["Team"])[
            "Player_Name"
        ].nunique()
        try:
            team_ball_id = (
                unique_players_per_team.where(unique_players_per_team == 1)
                .dropna()
                .index[0]
            )
        except:
            raise SyntaxWarning("Ball ID was not detected..")
        locations_df = locations_df.set_index("Team").drop(index=team_ball_id).reset_index()

        print(f"Calculating expected goals for game {game_id} ...")
        expected_goal_df = add_expected_goal(events_df)

        # preparing to calculate aggregate df
        # aggregate_df = locations_df.copy(deep=True)
        # aggregate_df = aggregate_df.set_index(["Player_Name", "Timestamps"])
        #
        # aggregate_df["Expected_goals"] = aggregate_df.index.map(
        #     expected_goal_df.set_index(["Player", "End_Ts (ms)"])["expGoal"]
        # )
        # aggregate_df.to_csv(f"{target_dir_agg}/{game_id}_aggregate.csv", index=False)


        # first we group the dataframe by the combination of player name and team name
        grouped_df = locations_df.groupby(["Player_Name", "Team"])

        # then we iterate over the groups (i.e. a combination of player and team)
        features_df = pd.DataFrame()
        print(f"Calculating features for game: {game_id} ...")
        for group_keys, group_df in tqdm(grouped_df):
            player_id, team_id = group_keys
            player_dict = {"Player": player_id, "Team": team_id}
            output_dict = calculate_features(group_df, player_dict)
            features_df = pd.concat((features_df, pd.DataFrame([output_dict])))

        cum_exp_goal = expected_goal_df.groupby("Player")["expGoal"].sum()
        features_df["Expected_goals"] = features_df["Player"].map(cum_exp_goal)

        # # detecting the id of the ball
        # unique_players_per_team = locations_df.groupby(["Team"])["Player_Name"].nunique()
        # ball_id = unique_players_per_team.where(unique_players_per_team == 1).dropna().index[0]
        #
        # # detecting the 2 goalkeepers
        # df = locations_df.copy(deep=True)
        # mean_x_df = df.groupby(["Player_Name", "Team"])["X"].mean().abs()
        # goal_keepers = mean_x_df.groupby("Team").max().drop(index=ball_id).index.values
        # #mean_x_df.reset_index().set_index("Team").drop(index=ball_id)

        if not os.path.isdir(f"{target_dir_cum}/{game_id}"):
            os.makedirs(f"{target_dir_cum}/{game_id}")

        grouped_by_team = features_df.groupby("Team")
        for team_id, team_df in grouped_by_team:
            team_df.to_csv(
                f"{target_dir_cum}/{game_id}/{game_id}_{team_id}_cumulative.csv", float_format="%.4f", index=False
            )
