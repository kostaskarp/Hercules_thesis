import os
import sys

import matplotlib.pyplot as plt

import pandas as pd

from src.cumulative_calculation import *
from src.bin_data import *

target_dir_features_base = "src/output_data/features"
target_dir_xpg_base = "src/output_data/expected_goals"
target_dir_binned = "src/output_data/binned"
target_dir_cumulative = "src/output_data/cumulative"
target_dir_aggregated = "src/output_data/aggegated"

def get_available_games():
    locations_data = os.listdir(target_dir_locations)
    locations_data.remove(".gitkeep")
    events_data = os.listdir(target_dir_events)
    events_data.remove(".gitkeep")
    games_from_events = pd.Series(events_data).apply(lambda x: x.split("_")[0])
    games_from_locations = pd.Series(locations_data).apply(lambda x: x.split("_")[0])
    if games_from_events.size != games_from_events.unique().size:
        raise SystemError("There are duplicate data in your events folder")
    if games_from_locations.size != games_from_locations.unique().size:
        raise SystemError("There are duplicate data in your locations folder")
    if games_from_locations.size != games_from_events.size:
        raise SystemError(
            "The number of events files does not match that of locations files"
        )
    else:
        return games_from_events


if __name__ == "__main__":

    # df = pd.read_csv(target_dir_binned+'/0118001ERE_5minute_old.csv')
    # grouped_df = df.groupby("Player")
    # for player, group_df in grouped_df:
    #     t = group_df["Time"]
    #     cum_xpg = np.cumsum(group_df["Expected_goal"].values)
    #     cum_vha_actions = np.cumsum(group_df["VHA_actions"].values)
    #     cum_hir_distance = np.cumsum(group_df["HIR_total_distance"].values)
    #     plt.plot(cum_hir_distance,cum_xpg)
    #     plt.show()
    # sys.exit()

    game_ids = get_available_games()

    bin_input_data(game_ids, target_dir_binned, dt=5)

    #calculate_locations_metrics(game_ids, target_dir_features_base)

    calculate_cumulative_goals(game_ids, target_dir_aggregated, target_dir_cumulative)

    # for game_id in game_ids:
    #     df = pd.read_csv(target_dir_locations + game_id + '_cleaned.csv')
    #     grpuped_df = df.groupby("Player_Name")
    #     for player , player_df in grpuped_df:
    #         clean_player_df = player_df.dropna()
    #         iis = np.arange(1,clean_player_df.shape[0])
    #         time_seconds = clean_player_df["Timestamp"]*1e-3
    #         dt_seconds = time_seconds.values[iis] - time_seconds.values[iis-1]
    #         pos = np.where(dt_seconds > 100.)[0]
    #         a = 1
    #         #half_time = pos[0]
    #     a = 1



    # for game_id in game_ids:
    #     xpg_file_name = f"{target_dir_xpg_base}/{game_id}_expected_goals.csv"
    #     features_file_name = f"{target_dir_features_base}/{game_id}_features.csv"
    #     if os.path.isfile(xpg_file_name) and os.path.isfile(features_file_name):
    #         xpg_df = pd.read_csv(xpg_file_name)
    #         xpg_df["Player"] = xpg_df["Player"].apply(lambda x: str(int(x))[0:5])
    #         feature_df = pd.read_csv(features_file_name)
    #         feature_df['Player'] = feature_df['Player'].apply(lambda x: str(int(x))[0:5])
    #         cum_exp_goal = xpg_df.groupby("Player")["expGoal"].sum()
    #         feature_df["expGoal"] = feature_df["Player"].map(cum_exp_goal)
    #     else:
    #         print(f"Skipping game {game_id} because data is missing for either events or locations")
    #     a = 1


