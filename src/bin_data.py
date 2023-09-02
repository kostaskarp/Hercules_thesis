import os.path

import pandas as pd
from tqdm import tqdm

from src.cumulative_calculation import *
from src.expected_goals.expected_goal import ExpectedGoals
from src.features.calculate_features import *

target_dir_locations = "./src/input_data/locations/"
target_dir_events = "./src/input_data/events/"


# Accepts the locations dataframe as input and then
# detects and removes the ID of the ball
def detect_and_remove_ball_id(locations_df):
    # detecting and removing the id of the ball
    unique_players_per_team = locations_df.groupby(["Team"])["Player_Name"].nunique()
    try:
        team_ball_id = (
            unique_players_per_team.where(unique_players_per_team == 1)
            .dropna()
            .index[0]
        )
    except:
        raise SyntaxWarning("Ball ID was not detected..")
    locations_df = locations_df.set_index("Team").drop(index=team_ball_id).reset_index()
    return locations_df


# Function that detects the 2 goalkeepers and returns a dictionary with
# their IDs and the flag/string "Goalkeeper" as value
def detect_goalkeepers(locations_df):
    # detecting the 2 goalkeepers
    df = locations_df.copy(deep=True)
    initial_time_df = (
        df.groupby(["Team", "Player_Name"])
        .agg({"Timestamp": min, "X": "mean"})
        .reset_index()
    )
    goal_keeper_dict = {}
    for team, group_df in initial_time_df.groupby("Team"):
        goal_keeper_id = group_df.loc[group_df["X"].idxmin()]["Player_Name"]
        goal_keeper_dict.update({goal_keeper_id: "Goalkeeper"})
    return goal_keeper_dict


def bin_input_data(game_ids, target_dir_base, target_dir_base_2, dt=5, rerun=False):
    # we loop over the detected game IDs from the input files
    for game_id in tqdm(game_ids):
        # First we check if the analysis for the game with id game_id
        # has already been done. if it has been done and we also don't
        # want to repeat the analysis then we skip (pass) this game_id
        # and give an informative message to the user
        if (
            os.path.isdir(f"{target_dir_base}/{game_id}")
            and os.path.isdir(f"{target_dir_base_2}/{game_id}")
            and not rerun
        ):
            status = "Nothing new to analyse. If you want to re-run set rerun=True."
        else:
            # In the following lines we read the data from the input folders and floor the
            # player integer ID at the 5th digit, because occasionally player IDs do not match in
            # events and locations files
            events_df = pd.read_csv(target_dir_events + game_id + "_Event_cleaned.csv")
            events_df["Player"] = events_df["Player"].apply(lambda x: str(int(x))[0:5])
            locations_df = pd.read_csv(target_dir_locations + game_id + "_cleaned.csv")
            # applying the same truncation like above.
            locations_df["Player_Name"] = locations_df["Player_Name"].apply(
                lambda x: str(int(x))[0:5]
            )

            locations_df = detect_and_remove_ball_id(locations_df)

            # separating first from second half for each player in the locations dataframe
            tmp_loc_df = pd.DataFrame()
            recorded_end_first_period_stamps = []
            recorded_start_second_period_stamps = []
            for player_id, group in locations_df.groupby("Player_Name"):
                # by dropping the missing rows, below, we effectively create large
                # time gaps in the data that can be used to identify the split
                # between first and second half of the game
                clean_player_df = group.dropna().reset_index(drop=True)
                # next we measure the time difference between adjacent
                # (consecutive) time measurements
                iis = np.arange(0, clean_player_df.shape[0] - 1)
                time_seconds = clean_player_df["Timestamp"]
                dt_seconds = time_seconds.values[iis + 1] - time_seconds.values[iis]
                # Finally, by applying the where function we find where the time jumps
                # more than 200ms from one measurement to the next. The position,
                # in the dt_seconds array, where for the first time this condition is met,
                # should mark the end of the first half
                try:
                    separator_position = np.where(dt_seconds > 200.0)[0][0]
                    clean_player_df["Half_time"] = pd.Series(
                        np.repeat("First-half", clean_player_df.shape[0])
                    ).where(
                        clean_player_df.index < separator_position, other="Second-half"
                    )
                    tmp_loc_df = pd.concat((tmp_loc_df, clean_player_df))
                    recorded_end_first_period_stamps.append(
                        clean_player_df["Timestamp"][separator_position]
                    )
                    recorded_start_second_period_stamps.append(
                        clean_player_df["Timestamp"][separator_position + 1]
                    )
                except:
                    a = 1
                    clean_player_df["Half_time"] = pd.Series(
                        np.repeat(pd.NA, clean_player_df.shape[0])
                    )
            locations_df = tmp_loc_df

            # Below we calculate and append the expected goals as
            # a new column to the events dataframe
            events_df = add_expected_goal(events_df)

            # Below, we do the necessary preparations to apply the binning
            dt_ms = dt * 60.0 * 1000.0
            start_time = min(
                locations_df["Timestamp"].values[0], events_df["End_Ts (ms)"].values[0]
            )
            end_of_half_time = np.min(np.array(recorded_end_first_period_stamps))
            start_of_second_half = np.max(np.array(recorded_start_second_period_stamps))
            end_time = max(
                locations_df["Timestamp"].values[-1],
                events_df["End_Ts (ms)"].values[-1],
            )

            # creating the time bins for the first half
            first_half_bins = np.arange(start_time, end_of_half_time, dt_ms)
            if end_of_half_time > first_half_bins[-1]:
                first_half_bins = np.append(first_half_bins, end_of_half_time)

            # creating the time bins for the second half
            second_half_bins = np.arange(start_of_second_half, end_time, dt_ms)
            if end_time > second_half_bins[-1]:
                second_half_bins = np.append(second_half_bins, end_time)

            # combining the 2 halfs
            dt_bins = np.concatenate((first_half_bins, second_half_bins))

            inds_locations = np.digitize(locations_df["Timestamp"].values, dt_bins)
            inds_events = np.digitize(events_df["End_Ts (ms)"].values, dt_bins)

            locations_df_copy = locations_df.copy(deep=True)
            locations_df_copy["Bin_ind"] = inds_locations

            locations_grouped_by_bin = locations_df_copy.groupby(
                ["Player_Name", "Bin_ind"]
            )
            loc_output_df = pd.DataFrame()
            # print(f"Calculating binned features for game {game_id} ...")
            for player_bin_pair, group_df in locations_grouped_by_bin:
                player_id, bin_edge = player_bin_pair
                t_bin_start = np.around(dt_bins[bin_edge - 1] / (60.0 * 1000.0), 3)
                if bin_edge <= dt_bins.size - 1:
                    t_bin_end = np.around(dt_bins[bin_edge] / (60.0 * 1000.0), 3)
                else:
                    t_bin_end = np.around(end_time / (60.0 * 1000.0), 3)
                player_dict = {
                    "Player": player_id,
                    "Team": group_df["Team"].values[0],
                    "Time": f"{t_bin_start}-{t_bin_end}",
                    "Half-time": group_df["Half_time"].values[0],
                }
                output_dict = calculate_features(group_df, player_dict)
                loc_output_df = pd.concat((loc_output_df, pd.DataFrame([output_dict])))

            events_df_copy = events_df.copy(deep=True)
            events_df_copy["Bin_ind"] = inds_events
            events_grouped_by_bin = events_df_copy.groupby(["Player", "Bin_ind"])
            xpg_output_df = pd.DataFrame()
            for player_bin_pair, group_df in events_grouped_by_bin:
                player_id, bin_edge = player_bin_pair
                t_bin_start = np.around(dt_bins[bin_edge - 1] / (60.0 * 1000.0), 3)
                if bin_edge <= dt_bins.size - 1:
                    t_bin_end = np.around(dt_bins[bin_edge] / (60.0 * 1000.0), 3)
                else:
                    t_bin_end = np.around(end_time / (60.0 * 1000.0), 3)
                cumulative_xpg_goal_in_bin = group_df["expGoal"].sum()
                bin_dict = {
                    "Player": player_id,
                    "Time": f"{t_bin_start}-{t_bin_end}",
                    "Expected_goal": cumulative_xpg_goal_in_bin,
                }
                xpg_output_df = pd.concat((xpg_output_df, pd.DataFrame([bin_dict])))

            xpg_output_df["Time"] = xpg_output_df["Time"].astype(str)
            loc_output_df["Time"] = loc_output_df["Time"].astype(str)

            final_merged_df = loc_output_df.merge(
                xpg_output_df, how="outer", on=["Player", "Time"]
            )

            if not os.path.isdir(f"{target_dir_base}/{game_id}"):
                os.makedirs(f"{target_dir_base}/{game_id}")
            if not os.path.isdir(f"{target_dir_base_2}/{game_id}"):
                os.makedirs(f"{target_dir_base_2}/{game_id}")

            final_grouped_by_team = final_merged_df.groupby("Team")
            for team_id, final_grouped_by_team in final_grouped_by_team:
                final_grouped_by_team.to_csv(
                    f"{target_dir_base}/{game_id}/{game_id}_{team_id}_{dt}minute.csv",
                    float_format="%.4f",
                    index=False,
                )

                cumulative_total_df = (
                    final_grouped_by_team.drop(["Time", "Half-time"], axis=1)
                    .groupby("Player")
                    .sum()
                    .reset_index()
                )
                cumulative_total_df["Half-time"] = pd.Series(
                    np.repeat("Total", cumulative_total_df.shape[0])
                )

                cumulative_per_halfs_df = (
                    final_grouped_by_team.drop(["Time"], axis=1)
                    .groupby(["Player", "Half-time"])
                    .sum()
                    .reset_index()
                )

                joint_cumulative_df = pd.concat(
                    (cumulative_total_df, cumulative_per_halfs_df)
                ).sort_values("Player")

                joint_cumulative_df.to_csv(
                    f"{target_dir_base_2}/{game_id}/{game_id}_{team_id}",
                    float_format="%.4f",
                    index=False,
                )
            status = f"Successfully analysed {len(game_ids)} games."
    return status
