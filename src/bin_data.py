import os.path

import pandas as pd

from src.features.calculate_features import *
from tqdm import tqdm

from src.cumulative_calculation import *
from src.expected_goals.expected_goal import ExpectedGoals

target_dir_locations = "./src/input_data/locations/"
target_dir_events = "./src/input_data/events/"


def bin_input_data(game_ids, target_dir_base, dt=5):
    for i, game_id in enumerate(game_ids):
        events_df = pd.read_csv(target_dir_events + game_id + "_Event_cleaned.csv")
        # trucating the player integer ID to the fifth number because
        # player IDs may not match in the events and locations files
        events_df["Player"] = events_df["Player"].apply(lambda x: str(int(x))[0:5])
        locations_df = pd.read_csv(target_dir_locations + game_id + "_cleaned.csv")
        # applying the same truncation like above.
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
        locations_df = (
            locations_df.set_index("Team").drop(index=team_ball_id).reset_index()
        )

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

        # separating first from second half for each player in the locations dataframe
        tmp_loc_df = pd.DataFrame()
        recorded_end_first_period_stamps = []
        recorded_start_second_period_stamps = []
        for player_id, group in locations_df.groupby("Player_Name"):
            clean_player_df = group.dropna().reset_index(drop=True)
            iis = np.arange(0, clean_player_df.shape[0] - 1)
            time_seconds = clean_player_df["Timestamp"]
            dt_seconds = time_seconds.values[iis + 1] - time_seconds.values[iis]
            try:
                separator_position = np.where(dt_seconds > 200.0)[0][0]
                clean_player_df["Half_time"] = pd.Series(
                    np.repeat("First-half", clean_player_df.shape[0])
                ).where(clean_player_df.index < separator_position, other="Second-half")
                tmp_loc_df = pd.concat((tmp_loc_df, clean_player_df))
                recorded_end_first_period_stamps.append(
                    clean_player_df["Timestamp"][separator_position]
                )
                recorded_start_second_period_stamps.append(
                    clean_player_df["Timestamp"][separator_position + 1]
                )
            except:
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
            locations_df["Timestamp"].values[-1], events_df["End_Ts (ms)"].values[-1]
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

        locations_grouped_by_bin = locations_df_copy.groupby(["Player_Name", "Bin_ind"])
        loc_output_df = pd.DataFrame()
        print(f"Calculating binned features for game {game_id} ...")
        for player_bin_pair, group_df in tqdm(locations_grouped_by_bin):
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
                #"Game-time": f"",
                "Half-time": group_df["Half_time"].values[0],
            }
            output_dict = calculate_features(group_df, player_dict)
            loc_output_df = pd.concat((loc_output_df, pd.DataFrame([output_dict])))

        events_df_copy = events_df.copy(deep=True)
        events_df_copy["Bin_ind"] = inds_events
        events_grouped_by_bin = events_df_copy.groupby(["Player", "Bin_ind"])
        xpg_output_df = pd.DataFrame()
        print(f"Calculating binned expected goals for game {game_id} ...")
        for player_bin_pair, group_df in tqdm(events_grouped_by_bin):
            player_id, bin_edge = player_bin_pair
            t_bin_start = np.around(dt_bins[bin_edge - 1] / (60.0 * 1000.0), 3)
            if bin_edge <= dt_bins.size - 1:
                t_bin_end = np.around(dt_bins[bin_edge] / (60.0 * 1000.0), 3)
            else:
                t_bin_end = np.around(end_time / (60.0 * 1000.0), 3)
            cumulative_xpg_goal_in_bin = group_df["expGoal"].sum()
            # goals = []
            # for i, row in group_df.iterrows():
            #     goals.append(ExpectedGoals(row["X"], row["Y"]))
            # cum_expected_goals = np.sum(np.array(goals))
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

        final_grouped_by_team = final_merged_df.groupby("Team")
        for team_id, final_grouped_by_team in final_grouped_by_team:
            final_grouped_by_team["Goalkeeper"] = final_grouped_by_team[
                "Player"
            ].map(goal_keeper_dict)
            final_grouped_by_team.to_csv(
                f"{target_dir_base}/{game_id}/{game_id}_{team_id}_{dt}minute.csv",
                float_format="%.4f",
                index=False,
            )
