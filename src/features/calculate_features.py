import numpy as np
import pandas as pd

# Generalised functions


def calc_total_distance(input_df: pd.DataFrame) -> float:
    x = input_df["X"].dropna().values
    y = input_df["Y"].dropna().values
    indices = np.arange(1, x.size)
    delta_x = x[indices] - x[indices - 1]
    delta_y = y[indices] - y[indices - 1]
    tot_distance = np.sum(np.sqrt(delta_x * delta_x + delta_y * delta_y))
    return tot_distance


# Features


def calcVHA_acctions(input_dict, playerDf):
    playerAcc = playerDf["Acceleration"].values
    vha_counter = 0.0
    for AccI in playerAcc:
        VHA_status = False
        if AccI > 3.5 and VHA_status == False:
            VHA_status = True
            vha_counter += 1
        else:
            if AccI < 3.5 and VHA_status == True:
                VHA_status = False
    input_dict.update({"VHA_actions": vha_counter})
    return input_dict


def calcVHD_acctions(input_dict, playerDf):
    playerAcc = playerDf["Acceleration"].values
    vhd_counter = 0.0
    for AccI in playerAcc:
        VHD_status = False
        if AccI < -3.5 and VHD_status == False:
            VHD_status = True
            vhd_counter += 1
        else:
            if AccI > -3.5 and VHD_status == True:
                VHD_status = False
    input_dict.update({"VHD_actions": vhd_counter})
    return input_dict


def calcHA_acctions(input_dict, playerDf):
    playerAcc = playerDf["Acceleration"].values
    ha_counter = 0.0
    for AccI in playerAcc:
        HA_status = False
        if AccI <= 3.5 and AccI > 2.5 and HA_status == False:
            HA_status = True
            ha_counter += 1
        else:
            if AccI < 2.5 and HA_status == True:
                HA_status = False
    input_dict.update({"HA_actions": ha_counter})
    return input_dict


def calcHD_acctions(input_dict, playerDf):
    playerAcc = playerDf["Acceleration"].values
    hd_counter = 0.0
    for AccI in playerAcc:
        HD_status = False
        if AccI > -3.5 and AccI < -2.5 and HD_status == False:
            HD_status = True
            hd_counter += 1
        else:
            if AccI > -2.5 and HD_status == True:
                HD_status = False
    input_dict.update({"HD_actions": hd_counter})
    return input_dict


def total_distance(input_dict: dict, input_df: pd.DataFrame) -> dict:
    tot_distance = calc_total_distance(input_df)
    input_dict.update({"total_distance": tot_distance})
    return input_dict


def total_time_played(input_dict: dict, input_df: pd.DataFrame) -> dict:
    tot_time_m = ((input_df.dropna().shape[0] - 2) * 0.1) / 60.0
    iis = np.arange(1, input_df.dropna().shape[0])
    time_series = input_df["Timestamp"].values
    total_time_played = np.sum(time_series[iis] - time_series[iis - 1])
    input_dict.update({"total_time_played [m]": (total_time_played * 1e-3) / 60.0})
    return input_dict


def high_intensity_running_distance(input_dict: dict, input_df: pd.DataFrame) -> dict:
    # playerAcc = playerDf['Acceleration'].values
    hir_mask = (input_df["Speed"] >= 5.5) & (input_df["Speed"] <= 7.0)
    hir_df = input_df.loc[hir_mask, :]
    tot_distance_hir = calc_total_distance(hir_df)
    input_dict.update({"HIR_total_distance": tot_distance_hir})
    return input_dict


feature_functions = [
    total_distance,
    high_intensity_running_distance,
    calcHA_acctions,
    calcHD_acctions,
    calcVHA_acctions,
    calcVHD_acctions,
    total_time_played,
]


def calculate_features(input_df, player_dict):
    for feature_function in feature_functions:
        player_dict = feature_function(player_dict, input_df)
    return player_dict
