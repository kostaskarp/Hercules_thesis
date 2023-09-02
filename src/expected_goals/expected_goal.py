import os

import pandas as pd


def ExpectedGoals(Xloc, Yloc, Max_X=52.5):
    """Takes a df with event data and executes a row-wise computation of
    the xG of every pass
    """

    X = abs(Xloc)
    Y = abs(Yloc)

    SixYardLine = Max_X - 5.5
    ScoreBox = Max_X - 16.0
    NearBox = Max_X - 20.0
    LongShot = Max_X - 25.0

    if X > SixYardLine:
        if Y < 2.0:
            xG = 0.37
        elif Y > 2.0 and Y < 4.15:
            xG = 0.22
        elif Y > 4.15 and Y < 10:
            xG = 0.09
        elif Y > 10 and Y < 20.15:
            xG = 0.03
        else:
            xG = 0.01
    elif X < SixYardLine and X > ScoreBox:
        if Y < 4.15:
            xG = 0.20
        elif Y > 4.15 and Y < 10:
            xG = 0.08
        elif Y > 10 and Y < 20.15:
            xG = 0.03
        else:
            xG = 0.01
    elif X < ScoreBox and X > NearBox:
        if Y < 10:
            xG = 0.07
        elif Y > 10 and Y < 20.15:
            xG = 0.03
        else:
            xG = 0.01
    elif X > NearBox and X < LongShot:
        if Y < 20.15:
            xG = 0.02
        else:
            xG = 0.01
    else:
        xG = 0

    return xG


def eventWork():
    SOURCE_DIRECTORY = "/home/hercules/Documents/Ptyxiaki/Sources/events/"
    TARGET_DIRECTORY = "/home/hercules/Documents/Ptyxiaki/Proccesed/xG_per_action/"
    eventFiles = os.listdir(SOURCE_DIRECTORY)
    for fileName in eventFiles:
        goals = []
        df = pd.read_csv(SOURCE_DIRECTORY + fileName)
        for i, row in df.iterrows():
            # print(row['Start_Ts (ms)'],row['End_Ts (ms)'],row['X'],row['Y'],row['Player'],row['Event'])
            goals.append(ExpectedGoals(row["X"], row["Y"]))
        df["expGoal"] = goals  # adds an "expGoal" column
        df = df[df["Event"] == "Pass"]  # keeps only the pass actions
        df.to_csv(
            TARGET_DIRECTORY + fileName,
        )


""" Creates files with the passing actions only and their xG rating"""
