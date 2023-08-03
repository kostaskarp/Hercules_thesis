from src.bin_data import *

target_dir_binned = "src/output_data/binned"
target_dir_cumulative = "src/output_data/cumulative"
# target_dir_aggregated = "src/output_data/aggregated"

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

    # reading quickly the input folder to get the IDs of distinct games
    game_ids = get_available_games()

    # The function below, applies an arbitrary binning (here dt = 5 minute) to the input data
    # for both locations and event files and also the metrics that were derived
    # based on them. It saves the outputs in the proper output_data folder path

    status = bin_input_data(game_ids, target_dir_binned, target_dir_cumulative, dt=5, rerun=True)
    print(status)

    # The function below, calculates sum of metrics and expected goals for each player and
    # saves it in the proper output_data folder path

    #calculate_cumulative_goals(game_ids, target_dir_aggregated, target_dir_cumulative)

