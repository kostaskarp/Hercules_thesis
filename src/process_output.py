import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def distinguish_clusters(parx, pary, deps, dmin_samples):
    X = np.zeros((parx.size, 2))
    X[:, 0], X[:, 1] = parx, pary
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    kmeans = DBSCAN(eps=deps, min_samples=dmin_samples, n_jobs=-1).fit(X)
    # kmeans = KMeans(n_clusters=n_clusters,init='random', max_iter=100, tol=0.1,algorithm='auto').fit(X)
    cluster_labels = kmeans.labels_
    return cluster_labels


def make_plot(dataframe, hyperparams):
    # Extract column names
    columns = dataframe.columns

    # Determine the number of plots (excluding the last column)
    num_plots = dataframe.shape[1] - 1
    cols = 3
    mod = num_plots % cols
    if mod > 0:
        rows = int(num_plots / cols) + 1
    else:
        rows = int(num_plots / cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))

    # Flatten the 2D array of axes for easy indexing
    axes = axes.flatten()

    # Plot each column against the last column (y-axis)
    for i in range(num_plots):
        ax = axes[i]
        x_column = columns[i]
        y_column = columns[-1]  # Always use the last column for the y-axis
        # distinguishing clusters
        cluster_labels = distinguish_clusters(
            dataframe[x_column].values,
            dataframe[y_column].values,
            hyperparams[i][0],
            hyperparams[i][1],
        )
        intermediate_df = dataframe[[x_column, y_column]]
        intermediate_df["cluster_labels"] = cluster_labels
        nn = 0
        for cluster_label, cluster_df in intermediate_df.groupby("cluster_labels"):
            if cluster_label == -1:
                ax.plot(
                    cluster_df[x_column],
                    cluster_df[y_column],
                    "k+",
                    alpha=0.2,
                    label=str(cluster_label),
                )
            else:
                res = pearsonr(cluster_df[x_column], cluster_df[y_column])
                ax.plot(
                    cluster_df[x_column],
                    cluster_df[y_column],
                    "o",
                    label=f"r={np.around(res[0], 2)}, p = {np.around(res[1],3)}",
                )
            nn = nn + 1

        # ax.scatter(np.cumsum(dataframe[x_column].values),np.cumsum(dataframe[y_column].values))
        # ax.scatter(dataframe[x_column], dataframe[y_column])
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{x_column} vs {y_column}")
        ax.legend()

    # Remove any empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot grid
    plt.show()


# Example usage:
# Assuming 'df' is your DataFrame
# plot_grid_with_last_column(df)


if __name__ == "__main__":
    base_path = "./output_data/cumulative"
    game_files = os.listdir(base_path)
    dataframe = pd.DataFrame()
    for file in tqdm(game_files):
        if file != ".gitkeep":
            team_files = os.listdir(f"{base_path}/{file}")
            for team_file in team_files:
                df = pd.read_csv(f"{base_path}/{file}/{team_file}").dropna(
                    subset=["Expected_goal"]
                )
            dataframe = pd.concat((dataframe, df))
    plot_cols = [
        "total_distance",
        "HIR_total_distance",
        "HA_actions",
        "HD_actions",
        "VHA_actions",
        "VHD_actions",
        "Expected_goal",
    ]
    hyper_parameters = [
        [0.6, 3],
        [0.35, 6],
        [0.55, 3],
        [0.5, 6],
        [0.36, 5],
        [0.38, 7],
    ]
    make_plot(dataframe[plot_cols], hyper_parameters)
    a = 1
