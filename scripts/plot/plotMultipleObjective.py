import argparse
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def load_csv_files_from_directory(directory_path):
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = np.loadtxt(file_path, delimiter=",")
            all_data.append(data)
    return np.concatenate(all_data, axis=0)


def plot_data(single_file_path, directory_path, problem_name):
    # Load data from the single CSV file
    single_df = pd.read_csv(single_file_path)

    # Load data from the directory
    directory_dfs = load_csv_files_from_directory(directory_path)

    # Load Pareto front data
    pareto_front_path = f"data/ground_truth/pareto_fronts/{problem_name}_300.csv"
    pareto_front_df = pd.read_csv(pareto_front_path)

    # Plot the data
    plt.figure()

    # Plot single file data
    plt.scatter(
        single_df.iloc[:, 0],
        single_df.iloc[:, 1],
        label="Sequential MOEA/D",
        color="blue",
        s=1,
        marker="o",
        zorder=2,  # Change the marker shape here
    )

    # Plot directory data
    plt.scatter(
        directory_dfs[:, 0],
        directory_dfs[:, 1],
        color="red",
        s=1,
        marker="x",
        label="MP-MOEA/D",
        zorder=2,  # Change the marker shape here
    )

    # Plot Pareto front data with a different line width
    plt.plot(
        pareto_front_df.iloc[:, 0],
        pareto_front_df.iloc[:, 1],
        color="lightgray",
        linestyle="-",
        label="Pareto Front",
        zorder=1,  # Change the line width here
    )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    plt.title(f"{problem_name}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multiple objective data.")
    parser.add_argument(
        "-s", "--sequential", required=True, help="Path to the sequential CSV file."
    )
    parser.add_argument(
        "-p",
        "--parallel",
        required=True,
        help="Path to the directory containing CSV files.",
    )
    parser.add_argument(
        "-n", "--problem_name", required=True, help="Name of the problem."
    )

    args = parser.parse_args()

    plot_data(args.sequential, args.parallel, args.problem_name)
