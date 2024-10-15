import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.spatial import distance


def load_csv(file_path):
    return pd.read_csv(file_path, header=None).values


def calculate_igd(pareto_front, solutions):
    igd = 0
    for pf_point in pareto_front:
        min_dist = np.square(
            np.min([distance.euclidean(pf_point, sol) for sol in solutions])
        )
        igd += min_dist
    return np.sqrt(igd) / len(pareto_front)


def is_dominated(sol, solutions):
    for other in solutions:
        if all(other <= sol) and any(other < sol):
            return True
    return False


def filter_non_dominated(solutions):
    non_dominated = []
    for sol in solutions:
        if not is_dominated(sol, solutions):
            non_dominated.append(sol)
    return np.array(non_dominated)


def main(directory_path):
    results = {}
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            pareto_front_path = os.path.join(
                "data", "ground_truth", "pareto_fronts", f"{subdir}_300.csv"
            )
            if not os.path.exists(pareto_front_path):
                continue
            pareto_front = load_csv(pareto_front_path)
            igd_values = []
            for file in os.listdir(os.path.join(subdir_path, "objective")):
                file_path = os.path.join(subdir_path, "objective", file)
                if file.endswith(".csv"):
                    solutions = load_csv(file_path)
                    non_dominated_solutions = filter_non_dominated(solutions)
                    igd = calculate_igd(pareto_front, non_dominated_solutions)
                    igd_values.append((file, igd))
            igd_values.sort(key=lambda x: x[1])
            avg_igd = np.mean([igd for _, igd in igd_values])
            results[subdir] = {
                "average_igd": avg_igd,
                "files_sorted_by_igd": [file for file, _ in igd_values],
            }
    with open(os.path.join(directory_path, "igd.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python idg.py <directory_path>")
        sys.exit(1)
    main(sys.argv[1])
