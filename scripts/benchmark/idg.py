import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.spatial import distance


def calculate_igd(pareto_front, solutions):
    igd = 0
    for pf_point in pareto_front:
        min_dist = np.min([distance.euclidean(pf_point, sol) for sol in solutions])
        igd += min_dist
    return igd / len(pareto_front)


def main(directory_path, option):
    results = {}
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            pareto_front_path = os.path.join("data", "ground_truth", "pareto_fronts", f"{subdir}_300.csv")
            if not os.path.exists(pareto_front_path):
                continue
            pareto_front = pd.read_csv(pareto_front_path, header=None).values
            igd_values = []
            for sub_sub in os.listdir(os.path.join(subdir_path, "objective")):
                solutions = []
                file_path = os.path.join(subdir_path, "objective", sub_sub)
                if sub_sub.endswith(".csv"):
                    solutions = pd.read_csv(file_path).values
                if option == "-par":
                    solutions = solutions[:, 1:]
                solutions = np.array(solutions)
                igd = calculate_igd(pareto_front, solutions)
                igd_values.append((sub_sub, igd))
            igd_values.sort(key=lambda x: x[1])
            avg_igd = np.mean([igd for _, igd in igd_values])
            min = igd_values[0]
            max = igd_values[-1]
            median = igd_values[(len(igd_values) // 2 if len(igd_values) % 2 == 0 else len(igd_values) // 2 + 1)]
            std = np.std([igd for _, igd in igd_values])
            results[subdir] = {
                "averageIgd": avg_igd,
                "standardDeviation": std,
                "min": min,
                "max": max,
                "median": median,
                "igdValues": igd_values,
            }
    with open(os.path.join(directory_path, "igd.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    option = "-seq"
    if len(sys.argv) == 3:
        option = sys.argv[2]
    elif len(sys.argv) != 2:
        print("Usage: python idg.py <directory_path> [-par|-seq]")
        sys.exit(1)
    main(sys.argv[1], option)
