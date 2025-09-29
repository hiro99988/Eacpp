"""
全ての試行のIGDを問題・アルゴリズムごとにプロットする．
横軸が世代数、縦軸がIGD，全ての試行のigdの推移をプロットする．
"""

import os
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt

from base import *


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python all_gen_igd.py <dir>")
        sys.exit(1)

    dir = args[0]

    for problem_dir in get_non_results_directories(dir):
        problem_name = os.path.basename(problem_dir)
        print(f"Processing {problem_name}...")
        for algorithm_dir in get_non_results_directories(problem_dir):
            for trial_csv in glob.glob(os.path.join(algorithm_dir, CsvSchema.Igd.DIR, "*.csv")):
                df = pd.read_csv(trial_csv)
                generation = df[CsvSchema.Igd.GENERATION].values
                igd_values = df[CsvSchema.Igd.IGD].values
                plt.plot(generation, igd_values)
            plt.xlabel("Generation")
            plt.ylabel("IGD")
            plt.title(f"{problem_name} - {os.path.basename(algorithm_dir)}")
            plt.grid()
            plt.tight_layout()
            basename = os.path.basename(os.path.normpath(dir))
            output_dir = os.path.join(dir, Results.DIR, "all_gen_igd")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{problem_name}_{os.path.basename(algorithm_dir)}.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
