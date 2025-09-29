"""
全ての試行の最終目的関数値を同時にプロットする
"""

import os
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from base import *

linewidth = 5
pointsize = 3


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python all_objective.py <dir>")
        sys.exit(1)

    dir = args[0]

    # PARETO_FRONT_DIRに存在するcsvファイル名を取得（.csvは除外）
    pareto_front_csv_files = glob.glob(os.path.join(PARETO_FRONT_DIR, "*.csv"))
    pareto_front_problems = [os.path.splitext(os.path.basename(f))[0] for f in pareto_front_csv_files]

    for problem_dir in get_non_results_directories(dir):
        problem_name = os.path.basename(problem_dir)
        # pareto_front_problemsのいずれかでproblem_nameが始まる場合のみ処理を行う．一致したときは，そのpareto_front_problemの名前を取得する
        matching_pf = next((pf for pf in pareto_front_problems if problem_name.startswith(pf)), None)
        if matching_pf is None:
            print(f"Skipping {problem_name} as no matching Pareto front found.")
            continue
        print(f"Processing {problem_name}...")
        pareto_front_csv = os.path.join(PARETO_FRONT_DIR, matching_pf + ".csv")
        if not os.path.exists(pareto_front_csv):
            print(f"Pareto front file not found: {pareto_front_csv}")
            continue
        df = pd.read_csv(pareto_front_csv)
        obj_num = sum(1 for col in df.columns if col.startswith("f"))
        if obj_num < 2 or obj_num > 3:
            print(f"Objective number is not 2 or 3 for {problem_name}. Skipping.")
            continue
        pareto_front_obj1 = df["f1"].values
        pareto_front_obj2 = df["f2"].values
        if obj_num == 3:
            pareto_front_obj3 = df["f3"].values
        for algorithm_dir in get_non_results_directories(problem_dir):
            if obj_num == 2:
                # 2次元プロット
                plt.figure()
                plt.grid()
                plt.plot(
                    pareto_front_obj1, pareto_front_obj2, label=PARETO_FRONT_LABEL, color="black", linewidth=linewidth, zorder=0
                )
                for trial_csv in glob.glob(os.path.join(algorithm_dir, CsvSchema.Objective.DIR, "*.csv")):
                    df = pd.read_csv(trial_csv)
                    obj1 = df[CsvSchema.Objective.OBJECTIVE1].values
                    obj2 = df[CsvSchema.Objective.OBJECTIVE2].values
                    plt.scatter(obj1, obj2, s=pointsize, zorder=1)
                plt.xlabel("Objective1")
                plt.ylabel("Objective2")
                plt.title(f"{problem_name} - {os.path.basename(algorithm_dir)}")
                plt.legend()
                plt.tight_layout()
            elif obj_num == 3:
                # 3次元プロット
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.grid()
                ax.scatter(
                    pareto_front_obj1,
                    pareto_front_obj2,
                    pareto_front_obj3,
                    label=PARETO_FRONT_LABEL,
                    color="black",
                    s=pointsize * 5,
                    zorder=0,
                )
                for trial_csv in glob.glob(os.path.join(algorithm_dir, CsvSchema.Objective.DIR, "*.csv")):
                    df = pd.read_csv(trial_csv)
                    obj1 = df[CsvSchema.Objective.OBJECTIVE1].values
                    obj2 = df[CsvSchema.Objective.OBJECTIVE2].values
                    obj3 = df[CsvSchema.Objective.OBJECTIVE3].values
                    ax.scatter(obj1, obj2, obj3, s=pointsize, zorder=1)
                ax.view_init(azim=225)
                ax.set_xlabel("Objective1")
                ax.set_ylabel("Objective2")
                ax.set_zlabel("Objective3")
                ax.set_title(f"{problem_name} - {os.path.basename(algorithm_dir)}")
                ax.legend()
                plt.tight_layout()

            basename = os.path.basename(os.path.normpath(dir))
            output_dir = os.path.join(dir, Results.DIR, "all_objective")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{problem_name}_{os.path.basename(algorithm_dir)}.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
