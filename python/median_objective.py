import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import glob
import numpy as np

from base import *


def extract_median_objective(directory_path):
    result = {}

    # PARETO_FRONT_DIRに存在するcsvファイル名を取得（.csvは除外）
    pareto_front_csv_files = glob.glob(os.path.join(PARETO_FRONT_DIR, "*.csv"))
    pareto_front_problems = [os.path.splitext(os.path.basename(f))[0] for f in pareto_front_csv_files]

    for problem_dir in get_non_results_directories(directory_path):
        problem_name = os.path.basename(problem_dir)
        # pareto_front_problemsのいずれかでproblem_nameが始まる場合のみ処理を行う．一致したときは，そのpareto_front_problemの名前を取得する
        matching_pf = next((pf for pf in pareto_front_problems if problem_name.startswith(pf)), None)
        if matching_pf is None:
            print(f"Skipping {problem_name} as no matching Pareto front found.")
            continue
        pareto_front_csv = os.path.join(PARETO_FRONT_DIR, matching_pf + ".csv")
        if not os.path.exists(pareto_front_csv):
            print(f"File not found: {pareto_front_csv}")
        pareto_front_df = pd.read_csv(pareto_front_csv)
        pareto_front_data = pareto_front_df[["f1", "f2"]].values.tolist()
        # 目的数を取得
        objective_columns = [col for col in pareto_front_df.columns if col.startswith("f")]
        objective_count = len(objective_columns)
        # 目的数が2でない場合はスキップ
        if objective_count != 2:
            print(f"Objective count is not 2 for {problem_name}. Skipping.")
            continue
        print(f"Processing {problem_name}...")
        for algorithm_dir in get_non_results_directories(problem_dir):
            algorithm_name = os.path.basename(algorithm_dir)
            # igdのjsonファイルを読み込む
            igd_json = os.path.join(directory_path, Results.IGD_PLUS, problem_name + ".json")
            if not os.path.exists(igd_json):
                print(f"File not found: {igd_json}")
                continue
            with open(igd_json, "r") as f:
                igd_data = json.load(f)
            # medianのtrialのcsvファイルを読み込む
            median_trial = igd_data[algorithm_name][JsonSchema.Indicator.MED]["trial"]
            median_trial_csv = os.path.join(algorithm_dir, CsvSchema.Objective.DIR, "trial_" + str(median_trial) + ".csv")
            if not os.path.exists(median_trial_csv):
                print(f"File not found: {median_trial_csv}")
                continue
            df = pd.read_csv(median_trial_csv)
            if problem_name not in result:
                result[problem_name] = {}
                result[problem_name][PARETO_FRONT_LABEL] = pareto_front_data
            # "objective1"と"objective2"の列を抽出してリストのリストに変換
            objective_data = df[[CsvSchema.Objective.OBJECTIVE1, CsvSchema.Objective.OBJECTIVE2]].values.tolist()
            result[problem_name][algorithm_name] = objective_data
    return result


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python objective_median.py <dir> ...")
        sys.exit(1)

    dirs = args[0:]

    for dir in dirs:
        result = extract_median_objective(dir)
        if not result:
            print(f"No results found in {dir}.")
            continue
        plot_all_individual(
            result,
            xlabel="Objective 1",
            ylabel="Objective 2",
            save_dir=os.path.join(dir, Results.DIR, "median_objective"),
            log_scale=False,
            dpi=300,
            plot_mode="scatter",
        )


if __name__ == "__main__":
    main()
