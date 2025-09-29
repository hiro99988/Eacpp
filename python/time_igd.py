import os
import glob
import pandas as pd
import numpy as np
import sys

from base import *


def calculate_time_avg_igd(algorithm_dir):
    igd_dir = os.path.join(algorithm_dir, CsvSchema.Igd.DIR)
    # 全ての試行の実行時間
    trials = []
    for trial_csv in glob.glob(os.path.join(igd_dir, "*.csv")):
        df = pd.read_csv(trial_csv)
        trials.append(df)

    # 最後のヘッダー名を取得
    indicator_header = trials[0].columns[-1]

    # 全ての試行の中で最小の実行時間を取得
    min_elapsed_time = min(df[CsvSchema.Igd.EXECUTION_TIME].iloc[-1] for df in trials)
    # 全ての試行の世代間の時間差の平均を計算
    avg_interval = np.mean([np.mean(np.diff(df[CsvSchema.Igd.EXECUTION_TIME].values)) for df in trials])
    # 最小の実行時間までのavg_interval秒ごとのIGDの平均を計算
    time_avg_igd = []
    for t in np.arange(0, min_elapsed_time, avg_interval):
        igd_values = []
        for df in trials:
            eligible = df[df[CsvSchema.Igd.EXECUTION_TIME] <= t]
            if not eligible.empty:
                igd_values.append(eligible[indicator_header].iloc[-1])
        avg_igd = np.mean(igd_values) if igd_values else np.nan
        # time_avg_igd.append([t, avg_igd])
        std_igd = np.std(igd_values)
        time_avg_igd.append([t, avg_igd, std_igd])

    return time_avg_igd


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python time_igd.py <dir> ...")
        sys.exit(1)

    for base_dir in args:
        time_avg_igd_dict = {}
        for problem_dir in get_non_results_directories(base_dir):
            problem_name = os.path.basename(problem_dir)
            # if not problem_name.startswith("DTLZ1-3-7"):
            #     continue
            print(f"Processing {problem_name}...")
            time_avg_igd_dict[problem_name] = {}
            for algorithm_dir in get_non_results_directories(problem_dir):
                algorithm_name = os.path.basename(algorithm_dir)
                time_avg_igd_dict[problem_name][algorithm_name] = calculate_time_avg_igd(algorithm_dir)

        time_igd_output_dir = os.path.join(base_dir, Results.DIR, "time_igd")
        save_to_csv(time_avg_igd_dict, time_igd_output_dir)

        plot_all_individual(time_avg_igd_dict, "Execution Time (s)", "Average IGD+ (log)", time_igd_output_dir, dpi=300)


if __name__ == "__main__":
    main()
