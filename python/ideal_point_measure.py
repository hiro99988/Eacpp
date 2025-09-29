import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
import json
import glob

from base import *


def get_ideal_point(gen, data):
    ideal_point = data[0, 1:]
    for i in range(0, len(data) - 1):
        if int(data[i + 1, 0]) > gen:
            return ideal_point
        ideal_point = data[i + 1, 1:]
    return ideal_point


def fill_in_blanks(data, generations):
    filled_data = []
    current_ideal = data[0, 1:]
    gen = 0
    for i in range(len(data) - 1):
        while gen < data[i + 1, 0]:
            filled_data.append(current_ideal)
            gen += 1
        current_ideal = data[i + 1, 1:]

    for i in range(gen, generations + 1):
        filled_data.append(current_ideal)

    return np.array(filled_data)


def calculate_avg_aipd(df, generations, generation_interval):
    unique_ranks = df[CsvSchema.IdealPoint.RANK].unique()
    # 各ランクのデータを取得
    grouped_data = {
        rank: df[df[CsvSchema.IdealPoint.RANK] == rank].loc[:, CsvSchema.IdealPoint.GENERATION :].to_numpy()
        for rank in unique_ranks
    }
    # 欠如している世代のデータを保管する
    ideal_points = np.array([fill_in_blanks(grouped_data[rank], generations) for rank in unique_ranks])
    # 各世代の理想点と最小値との差の平均を算出
    avg_distances_raw = []
    for i in range(generations + 1):
        # 全グループの理想点の最小値を算出
        min_ideal_point = np.min(ideal_points[:, i], axis=0)
        # 理想点と最小値との差（距離）の平均を算出
        avg_distance = np.mean(np.linalg.norm(ideal_points[:, i] - min_ideal_point, axis=1))
        avg_distances_raw.append(avg_distance)

    return np.array(avg_distances_raw)


def calculate_gen_avg_aipd(algorithm_dir, generation_interval):
    ideal_point_dir = os.path.join(algorithm_dir, CsvSchema.IdealPoint.DIR)
    # 全ての試行の理想点
    trials = []
    for trial_csv in glob.glob(os.path.join(ideal_point_dir, "*.csv")):
        df = pd.read_csv(trial_csv)
        trials.append(df)

    # 全ての試行の中で最大の世代数を取得
    max_generation = max(df[CsvSchema.IdealPoint.GENERATION].max() for df in trials)

    trial_avg_aipds = np.array([calculate_avg_aipd(df, max_generation, generation_interval) for df in trials])
    # 各世代の平均を計算
    avg_aipds = np.mean(trial_avg_aipds, axis=0)
    std_aipds = np.std(trial_avg_aipds, axis=0)

    # 世代間隔での平均を計算
    gen_avg_aipds = []
    if generation_interval == 1:
        for i, avg_aipd in enumerate(avg_aipds):
            gen_avg_aipds.append([i, avg_aipd, std_aipds[i]])
    else:
        for i in range(0, len(avg_aipds), generation_interval):
            mean_gen = i + generation_interval // 2
            avg = np.mean(avg_aipds[i : i + generation_interval])
            gen_avg_aipds.append([mean_gen, avg, np.mean(std_aipds[i : i + generation_interval])])

    return gen_avg_aipds


def main():
    args = sys.argv[1:]

    if len(args) < 1 or len(args) > 2:
        print("Usage: python ideal_point_measure.py <dir> [<generation_interval>]")
        sys.exit(1)

    dir = args[0]
    generation_interval = int(args[1]) if len(args) == 2 else 1

    gen_avg_aipd_dict = {}
    for problem_dir in get_non_results_directories(dir):
        problem_name = os.path.basename(problem_dir)

        print(f"Processing {problem_name}...")
        gen_avg_aipd_dict[problem_name] = {}
        for algorithm_dir in get_non_results_directories(problem_dir):
            algorithm_name = os.path.basename(algorithm_dir)
            print(f"  Processing {algorithm_name}...")
            gen_avg_aipd_dict[problem_name][algorithm_name] = calculate_gen_avg_aipd(algorithm_dir, generation_interval)

    # 結果をcsvファイルに保存
    avg_aipd_output_dir = os.path.join(dir, Results.DIR, "avg_aipd")
    save_to_csv(
        gen_avg_aipd_dict,
        avg_aipd_output_dir,
    )

    plot_all_individual(
        gen_avg_aipd_dict,
        xlabel="Generation",
        ylabel="Average AIPD (log)",
        save_dir=avg_aipd_output_dir,
        log_scale=True,
        dpi=300,
    )


if __name__ == "__main__":
    main()
