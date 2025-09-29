import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from base import *

# stats 内で使うキー
TRIAL_KEY = "trial"
TIME_KEY = "time"
RANK_KEY = "rank"
AVG_KEY = "average"
STD_KEY = "standardDeviation"
MAX_KEY = "max"
MIN_KEY = "min"
MEDIAN_KEY = "median"
IGD_PLUS_KEY = "igd+"
TIME_DIR = "time"
EXEC_DIR = "execution_time"
RANK_DIR = "rank"

# TODO: IGD以外の指標に対応するため，指標のKEY名をcsvの最後のヘッダーから取得し，IGD_KEYの代わりに使用する


def compute_stats(values: np.ndarray, value_key: str) -> dict:
    """平均・標準偏差・最大・最小・中央値の trial/value を返す."""
    sorted_idx = np.argsort(values)

    def entry(i):
        return {TRIAL_KEY: int(i + 1), value_key: float(values[i])}

    return {
        AVG_KEY: float(np.mean(values)),
        STD_KEY: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        MAX_KEY: entry(sorted_idx[-1]),
        MIN_KEY: entry(sorted_idx[0]),
        MEDIAN_KEY: entry(sorted_idx[len(sorted_idx) // 2]),
    }


def make_sorted_df(values: np.ndarray, col_name: str) -> pd.DataFrame:
    """ソート済みの DataFrame を作成（rank, trial, {col_name}）."""
    idx_sorted = np.argsort(values)
    return pd.DataFrame(
        {
            RANK_KEY: np.arange(1, len(values) + 1),
            TRIAL_KEY: idx_sorted + 1,
            col_name: values[idx_sorted],
        }
    )


def calculate_indicator_statistics(algorithm_dir: str):
    """
    指定されたアルゴリズムディレクトリのすべての指標 CSV から， 最終値の統計を計算する
    """
    indicator_dir = os.path.join(algorithm_dir, CsvSchema.Igd.DIR)
    final_values = []
    indicator_key = None
    files = get_sorted_trial_csv_files(indicator_dir)
    for file in files:
        df = pd.read_csv(file)
        if indicator_key is None:
            indicator_key = df.columns[-1]
        final_values.append(df[indicator_key].values[-1])
    final_values = np.array(final_values)
    stats = compute_stats(final_values, indicator_key)
    df = make_sorted_df(final_values, indicator_key)
    return stats, df, final_values


def calculate_execution_time_statistics(algorithm_dir: str):
    """
    指定されたアルゴリズムディレクトリから、実行時間と通信時間の統計を計算する
    """
    df = pd.read_csv(os.path.join(algorithm_dir, CsvSchema.ElapsedTime.FILE))
    exec_times = df[CsvSchema.ElapsedTime.EXECUTION_TIME].values

    e = compute_stats(exec_times, TIME_KEY)

    stats = {
        JsonSchema.Time.AVG_EXEC: e[AVG_KEY],
        JsonSchema.Time.STD_EXEC: e[STD_KEY],
        JsonSchema.Time.MAX_EXEC: e[MAX_KEY],
        JsonSchema.Time.MIN_EXEC: e[MIN_KEY],
        JsonSchema.Time.MED_EXEC: e[MEDIAN_KEY],
    }

    execution_times_df = make_sorted_df(exec_times, CsvSchema.ElapsedTime.EXECUTION_TIME)

    return stats, execution_times_df, exec_times


def setup_output_dirs(base_dir: str):
    """
    結果を保存するためのディレクトリを作成する
    """
    time_dir = os.path.join(base_dir, Results.DIR, TIME_DIR)
    exec_dir = os.path.join(time_dir, EXEC_DIR)
    wilcoxon_dir = os.path.join(base_dir, Results.DIR, "wilcoxon")
    # TODO: ヘッダー名を使用
    indicator_dir = os.path.join(base_dir, Results.DIR, "igd+")
    indicator_rank_dir = os.path.join(indicator_dir, RANK_DIR)
    os.makedirs(exec_dir, exist_ok=True)
    os.makedirs(wilcoxon_dir, exist_ok=True)
    os.makedirs(indicator_rank_dir, exist_ok=True)
    return time_dir, exec_dir, wilcoxon_dir, indicator_dir, indicator_rank_dir


def collect_problem_stats(problem_dir: str):
    """
    指定された問題のディレクトリから、すべてのアルゴリズムの実行時間と指標の統計を収集する
    """
    exec_stats_list = {}
    indicator_stats_list = {}
    exec_dfs = {}
    indicator_dfs = {}
    exec_times_list = {}
    indicator_values_list = {}
    for algo_dir in get_non_results_directories(problem_dir):
        name = os.path.basename(algo_dir)
        exec_stats, exec_df, exec_times = calculate_execution_time_statistics(algo_dir)
        indicator_stats, indicator_df, indicator_values = calculate_indicator_statistics(algo_dir)
        exec_stats_list[name] = exec_stats
        indicator_stats_list[name] = indicator_stats
        exec_dfs[name] = exec_df
        indicator_dfs[name] = indicator_df
        exec_times_list[name] = exec_times
        indicator_values_list[name] = indicator_values
    return (
        exec_stats_list,
        indicator_stats_list,
        exec_dfs,
        indicator_dfs,
        exec_times_list,
        indicator_values_list,
    )


def save_csvs(dir: str, dfs: dict):
    """
    各アルゴリズムの DataFrame を CSV ファイルとして保存する
    """
    for name, df in dfs.items():
        df.to_csv(os.path.join(dir, f"{name}.csv"), index=False)


def save_stats_json(stats_list: dict, output_path: str):
    """
    統計情報を JSON ファイルとして保存する
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            stats_list,
            f,
            indent=4,
        )


def perform_wilcoxon_and_plot(exec_times_list, indicator_values_list, output_path, title):
    """
    全アルゴリズム同士の実行時間と指標値についてウィルコクソン符号順位検定を行い、
    アルゴリズム名を行列の軸に配置し、対戦結果を表示する関数。
    '+': 行のアルゴリズムが列のアルゴリズムより有意に良い(小さい)
    '-': 行のアルゴリズムが列のアルゴリズムより有意に悪い(大きい)
    '~': 有意差なし
    """
    alpha = 0.05
    algorithms = list(exec_times_list.keys())
    n_algos = len(algorithms)

    # 実行時間とIGD+の2つの比較結果マトリックスを作成
    exec_matrix = [["" for _ in range(n_algos)] for _ in range(n_algos)]
    igd_matrix = [["" for _ in range(n_algos)] for _ in range(n_algos)]

    # 全対全比較を実行
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i == j:
                exec_matrix[i][j] = "~"  # 対角線は自分自身なので有意差なし
                igd_matrix[i][j] = "~"
                continue

            # 実行時間の比較
            try:
                stat_exec, p_exec = wilcoxon(exec_times_list[algo1], exec_times_list[algo2], alternative="two-sided")
                if p_exec < alpha:
                    stat_exec, p_exec = wilcoxon(
                        exec_times_list[algo1],
                        exec_times_list[algo2],
                        alternative="less",
                    )
                    exec_matrix[i][j] = "+" if p_exec < alpha else "-"
                else:
                    exec_matrix[i][j] = "~"
            except (ValueError, ZeroDivisionError):
                # エラーの場合は中央値で直接比較
                median1_exec = np.median(exec_times_list[algo1])
                median2_exec = np.median(exec_times_list[algo2])
                if abs(median1_exec - median2_exec) < 1e-10:  # ほぼ同じ
                    exec_matrix[i][j] = "~"
                else:
                    exec_matrix[i][j] = "+" if median1_exec < median2_exec else "-"

            # IGD+の比較
            try:
                # 差分を計算してウィルコクソン検定を実行
                diff_igd = np.array(indicator_values_list[algo1]) - np.array(indicator_values_list[algo2])

                # すべて同じ値の場合はスキップ
                if np.all(diff_igd == 0):
                    igd_matrix[i][j] = "~"
                else:
                    stat_igd, p_igd = wilcoxon(diff_igd)
                    if p_igd < alpha:
                        # 差分の中央値で判定（負なら algo1 が良い）
                        median_diff_igd = np.median(diff_igd)
                        igd_matrix[i][j] = "+" if median_diff_igd < 0 else "-"
                    else:
                        igd_matrix[i][j] = "~"
            except (ValueError, ZeroDivisionError):
                # エラーの場合は中央値で直接比較
                median1_igd = np.median(indicator_values_list[algo1])
                median2_igd = np.median(indicator_values_list[algo2])
                if abs(median1_igd - median2_igd) < 1e-10:  # ほぼ同じ
                    igd_matrix[i][j] = "~"
                else:
                    igd_matrix[i][j] = "+" if median1_igd < median2_igd else "-"

    # 1つの表を作成（実行時間とIGD+の結果を統合）
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 結合されたデータを作成
    # 左上のセルにタイトル、その他のヘッダー行にアルゴリズム名
    combined_data = [[title] + algorithms]

    for i, algo in enumerate(algorithms):
        row = [algo]  # 行の最初にアルゴリズム名
        for j in range(len(algorithms)):
            if i == j:
                # 対角線の場合
                cell_text = "~/~"
            else:
                # 実行時間/IGD+の形式で結合
                cell_text = f"{exec_matrix[i][j]}/{igd_matrix[i][j]}"
            row.append(cell_text)
        combined_data.append(row)

    # 表を作成
    table = ax.table(cellText=combined_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.axis("off")

    # セルの背景色とスタイルを設定
    n_rows = len(combined_data)
    n_cols = len(combined_data[0])

    # タイトルセル（左上）の設定
    table[(0, 0)].set_text_props(ha="center", fontweight="bold", fontsize=12)
    table[(0, 0)].set_facecolor("#D0D0D0")

    # ヘッダー行の設定（アルゴリズム名）
    for j in range(1, n_cols):
        table[(0, j)].set_facecolor("#E0E0E0")
        table[(0, j)].set_text_props(ha="center", fontweight="bold")

    # ヘッダー列の設定（アルゴリズム名）
    for i in range(1, n_rows):
        table[(i, 0)].set_facecolor("#E0E0E0")
        table[(i, 0)].set_text_props(ha="center", fontweight="bold")

    # 対角線セルの設定
    for i in range(1, min(n_rows, n_cols)):
        table[(i, i)].set_facecolor("#F0F0F0")

    # 表のサイズを調整
    table.scale(1, 2)

    # 凡例を追加
    legend_text = "Format: Execution Time / IGD+\n+: Significantly better, -: Significantly worse, ~: No significant difference"
    ax.text(
        0.5,
        -0.1,
        legend_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def statistics():
    args = sys.argv[1:]
    if not args:
        print("Usage: python execution_time.py <dir> …")
        sys.exit(1)

    for base_dir in args:
        (
            time_dir,
            exec_dir,
            wilcoxon_dir,
            indicator_dir,
            indicator_rank_dir,
        ) = setup_output_dirs(base_dir)
        for problem_dir in get_non_results_directories(base_dir):
            problem_name = os.path.basename(problem_dir)
            (
                exec_stats_list,
                indicator_stats_list,
                exec_dfs,
                indicator_dfs,
                exec_times_list,
                indicator_values_list,
            ) = collect_problem_stats(problem_dir)

            problem_exec_dir = os.path.join(exec_dir, problem_name)
            problem_indicator_dir = os.path.join(indicator_rank_dir, problem_name)
            os.makedirs(problem_exec_dir, exist_ok=True)
            os.makedirs(problem_indicator_dir, exist_ok=True)
            save_csvs(problem_exec_dir, exec_dfs)
            save_csvs(problem_indicator_dir, indicator_dfs)

            exec_json_path = os.path.join(time_dir, f"{problem_name}.json")
            indicator_json_path = os.path.join(indicator_dir, f"{problem_name}.json")
            save_stats_json(exec_stats_list, exec_json_path)
            save_stats_json(indicator_stats_list, indicator_json_path)

            perform_wilcoxon_and_plot(
                exec_times_list,
                indicator_values_list,
                os.path.join(wilcoxon_dir, f"{problem_name}.png"),
                problem_name,
            )


if __name__ == "__main__":
    statistics()
