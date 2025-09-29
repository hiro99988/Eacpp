import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import os
import re

# 論文でのType 3フォントの使用を避けるための設定
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.rcParams.update({"font.size": 18})

PARETO_FRONT_DIR = "data/ground_truth/pareto_fronts"
PARETO_FRONT_LABEL = "Reference Pareto Front"

# TODO: execution_time.py, igd.py の文字列を定数化する
# TODO: csvヘッダー，json変数の名前をスネークケースにする
# TODO: 指標の csv のヘッダーを igd -> value に変更する


class Results:
    """
    結果を保存する results ディレクトリ名を定義するクラス
    """

    DIR = "results"
    IGD = DIR + "/igd"
    IGD_PLUS = DIR + "/igd+"


class CsvSchema:
    """
    CSVファイルのディレクトリ名とヘッダー定義を一括管理するクラス
    """

    class IdealPoint:
        DIR = "idealPoint"
        RANK = "rank"
        GENERATION = "generation"
        OBJECTIVE1 = "objective1"
        OBJECTIVE2 = "objective2"

        def OBJECTIVE(self, i):
            """
            目的数に応じたヘッダーを返す
            """

            return f"objective{i}"

    class Igd:
        DIR = "igdPlus"
        GENERATION = "generation"
        EXECUTION_TIME = "execution_time_s"
        IGD = "igd+"

    class Objective:
        DIR = "objective"
        RANK = "rank"
        OBJECTIVE1 = "objective1"
        OBJECTIVE2 = "objective2"
        OBJECTIVE3 = "objective3"

        def OBJECTIVE(self, i):
            """
            目的数に応じたヘッダーを返す
            """

            return f"objective{i}"

    class ElapsedTime:
        FILE = "elapsedTimes.csv"
        TRIAL = "trial"
        INITIALIZATION_TIME = "initialization_time_s"
        EXECUTION_TIME = "execution_time_s"


class JsonSchema:
    """
    JSONファイルのスキーマを定義するクラス
    """

    class Indicator:
        AVG_IGD = "average"
        STD = "standardDeviation"
        MIN = "min"
        MAX = "max"
        MED = "median"
        VALUES = "values"

    class Time:
        AVG_EXEC = "averageExecutionTime"
        STD_EXEC = "standardDeviationExecutionTime"
        MAX_EXEC = "maxExecutionTime"
        MIN_EXEC = "minExecutionTime"
        MED_EXEC = "medianExecutionTime"


def get_non_results_directories(dir):
    """
    指定されたディレクトリ内の、results ディレクトリを除くすべてのサブディレクトリを取得する
    """
    return [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d != Results.DIR]


def convert_algorithm_name(text):
    """
    アルゴリズム名を正式名称に変換する関数
    """
    ALGORITHM_SUBSTITUTIONS = {"MOEAD": "MOEA/D"}

    # Replace longer keys first to avoid overlapping issues.
    for src, target in sorted(ALGORITHM_SUBSTITUTIONS.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(src, target)
    return text


def plot_combined_figures(
    data_dict,
    xlabel,
    ylabel,
    save_path,
    log_scale=True,
    dpi=-1,
    plot_mode="plot",
):
    titles = list(data_dict.keys())
    n_figs = len(titles)
    cols = int(np.ceil(np.sqrt(n_figs)))
    rows = int(np.ceil(n_figs / cols))
    # 1:1.414は黄金比らしい
    fig, axs = plt.subplots(rows, cols, figsize=((1.414 * 4) * cols, 4 * rows))
    if n_figs == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # 各問題ごとにプロット
    for ax, title in zip(axs, titles):
        # マーカーや線種を循環的に取得
        marker_cycle = cycle(["o", "s", "^", "D", "x", "*", "v", "p"])
        linestyle_cycle = cycle(["-", "--", "-.", ":"])
        for algorithm, xy_data in data_dict[title].items():
            x_data, y_data = list(zip(*xy_data))
            # 次のマーカーと線種に進める
            marker = next(marker_cycle)
            linestyle = next(linestyle_cycle)
            if plot_mode == "scatter":
                if algorithm == PARETO_FRONT_LABEL:
                    ax.scatter(
                        x_data,
                        y_data,
                        label=convert_algorithm_name(algorithm),
                        s=10,
                        color="gray",
                        marker=marker,
                    )
                else:
                    ax.scatter(
                        x_data,
                        y_data,
                        label=convert_algorithm_name(algorithm),
                        s=10,
                        marker=marker,
                    )
            else:
                ax.plot(
                    x_data,
                    y_data,
                    label=convert_algorithm_name(algorithm),
                    linestyle=linestyle,
                )
        # 各サブプロットの設定
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both")

    # 不要なサブプロットを削除
    for ax in axs[len(titles) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if dpi == -1:
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, dpi=dpi)
    plt.show()
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def plot_individual(data_dict, title, xlabel, ylabel, save_path, log_scale=True, dpi=-1, plot_mode="plot"):
    # マーカーや線種を循環的に取得
    marker_cycle = cycle(["o", "s", "^", "D", "x", "*", "v", "p"])
    linestyle_cycle = cycle(["-", "--", "-.", ":"])
    plt.figure()
    for algorithm, xy_data in data_dict.items():
        # 次のマーカーと線種に進める
        marker = next(marker_cycle)
        linestyle = next(linestyle_cycle)
        if len(xy_data[0]) == 2:
            x_data, y_data = list(zip(*xy_data))
            std = None
        elif len(xy_data[0]) == 3:
            x_data, y_data, std = list(zip(*xy_data))
        if plot_mode == "scatter":
            if algorithm == PARETO_FRONT_LABEL:
                plt.scatter(
                    x_data,
                    y_data,
                    label=convert_algorithm_name(algorithm),
                    s=10,
                    color="gray",
                    marker=marker,
                )
            else:
                plt.scatter(
                    x_data,
                    y_data,
                    label=convert_algorithm_name(algorithm),
                    s=10,
                    marker=marker,
                )
        else:
            if std is not None:
                plt.fill_between(
                    x_data,
                    np.array(y_data) - np.array(std),
                    np.array(y_data) + np.array(std),
                    alpha=0.2,
                )
            plt.plot(x_data, y_data, label=convert_algorithm_name(algorithm), linestyle=linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    plt.legend(fontsize=14)
    plt.grid(True, which="both")  # x, y 両方のグリッドを表示
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if dpi == -1:
        plt.savefig(save_path)
    else:
        plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Figure saved to {save_path}")


def plot_all_individual(data_dict, xlabel, ylabel, save_dir, log_scale=True, dpi=-1, plot_mode="plot"):
    for title, subdict in data_dict.items():
        save_path = os.path.join(save_dir, f"{title}.pdf")
        plot_individual(subdict, title, xlabel, ylabel, save_path, log_scale, dpi, plot_mode)


def save_to_csv(data_dict, save_dir, header_prefix="objective", headers=None):
    os.makedirs(save_dir, exist_ok=True)
    for problem, algorithms in data_dict.items():
        problem_dir = os.path.join(save_dir, problem)
        os.makedirs(problem_dir, exist_ok=True)
        for algo, data in algorithms.items():
            filename = f"{algo}.csv"
            file_path = os.path.join(problem_dir, filename)
            # data の次元だけヘッダーを作成
            headers = [header_prefix + str(i + 1) for i in range(len(data[0]))]
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(file_path, index=False)


def get_sorted_csv_files(dir_path: str, num_index: int) -> list[str]:
    """
    指定ディレクトリ内の .csv ファイルを取得し，
    ファイル名から抽出した数値の num_index 番目をキーにソートしてパスのリストを返す．

    Parameters
    ----------
    dir_path : str
        CSV ファイルが格納されているディレクトリのパス
    num_index : int
        ファイル名中から抽出した数字リストの何番目をソートキーに使うか (0 始まり)

    Returns
    -------
    List[str]
        ソート済み CSV ファイルのフルパスリスト
    """
    # ディレクトリ内の .csv ファイルをすべて収集
    csv_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(dir_path, f))
    ]

    def key_fn(path):
        # ファイル名から数字をすべて抽出
        nums = re.findall(r"\d+", os.path.basename(path))
        # 対象インデックスに数字があれば int 扱いで返す，なければ大きい値
        try:
            return int(nums[num_index])
        except (IndexError, ValueError):
            return float("inf")

    # 抽出した数値をもとにソート
    return sorted(csv_files, key=key_fn)


def get_sorted_trial_csv_files(dir_path: str) -> list[str]:
    """
    指定ディレクトリ内の trial_*.csv ファイルを取得し，試行番号でソートしてパスのリストを返す．

    Parameters
    ----------
    dir_path : str
        CSV ファイルが格納されているディレクトリのパス

    Returns
    -------
    List[str]
        ソート済み trial_*.csv ファイルのフルパスリスト
    """
    return get_sorted_csv_files(dir_path, 0)
