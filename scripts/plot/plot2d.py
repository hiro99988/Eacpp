import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm


def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="2D data plot")
    parser.add_argument("-p", "--path", type=str, help="Path to the data file")
    parser.add_argument("-t", "--title", type=str, help="Title of the plot")
    parser.add_argument("-x", "--xlabel", type=str, help="X-axis label")
    parser.add_argument("-y", "--ylabel", type=str, help="Y-axis label")
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Save the plot (y/n)",
    )
    args = parser.parse_args()

    # ファイルの拡張子をチェック
    _, file_extension = os.path.splitext(args.path)
    # データの読み込み
    if file_extension == ".csv":
        data = np.loadtxt(args.path, delimiter=",")
    elif file_extension == ".txt":
        data = np.loadtxt(args.path)
    else:
        print("Invalid file extension")
        return

    # プロット
    plt.scatter(data[:, 0], data[:, 1], s=1)
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)

    # プロットの保存
    if args.save == "y":
        plt.savefig("plot.png")

    # プロットの表示
    plt.show()


if __name__ == "__main__":
    main()
