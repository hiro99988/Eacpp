import argparse
import matplotlib.pyplot as plt
import numpy as np

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

# CSVファイルからデータを読み込む
data = np.loadtxt(args.path, delimiter=",")

# 点と線を描画
plt.plot(data[:, 0], data[:, 1], linestyle="-")

# グラフを表示
plt.show()
