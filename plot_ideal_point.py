import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm


def main():
    # ディレクトリ内の全ファイルを取得
    files = os.listdir("out/data/mp_moead/ideal_point")

    # 色のリスト
    colors = ["green", "red"]

    # 各ファイルのデータをプロット
    for i, file in enumerate(files):
        file_path = os.path.join("out/data/mp_moead/ideal_point", file)
        data = np.loadtxt(file_path)
        plt.scatter(data[:, 0], data[:, 1], s=10, color=cm.hsv(i / len(files)), label=i)

    plt.legend()

    # プロットの表示
    plt.show()


if __name__ == "__main__":
    main()
