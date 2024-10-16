import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    data_dir = "out/data//MpMoead/241016-135052/ZDT1/idealPoint/1"
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv")],
        key=lambda x: int(x.split(".")[0]),
    )

    for file in files:
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path, header=None).values

        x_vals, y_vals = zip(*data)
        label = file.split(".")[0]
        if label == "0" or label == "49" or label == "24":
            plt.plot(
                x_vals,
                y_vals,
                label=label,
                linestyle="-",
            )

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Ideal Point Data Points")
    plt.show()

    # save_dir = "out/data/mp_moead/plots/ideal_point"
    # os.makedirs(save_dir, exist_ok=True)
    # output_path = os.path.join(save_dir, "1_500_17_5_299.png")
    # plt.savefig(output_path)


if __name__ == "__main__":
    main()
