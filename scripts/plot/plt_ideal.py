import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    data_dir = "out/data/MpMoead/241017-021237/ZDT1/idealPoint/13"
    moead_dir = "out/data/Moead/241017-043208/ZDT1/idealPoint/1.csv"

    data = pd.read_csv(moead_dir, header=None).values
    gen, x_vals, y_vals = zip(*data)
    plt.plot(
        x_vals,
        y_vals,
        label="Sequential MOEA/D" + f": {len(x_vals)} times",
        linestyle="-",
        color="red",
    )
    plt.scatter(x_vals[:-1], y_vals[:-1], s=5, color="red")
    plt.scatter(x_vals[-1], y_vals[-1], s=40, color="red")

    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv")],
        key=lambda x: int(x.split(".")[0]),
    )

    color = ["orange", "blue", "green"]
    color_idx = 0
    for file in files:
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path, header=None).values

        gen, x_vals, y_vals = zip(*data)
        label = file.split(".")[0]
        if label == "0" or label == "49" or label == "24":
            plt.plot(
                x_vals,
                y_vals,
                label="rank " + str(int(label) + 1) + f": {len(x_vals)} times",
                linestyle="-",
                color=color[color_idx],
            )
            plt.scatter(x_vals[:-1], y_vals[:-1], s=5, color=color[color_idx])
            plt.scatter(x_vals[-1], y_vals[-1], s=40, color=color[color_idx])
            color_idx += 1

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.legend()
    # plt.show()

    save_dir = "out/data/plots/"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "idealPoint.png")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
