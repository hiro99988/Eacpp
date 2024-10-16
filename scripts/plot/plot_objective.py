import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    data_dir = "out/data/MpMoead/241016-201434/ZDT1/objective/10"
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".csv")],
        key=lambda x: int(x.split(".")[0]),
    )

    pareto_front = np.loadtxt(
        "data/ground_truth/pareto_fronts/ZDT1_300.csv", delimiter=","
    )
    plt.plot(
        pareto_front[:, 0],
        pareto_front[:, 1],
        color="lightgray",
        linestyle="-",
        label="Pareto Front",
        zorder=1,
    )

    for file in files:
        file_path = os.path.join(data_dir, file)
        data = pd.read_csv(file_path, header=None).values

        x_vals, y_vals = data[:, 0], data[:, 1]
        label = file.split(".")[0]
        plt.scatter(x_vals, y_vals, s=15, zorder=2)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Objective Data Points")
    plt.show()

    # save_dir = "out/data/mp_moead/plots/objective"
    # os.makedirs(save_dir, exist_ok=True)
    # output_path = os.path.join(save_dir, "1_500_17_5_299.png")
    # plt.savefig(output_path)


if __name__ == "__main__":
    main()
