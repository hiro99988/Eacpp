import os
import matplotlib.pyplot as plt
import numpy as np


def main():
    data_dir = "out/data/mp_moead/objective"
    files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.startswith("objective-") and f.endswith(".txt")
        ],
        key=lambda x: int(x.split("-")[1].split(".")[0]),
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
        data = []
        with open(file_path, "r") as f:
            for line in f:
                x, y = map(float, line.strip().split())
                data.append((x, y))

        x_vals, y_vals = zip(*data)
        label = file.split("-")[1].split(".")[0]
        plt.scatter(x_vals, y_vals, label=label, s=15, zorder=2)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Objective Data Points")
    plt.show()


if __name__ == "__main__":
    main()
