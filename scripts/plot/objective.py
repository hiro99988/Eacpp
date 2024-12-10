import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(problem="ZDT1", option="-par"):
    data_file = "out/data/NtMoead/241210-130616/ZDT3/objective/trial_28.csv"

    pareto_front = pd.read_csv(f"data/ground_truth/pareto_fronts/{problem}_300.csv", header=None).values
    plt.plot(
        pareto_front[:, 0],
        pareto_front[:, 1],
        color="lightgray",
        linestyle="-",
        label="Pareto Front",
        zorder=1,
    )

    data = pd.read_csv(data_file).values
    if option == "-par":
        data = data[:, 1:]
    plt.scatter(
        data[:, 0],
        data[:, 1],
        s=15,
        zorder=2,
    )

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Objective Data Points")
    plt.show()

    # save_dir = "out/data/tmp"
    # os.makedirs(save_dir, exist_ok=True)
    # output_path = os.path.join(save_dir, "MpMoeadZDT1_.png")
    # plt.savefig(output_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem = sys.argv[1]
    else:
        problem = "ZDT1"
    if len(sys.argv) > 2:
        option = sys.argv[2]
    else:
        option = "-par"
    main(problem=problem, option=option)
