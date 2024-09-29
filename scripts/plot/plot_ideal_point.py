import os
import matplotlib.pyplot as plt


def main():
    data_dir = "out/data/mp_moead/ideal_point"
    files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.startswith("ideal_point-") and f.endswith(".txt")
        ],
        key=lambda x: int(x.split("-")[1].split(".")[0]),
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
        plt.scatter(x_vals, y_vals, label=label, s=15)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Ideal Point Data Points")
    plt.show()


if __name__ == "__main__":
    main()
