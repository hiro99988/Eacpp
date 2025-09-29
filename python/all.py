import sys
import subprocess


def main():
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python all.py <dir>")
        sys.exit(1)

    dir = args[0]

    print("statistics.pyを実行します...")
    subprocess.run(["python3", "python/statistics.py", dir])
    print("time_igd.pyを実行します...")
    subprocess.run(["python3", "python/time_igd.py", dir])
    print("ideal_point_measure.pyを実行します...")
    subprocess.run(["python3", "python/ideal_point_measure.py", dir])
    print("median_objective.pyを実行します...")
    subprocess.run(["python3", "python/median_objective.py", dir])
    print("all_gen_igd.pyを実行します...")
    subprocess.run(["python3", "python/all_gen_igd.py", dir])
    print("all_objective.pyを実行します...")
    subprocess.run(["python3", "python/all_objective.py", dir])
    print("all.pyが終了しました。")


if __name__ == "__main__":
    main()
