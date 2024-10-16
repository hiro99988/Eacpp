import os
import sys
import csv
import json


def calculate_average_execution_time(directory):
    average_times = {}

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            csv_file_path = os.path.join(subdir_path, "executionTimes.csv")
            if os.path.exists(csv_file_path):
                with open(csv_file_path, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    times = [float(row[1]) for row in reader]
                    if times:
                        average_times[subdir] = sum(times) / len(times)

    return average_times


def write_average_times_to_json(directory, average_times):
    output_file_path = os.path.join(directory, "averageExecutionTimes.json")
    with open(output_file_path, "w") as json_file:
        json.dump(average_times, json_file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python execution_time.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)

    average_times = calculate_average_execution_time(directory_path)
    write_average_times_to_json(directory_path, average_times)
