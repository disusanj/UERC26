import os
import json
import configparser
import numpy as np

from uerc26_utils import get_min_max_stats, compute_rt1, compute_rt2


# Global variables
ROOT_DIR = 'SUBMISSIONS'


if __name__ == "__main__":
    print("Aggregating metrics for all submissions")

    METRICS = {}
    MODEL_STATS = {}

    # Go through each submission directory
    for submission in os.listdir(ROOT_DIR):
        submission_path = os.path.join(ROOT_DIR, submission)

        if os.path.isdir(submission_path):
            config_path = os.path.join(submission_path, 'config.ini')

            # Read config
            config = configparser.ConfigParser()
            config.read(config_path)

            submitted_track = config.get("SUBMISSION", "track")
            submission_name = config.get("SUBMISSION", "name")

            metrics_file = os.path.join(submission_path, "metrics.json")

            if not os.path.exists(metrics_file):
                print(f"No metrics found for {submission}, skipping...")
                continue

            with open(metrics_file, "r") as f:
                METRICS[submission] = json.load(f)

            # Load model stats for this submission
            stats_file = os.path.join(submission_path, "model_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    MODEL_STATS[submission] = json.load(f)

            else:
                print(f"No model stats found for {submission}, skipping...")

    # Save the aggregated metrics for all submissions in a JSON file for later analysis
    with open("aggregated_metrics.json", "w") as f:
        json.dump(METRICS, f, indent=4)


    # Calculate average metrics across all submissions
    if METRICS:
        avg_metrics = {}
        for submission in METRICS.keys():
            avg_metrics[submission] = {}

            RUN_IDS = [key for key in METRICS[submission].keys()]

            for key in METRICS[submission]["0"].keys():
                avg_metrics[submission][key] = np.mean([METRICS[submission][ri][key] for ri in RUN_IDS])

            for key in MODEL_STATS[submission].keys():
                avg_metrics[submission][key] = MODEL_STATS[submission][key]

        stats = get_min_max_stats(avg_metrics)

        for submission in METRICS.keys():
            if submitted_track == "T1":
                score = compute_rt1(avg_metrics[submission], stats)
            else:
                score = compute_rt2(avg_metrics[submission], stats)

            avg_metrics[submission]["score"] = score

        # Print the aggregated metrics for all submissions
        for submission, metrics in avg_metrics.items():
            print(f"Metrics for {submission}:")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value}")
            print("\n")
