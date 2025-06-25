# usage: 
# python plotting_cost.py ./zresults/low_log_run1.json zresults/results_log.json zresults/normal_log_run2.json -o cost_vs_wall_clock_time_01.pdf
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Plot iteration vs. Wall Clock Time from JSON logs and a CSV cost file."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="One or more JSON log files, e.g. low_log_run1.json mixed_log_run1.json"
    )
    parser.add_argument(
        "--csv",
        default="cluster_cost.csv",
        help="Path to the CSV cost file (default: cluster_cost.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="iteration_vs_wall_clock_time.pdf",
        help="Output PDF filename (default: %(default)s)"
    )
    args = parser.parse_args()

    # Load the CSV into a DataFrame
    df_csv = pd.read_csv(args.csv)

    # A simple cycle of colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(10, 6))


    for idx, json_file in enumerate(args.json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)

        iterations = []
        wall_times = []
        min_wt = float('inf')
        for record in data:
            it   = record['cumulative_cost']
            best = record['best_so_far']
            a100 = best['A100']
            v100 = best['V100']

            match = df_csv[(df_csv['A100'] == a100) & (df_csv['V100'] == v100)]
            wt = match['Wall Clock Time'].iat[0] if not match.empty else None

            iterations.append(it)
            min_wt = min(min_wt, wt)
            wall_times.append(min_wt)

        plt.plot(
            iterations,
            wall_times,
            label=os.path.basename(json_file),
            color=colors[idx % len(colors)]
        )

    plt.xlabel('cost')
    plt.ylabel('Wall Clock Time')
    plt.title('cost vs Wall Clock Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to the user-chosen PDF
    plt.savefig(args.output, format='pdf')
    print(f"Saved plot to {args.output!r}")

if __name__ == "__main__":
    main()
