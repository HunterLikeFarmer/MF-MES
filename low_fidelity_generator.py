import pandas as pd
import numpy as np
import argparse

def add_noise(input_file, output_file, noise_level=0.1, seed=42):
    """Add Gaussian noise to Wall Clock Time and save new file."""
    np.random.seed(seed)

    # Load data
    df = pd.read_csv(input_file)

    # Add noise
    noise = np.random.normal(0, noise_level, size=len(df))
    df_noisy = df.copy()
    df_noisy['Wall Clock Time'] = df['Wall Clock Time'] * (1 + noise)

    # Save new file
    df_noisy.to_csv(output_file, index=False)
    print(f"Low-fidelity data written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate low-fidelity noisy version of Wall Clock Time.")
    parser.add_argument("--input", type=str, default="cluster_cost.csv", help="Input Excel file.")
    parser.add_argument("--output", type=str, default="cluster_cost_low_fidelity.csv", help="Output Excel file.")
    parser.add_argument("--noise", type=float, default=0.4, help="Noise level (default: 0.1 = 10%%).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    add_noise(args.input, args.output, noise_level=args.noise, seed=args.seed)
