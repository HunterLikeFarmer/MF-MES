import pandas as pd

# Read CSV of high fidelity globally
df = pd.read_csv('result.csv').drop_duplicates()

# get the x for high fidelity data
def get_pair_high_fidelity():
    return df[['A100','V100']].to_numpy().tolist() 

# get the y for high fidelity data
def get_wall_clock_times_high_fidelity():
    return df['Wall Clock Time'].tolist()

# Read CSV of low fidelity globally
df_low = pd.read_csv('result.csv').drop_duplicates()

# get the x for low fidelity data
def get_pair_low_fidelity():
    return df_low[['A100','V100']].to_numpy().tolist() 

# get the y for low fidelity data
def get_wall_clock_times_low_fidelity():
    return df_low['Wall Clock Time'].tolist()

if __name__ == "__main__":
    print(get_pair_high_fidelity())
    print(get_wall_clock_times_high_fidelity())
    