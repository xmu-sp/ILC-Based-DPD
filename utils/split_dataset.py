import pandas as pd
import os

# Load data
input_data = pd.read_csv('../datasets/OFDM_100MHz/Input.csv')
output_data = pd.read_csv('../datasets/OFDM_100MHz/Output.csv')


def partition_data(input_df, output_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    n = len(input_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        input_df.iloc[:train_end], output_df.iloc[:train_end],
        input_df.iloc[train_end:val_end], output_df.iloc[train_end:val_end],
        input_df.iloc[val_end:], output_df.iloc[val_end:]
    )


# Split and save
datasets = partition_data(input_data, output_data)
names = ['train_input', 'train_output', 'val_input', 'val_output', 'test_input', 'test_output']

for name, df in zip(names, datasets):
    df.to_csv(os.path.join('./', f'{name}.csv'), index=False)