import pandas as pd
import numpy as np
import os
import sys

def add_features(dataframe, steps_in_day=24):

    new_df = dataframe.copy()
    for column in dataframe.columns:
        new_df[(column + "_Diff")] = new_df[column].diff(1).fillna(0)
        new_df[(column + "_DailyAvg")] = new_df[column].rolling(steps_in_day).mean().fillna(0)
        new_df[(column + "_DailyDiff")] = new_df[column].diff(steps_in_day).fillna(0)

    return new_df

def extract_dataframe(data_dir, file_name):
    """
    Extracts a dataframe from a CSV file.

    Args:
        data_dir (str): The directory where the CSV file is located.
        file_name (str): The name of the CSV file.

    Returns:
        pandas.DataFrame: The extracted dataframe.
    """
    df = pd.read_csv(os.path.join(data_dir, file_name))
    df = df.drop(df.columns[0], axis=1)

    return df

import torch
def extract_data(dataframe):
    """
    Extracts the columns and values from a given dataframe.
    
    Args:
        dataframe (pandas.DataFrame): The dataframe from which to extract the data.
        
    Returns:
        Tuple[pandas.Index, torch.Tensor]: A tuple containing the columns of the dataframe
        and a tensor of the dataframe values.
    """
    return dataframe.columns, torch.tensor(dataframe.values)
def normalize_data(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std, mean, std

def write_last_line(string):
    sys.stdout.write("\r")
    sys.stdout.flush()
    print(string, end="")


def print_duration(duration):
    """
    Generates a formatted string representation of a given duration in hours, minutes, seconds, and milliseconds.
    
    Args:
        duration (float): The duration in seconds.
    
    Returns:
        str: The formatted string representation of the duration in the format "hours:minutes:seconds.milliseconds".
    """
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60 // 1
    miliseconds = abs((duration % 1) * 1000)
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}.{int(miliseconds)}"

def floating_window_batch_generator(
    data,
    sample_len,
    target_len,
    min_index,
    max_index,
    shuffle=False,
    batch_size=128,
    step=6,
):
    """
    Generate a batch of subsequences of data.

    Args:
        data: (samples, features)
        sample_len: number of samples in the past.
        target_len: number of samples to predict
        min_index: index of first sample
        max_index: index after the last sample
        shuffle: (boolean) if False, return batches in chronological order.
        batch_size: (int)
        step: (int) number of timesteps between samples
    """
    n_features = data.shape[-1]
    if max_index is None:
        max_index = len(data) - target_len - 1
    i = min_index

    window_size = (sample_len + target_len) * step

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index, max_index - (sample_len + target_len) * step, size=batch_size
            )
        else:
            if i + batch_size >= max_index - (sample_len + target_len) * step:
                i = min_index  # reset to beggining
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = torch.zeros((len(rows), sample_len, n_features))
        targets = torch.zeros((len(rows), target_len, n_features))
        for j, row in enumerate(rows):
            indices = range(row, row + window_size, step)
            samples[j] = data[indices[:sample_len]]
            targets[j] = data[indices[sample_len:]]

        yield samples, targets