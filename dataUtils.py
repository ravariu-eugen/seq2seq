import pandas as pd
import numpy as np
import os
import sys
import torch

def add_features(dataframe, steps_in_day=24):
    """
    Generates a new dataframe with 3 additional features based on the input dataframe.
    
    Parameters:
        - dataframe (DataFrame): The input dataframe.
        - steps_in_day (int, optional): The number of steps in a day. Default is 24.
    
    Returns:
        - new_df (DataFrame): The new dataframe with additional features.
    """
    new_df = dataframe.copy()
    for column in dataframe.columns:
        new_df[(column + "_Diff")] = new_df[column].diff(1).fillna(0)
        new_df[(column + "_DailyAvg")] = (
            new_df[column].rolling(steps_in_day).mean().fillna(0)
        )
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
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.sort_values(by=['Datetime'], inplace=True)
    df.index = df['Datetime']
    df = df.drop(df.columns[0], axis=1)

    return df





def extract_data(dataframe):
    """
    Extracts the columns and values from a given dataframe.

    Args:
        dataframe (pandas.DataFrame): The dataframe from which to extract the data.

    Returns:
        Tuple[pandas.Index, torch.Tensor]: A tuple containing the columns of the dataframe
        and a tensor of the dataframe values.
    """
    return dataframe.columns, torch.tensor(dataframe.values).float()


def normalize_data(data):
    """
    Normalize the given data.

    Parameters:
        data: The data to be normalized.

    Returns:
        tuple: A tuple containing three elements:
            - The normalized data.
            - The mean of the data.
            - The standard deviation of the data.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std, mean, std


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
	Generates a batch of floating windows from the given data.

	Parameters:
	- data (torch.Tensor): The input data.
	- sample_len (int): The length of each sample sequence.
	- target_len (int): The length of each target sequence.
	- min_index (int): The minimum index of the data to consider.
	- max_index (int): The maximum index of the data to consider.
	- shuffle (bool): Whether to shuffle the data. If False, the batches will be consecutive.
	- batch_size (int): The size of each batch.
	- step (int): The step size between each timestep in a window.

	Returns:
	- samples (torch.Tensor): The sample sequences.
	- targets (torch.Tensor): The target sequences.
	"""

    if max_index is None:
        max_index = len(data) - target_len - 1
    i = min_index

    window_size = (sample_len + target_len) * step

    while True:
        if shuffle:
            rows = torch.randint(min_index, max_index - window_size, size=(batch_size,))
        else:
            if i + batch_size >= max_index - window_size:
                i = min_index
            rows = torch.arange(i, i + batch_size)
            i += batch_size

        indices = torch.arange(sample_len + target_len)[None, :] * step + rows[:, None]
        samples = torch.stack(
            [data[indices[i, :sample_len]] for i in range(batch_size)]
        )
        targets = torch.stack(
            [data[indices[i, sample_len:]] for i in range(batch_size)]
        )

        yield samples, targets


def get_dataset_generators(data, sample_len, target_len, step, batch_size, shuffle=False):
    """
    Generate dataset generators for training and validation.

    Parameters:
        data (array-like): The input data for generating the dataset.
        sample_len (int): The length of each training sample.
        target_len (int): The length of the target sequence.
        step (int): The stride value for generating samples.
        batch_size (int): The number of samples in each batch.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to False.

    Returns:
        tuple: A tuple containing the training generator and the validation generator.
    """
    timesteps = len(data)
    train_max_index = int(timesteps * 0.7)

    train_generator = floating_window_batch_generator(
        data,
        sample_len=sample_len,
        target_len=target_len,
        min_index=0,
        max_index=train_max_index,
        step=step,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_generator = floating_window_batch_generator(
        data,
        sample_len=sample_len,
        target_len=target_len,
        min_index=train_max_index + 1,
        max_index=timesteps,
        step=step,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return train_generator, val_generator
