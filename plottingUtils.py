import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List

def plot_samples(samples, targets, feature, figsize=(12, 12)):
    assert feature < samples.shape[-1]

    chosen_sample = samples[0, :, feature]
    chosen_target = targets[0, :, feature]

    plt.figure(figsize=figsize)
    sample_x = np.linspace(0, len(chosen_sample), len(chosen_sample))
    target_x = np.linspace(
        len(chosen_sample) + 1,
        len(chosen_sample) + len(chosen_target),
        len(chosen_target),
    )

    plt.plot(sample_x, chosen_sample, color="red")
    plt.plot(target_x, chosen_target, color="black")

    plt.show()


# model plot



def plot_history(history: Dict[str, Dict[str, List]]):
    val_data = history["val"]
    train_data = history["train"]
    nr_metrics = len(train_data.keys())
    fig = plt.figure(figsize=(8, 4 * nr_metrics))

    for i, name in enumerate(train_data.keys()):
        ax = fig.add_subplot(nr_metrics, 1, i + 1)
        plt.plot(train_data[name], label="train " + name, color="b")
        plt.plot(val_data[name], label="val " + name, color="r")
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.legend()

    plt.show()


def plot_predictions(header, samples, targets, output, mean, std):
    samples = samples.cpu().detach()
    targets = targets.cpu().detach()
    output = output.cpu().detach()

    samples = samples * std + mean
    targets = targets * std + mean
    output = output * std + mean

    for feature in range(samples.shape[-1]):
        print(header[feature])
        temp_sample = samples[0, :, feature]
        temp_target = targets[0, :, feature]
        temp_output = output[0, :, feature]
        plt.figure(figsize=(20, 4))
        plt.plot(
            torch.cat((temp_sample, temp_target), dim=0),
            color="black",
            label="Ground Truth",
        )
        plt.plot(
            torch.cat((temp_sample, temp_output), dim=0),
            color="red",
            label="Prediction",
        )
        plt.legend()
        plt.show()
