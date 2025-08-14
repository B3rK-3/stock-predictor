from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
import glob
import re
import os
import wandb
from functions import (
    prepare_dataset,
    calculate_metrics,
    DATA_PATH,
    RESULTS_PATH,
    BASE_PATH,
    isVanillaLSTM,
)

if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM
else:
    from functions import LSTM as LSTM

warnings.filterwarnings("ignore")


def eval(device):
    pth_files = glob.glob(
        RESULTS_PATH + "/*.pth"
    )  # Assuming there's only one .pth file and its naming follows the specific pattern
    filename = pth_files[0]
    matches = re.findall(
        r"(\w+)_p(\d+)_in(\d+)_l(\d+)_h(\d+).pth",
        filename,
    )  # Extract numbers from the filename
    (
        stock,
        num_pred,
        num_in,
        num_layers,
        hidden_size,
    ) = [
        val if i == 0 else int(val) for i, val in enumerate(matches[0])
    ]  # Unpack the matched values into variables
    print(
        f"Extracted from filename: Stock: {stock} Time Window: {num_pred}, Number of Inputs: {num_in}, Number of Layers: {num_layers}, Hidden Size: {hidden_size}"
    )  # Print extracted values for confirmation

    _, _, x_test, y_test, time, input_size = prepare_dataset(stock, num_in, num_pred)

    # Initialize the LSTM and move it to GPU
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = torch.load(filename, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()  # Set the model to evaluation model

    # Assuming prediction, y_test_tensors, ARs, learning_rate, and n_epochs are already defined
    fig = plt.figure()  # Adjust the figure size if necessary

    future = 1
    all_metrics = []
    x_test = x_test.to(device)
    predictions = lstm(x_test)
    pred = predictions[:, future].detach().cpu().numpy()
    true = y_test[:, future].numpy()
    start = num_in + future

    offset = num_in + future
    print(
        f"Window length (num_in): {num_in}, horizon‐index (future): {future}, offset = {offset}"
    )

    for i in range(5):
        # the timestamp you’re using to plot pred[i]:
        t_pred = time.iloc[offset + i]
        # the true timestamp for y_test[i]:
        t_true = time.iloc[num_in + future + i]
        print(
            f"i={i:>2} | plot‐time={t_pred} "
            f"| actual‐time={t_true} "
            f"| pred={pred[i]:.3f} | true={true[i]:.3f}"
        )
    pred_times = time.iloc[start : start + len(pred)].to_numpy()

    # now plot
    ax = plt.subplot()
    ax.plot(pred_times, pred, label="predicted")
    ax.plot(pred_times, true, label="actual")
    ax.set_title(f"{future + 1}-hour ahead prediction for {stock}")
    ax.legend()

    # Evaluation metrics
    metrics = calculate_metrics(true, pred)
    all_metrics.append(metrics)
    # print(f"MAE: {metrics[0]}")
    # print(f"MSE: {metrics[1]}")
    print(f"RMSE: {metrics[2]}")
    # print(f"RMSLE: {metrics[3]}")
    # print(f"R-squared: {metrics[4]}")
    plt.show()

    # Print the metrics at the bottom
    all_metrics_np = np.array(
        all_metrics
    )  # Convert all_metrics to a NumPy array for easier manipulation
    means = np.mean(
        all_metrics_np, axis=0
    )  # Calculate the mean and standard deviation for each metric across the 7 runs
    stds = np.std(all_metrics_np, axis=0)
    mae_string = r"Average metrics for all tiles plotted:  $\mathrm{{MAE}} = {}$,  $\mathrm{{MSE}} = {}$,  $\mathrm{{RMSE}} = {}$,  $\mathrm{{RMSLE}} = {}$,  $R^2 = {}$".format(
        round(means[0], 3),
        round(means[1], 3),
        round(means[2], 3),
        round(means[3], 3),
        round(means[4], 3),
    )
    return metrics[2]


def eval_tune(device, state_dict, num_in, num_layers, num_pred, hidden_size, stock):
    _, _, x_test, y_test, time, input_size = prepare_dataset(stock, num_in, num_pred)

    # Initialize the LSTM and move it to GPU
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = state_dict
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()  # Set the model to evaluation model

    future = 1
    all_metrics = []
    x_test = x_test.to(device)
    predictions = lstm(x_test)
    pred = predictions[:, future].detach().cpu().numpy()
    true = y_test[:, future].numpy()

    # Evaluation metrics
    metrics = calculate_metrics(true, pred)
    all_metrics.append(metrics)
    print("Metrics:", metrics)
    # print(f"MAE: {metrics[0]}")
    # print(f"MSE: {metrics[1]}")
    print(f"RMSE: {metrics[2]}")
    # print(f"RMSLE: {metrics[3]}")
    # print(f"R-squared: {metrics[4]}")
    return metrics[2]


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Define the device (either 'cuda' for GPU or 'cpu' for CPU)
    print("Runs on: {}".format(device), " / Using", torch.cuda.device_count(), "GPUs!")
    eval(device)
