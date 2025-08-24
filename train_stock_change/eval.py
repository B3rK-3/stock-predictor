from functions import (
    calculate_metrics,
)
import sys
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from collections import OrderedDict
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import warnings
import torch
import glob
import re
import os
import wandb
import pandas as pd
import pandas_ta as pdt
from functions import prepare_dataset

isVanillaLSTM = True
if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM
else:
    from functions import LSTM as LSTM

warnings.filterwarnings("ignore")

l = re.split(r"[\\/]", os.path.abspath(os.getcwd()))
BASE_PATH = "/".join(l[:-1]) + "/"

DATA_PATH = BASE_PATH + "stock-predictor/eval_data"
RESULTS_PATH = BASE_PATH + "stock-predictor/results"
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists


def load_data(path):
    """Loads and preprocesses data for a single Active Region (AR)."""
    try:
        loaded = np.load(path, allow_pickle=True)
        return (
            None,
            None,
            torch.from_numpy(loaded["x_tests"]),
            torch.from_numpy(loaded["y_tests"]),
            loaded["times_tests"],
            loaded["x_trains"].shape[2],
        )
    except FileNotFoundError:
        print(f"Warning: Data file for {path} not found. Skipping.")
        return None


def eval(device, filename):
    matches = re.findall(
        r"ALL_p(\d+)_in(\d+)_l(\d+)_h(\d+).pth",
        filename,
    )  # Extract numbers from the filename
    (
        num_pred,
        num_in,
        num_layers,
        hidden_size,
    ) = [
        int(val) for i, val in enumerate(matches[0])
    ]  # Unpack the matched values into variables
    filepath = RESULTS_PATH + "/" + filename
    stock = "NVDA"
    print(
        f"Extracted from filename: Stock: {stock} Time Window: {num_pred}, Number of Inputs: {num_in}, Number of Layers: {num_layers}, Hidden Size: {hidden_size}"
    )  # Print extracted values for confirmation

    _, _, x_test, y_test, time, input_size = load_data('stock_data_eval.npz')

    # Initialize the LSTM and move it to GPU

    # Initialize the LSTM and move it to GPU
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = torch.load(filepath, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()  # Set the model to evaluation model

    # Assuming prediction, y_test_tensors, ARs, learning_rate, and n_epochs are already defined
    fig = plt.figure()  # Adjust the figure size if necessary

    # Loop to create 8 plots
    future = 1
    all_metrics = []
    with torch.no_grad():
        x_test = x_test.to(device)
        predictions = lstm(x_test)
        pred = predictions[:, future].detach().cpu().numpy()
        true = y_test[:, future].numpy()
    start = num_in + future

    offset = num_in + future
    print(
        f"Window length (num_in): {num_in}, horizon‐index (future): {future}, offset = {offset}"
    )


    pred_times = time.reshape(-1,1)

    # now plot
    ax = plt.subplot()
    ax.plot(pred_times[-50:], pred.reshape(-1,1)[-50:], label="predicted")
    ax.plot(pred_times[-50:], true.reshape(-1,1)[-50:], label="actual")
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


def eval_tune(
    device,
    state_dict,
    num_in,
    num_layers,
    num_pred,
    hidden_size,
    evalnpz_path,
    batch_size,
):
    _, _, x_test, y_test, time, input_size = load_data(evalnpz_path)

    # Initialize the LSTM and move it to GPU
    lstm = LSTM(input_size, hidden_size, num_layers, num_pred).to(device)
    saved_state_dict = state_dict
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' prefix
        new_state_dict[name] = v
    lstm.load_state_dict(new_state_dict)
    lstm.eval()  # Set the model to evaluation model

    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    # Loop to create 8 plots
    future = 1
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            predictions = lstm(x)
            pred_batch = predictions[:, future].detach().cpu().numpy()
            true_batch = y[:, future].cpu().numpy()
            all_predictions.append(pred_batch)
            all_actuals.append(true_batch)

    preds = np.concatenate(all_predictions, axis=0)
    trues = np.concatenate(all_actuals, axis=0)

    # Evaluation metrics
    metrics = calculate_metrics(trues, preds)
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
    eval(device, sys.argv[1])
