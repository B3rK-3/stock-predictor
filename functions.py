# functions to be used in pipeline

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from astropy.io import fits
import sys
from PIL import Image
import os
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import detrend
import warnings
from datetime import datetime, timedelta
import random
from ray import tune
import wandb

warnings.filterwarnings("ignore")


# Splitting Image Function
def split_image(
    image, size
):  # Splits a given image of size 512 by 512 into 9 equal-size square pieces. Returns a list of 9 PIL Image objects.
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    piece_size = width // size
    pieces = []
    for i in range(size):
        for j in range(size):
            x = j * piece_size
            y = i * piece_size
            im = Image.fromarray(image)
            piece = im.crop((x, y, x + piece_size, y + piece_size))
            pieces.append(np.array(piece))
    return np.array(pieces)


# Calculate split image mean
def get_piece_means(input_cube, size):
    parts = size**2
    means_timeline = np.zeros((np.shape(input_cube)[0], parts))
    for frame_num in range(0, np.shape(input_cube)[0]):  # for every frame
        pieces = split_image(input_cube[frame_num, :, :], size)
        means_timeline[frame_num, :] = np.array([np.mean(array) for array in pieces])
    return np.transpose(means_timeline)


# Calculate DTW
def dtws(size, pm_means):  # Should we take in account the neighboroung time points?
    dtw = np.zeros(np.shape(pm_means))
    dtwA = pm_means[:size, :]
    dtwB = pm_means[-size:, :]
    for i in range(0, size):
        # print(dtw_weights(size)[-(i+1)],dtw_weights(size)[i])
        dtw[i * size : (i + 1) * size, :] = (
            dtw_weights(size)[-(i + 1)] * dtwA + dtw_weights(size)[i] * dtwB
        )
    return dtw


# Get weighting for each row of grid
def dtw_weights(size):
    if size % 2 == 0:
        my_list = np.arange(0, size + 1)
        my_list = np.delete(my_list, int(size / 2)) / size
        print(my_list)
        sys.exit()
    else:
        my_list = np.linspace(0, 1, num=size + 1)
        average = (
            my_list[(size + 1) // 2 - 1] + my_list[(size + 1) // 2]
        ) / 2  # Calculate the average of the two middle elements
        my_list[(size + 1) // 2 - 1 : (size + 1) // 2 + 1] = [
            average
        ]  # Replace the two middle elements with the calculated average
        my_list = np.delete(my_list, int(size / 2))
    return my_list


# Straighten lines
def straighten(dtw_dist, old_line):
    new_line = 1
    return new_line


# Calculate derivatives
def derivative():
    x = 1
    return x


def scale_vid(x, pos, new_min, new_max):  # define the mapping function
    return "{:.2f}".format(new_min + ((x - 0) / (512 - 0)) * (new_max - new_min))


def print_progress_bar(
    iteration,
    total,
    prefix="Progress:",
    suffix="Complete",
    length=50,
    fill="#",
    empty="-",
    end_line="\r",
):
    progress = float(iteration) / float(total)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total} {suffix}")
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write(end_line)
        sys.stdout.flush()


# Functions for comparing frames in corrupt due to eclipse AR11726


def calculate_frame_difference_metric(framez):
    num_frames = framez.shape[0]
    frame_diffs = []
    for i in range(1, num_frames):
        diff = np.mean(np.abs(framez[i] - framez[i - 1]))
        frame_diffs.append(diff)
    return frame_diffs


def plot_frame_difference_metric(frame_diffs, cor_file):
    zero_start = None
    zero_end = None
    in_zeros = False
    for i, diff in enumerate(frame_diffs):
        if not in_zeros and diff == 0:
            in_zeros = True
            zero_start = i
        elif in_zeros and diff != 0:
            in_zeros = False
            zero_end = i - 1
            break
    if zero_start is not None and zero_end is not None:
        print(
            f"The frame_diffs array contains zeros from index {zero_start} to {zero_end} before becoming non-zero values again."
        )
    elif zero_start is not None:
        print(
            f"The frame_diffs array contains zeros starting from index {zero_start}, but non-zero values are not encountered again."
        )
    else:
        print("The frame_diffs array does not contain consecutive zeros.")
    plt.plot(frame_diffs)
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Difference Metric")
    plt.title(f"Frame Difference Metric vs. Frame Index for {cor_file}")
    plt.show()


##### Sept 18th and later


def min_max_scaling(arr, min_val, max_val):
    """
    Set values
    """
    return (arr - min_val) / (max_val - min_val)


def lstm_ready(
    tile, size, power_maps, mag_flux, num_in, num_pred
):  # ,min_p,max_p,min_i,max_i):
    # Read AR and create lstm ready data
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))
    final_flux = np.transpose(mag_flux, axes=(1, 0))
    X_trans = final_maps[:, :, tile]
    y_trans = final_flux[:, tile]
    X_ss, y_mm = split_sequences(X_trans, y_trans, num_in, num_pred)
    return torch.Tensor(X_ss), torch.Tensor(y_mm)


def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train, X_test, y_test):
    scheduler = StepLR(optimiser, step_size=n_epochs // 10, gamma=0.9)
    for epoch in range(n_epochs):
        lstm.train()
        # shuffle
        indices = torch.randperm(int(np.shape(X_train)[0]))
        X_train = X_train[indices]
        y_train = y_train[indices]
        # print(np.shape(X_train))
        outputs = lstm.forward(X_train)  # forward pass
        optimiser.zero_grad()  # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward()  # calculates the loss of the loss function
        optimiser.step()  # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        with (
            torch.no_grad()
        ):  # Turn off gradients for validation, saves memory and computations
            test_preds = lstm(X_test)
            test_loss = loss_fn(test_preds, y_test)
        # if epoch % int(n_epochs / 10) == 0:
        #     print(
        #         "Epoch: %d, train loss: %1.5f, test loss: %1.5f"
        #         % (epoch, loss.item(), test_loss.item())
        #     )
        scheduler.step()


def training_loop_w_stats(
    n_epochs, lstm, optimiser, loss_fn, X_train, y_train, X_test, y_test, scheduler
):
    results = []
    for epoch in range(n_epochs):
        lstm.train()
        # Shuffle
        indices = torch.randperm(int(np.shape(X_train)[0]))
        X_train = X_train[indices]
        y_train = y_train[indices]

        outputs = lstm.forward(X_train)  # Forward pass
        optimiser.zero_grad()  # Calculate the gradient, manually setting to 0
        loss = loss_fn(outputs, y_train)
        loss.backward()  # Calculates the loss of the loss function
        optimiser.step()  # Improve from loss, i.e., backprop

        # Test loss
        lstm.eval()
        with (
            torch.no_grad()
        ):  # Turn off gradients for validation, saves memory and computations
            test_preds = lstm(X_test)
            test_loss = loss_fn(test_preds, y_test)

        # if epoch % int(n_epochs/100) == 0:
        learning_rate = scheduler.get_last_lr()[0]
        # print(
        #     "Epoch: %d, train loss: %1.5f, test loss: %1.5f, learning rate: %1.5f"
        #     % (epoch, loss.item(), test_loss.item(), learning_rate)
        # )
        results.append(
            (epoch, loss.item(), test_loss.item(), learning_rate)
        )  # Collect results for saving
        scheduler.step(test_loss.item())

    return results


class LSTM(nn.Module):
    # __init__ stays the same...
    def __init__(self, input_size, hidden_size, num_layers, output_length, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.decoder_lstm = nn.LSTM(
            1, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        self.decoder_fc = nn.Linear(hidden_size, 1)

    # The forward pass now accepts the target tensor 'y' for teacher forcing
    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        # Encoder
        _, (hidden, cell) = self.encoder_lstm(x)

        # Decoder
        decoder_input = torch.zeros(x.size(0), 1, 1).to(x.device)
        outputs = []

        for t in range(self.output_length):
            out_dec, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            out = self.decoder_fc(out_dec)
            outputs.append(out)

            # Decide whether to use teacher forcing for the next step
            use_teacher_forcing = (y is not None) and (
                random.random() < teacher_forcing_ratio
            )

            if use_teacher_forcing:
                # Use the actual ground-truth value as the next input
                decoder_input = y[:, t].unsqueeze(1).unsqueeze(1)
            else:
                # Use the model's own prediction as the next input
                decoder_input = out

        outputs = torch.cat(outputs, dim=1).squeeze(-1)
        return outputs


class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length, dropout=0.0):
        super(VanillaLSTM, self).__init__()

        # A single LSTM layer that processes the entire input sequence
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input tensor shape: [batch_size, seq_length, input_size]
            dropout=dropout,
        )

        # A single linear layer to map the final LSTM state to the desired output length
        self.fc = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        # Pass the input sequence through the LSTM layer
        # We don't need the final hidden and cell states, just the output sequence
        lstm_out, _ = self.lstm(x)

        # We only need the output from the very last time step of the sequence
        # lstm_out has shape [batch_size, seq_length, hidden_size]
        # We take the last time step: lstm_out[:, -1, :]
        last_time_step_out = lstm_out[:, -1, :]

        # Pass the last time step's output to the linear layer to get the final prediction
        prediction = self.fc(last_time_step_out)

        return prediction


# split a multivariate sequence past, future samples (X and y)
def split_sequences(input_sequences, output_sequences, n_steps_in, n_steps_out):
    X, y = list(), list()  # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences):
            break
        # gather input and output of the pattern
        seq_x, seq_y = (
            input_sequences[i:end_ix],
            output_sequences[end_ix - 1 : out_end_ix],
        )
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


def amplify_fluctuations(y, amplification_factor=2):
    """
    Amplify the fluctuations of a data series.

    Parameters:
    - y: Data points (list or numpy array)
    - amplification_factor: Factor by which to amplify the fluctuations (default=2)

    Returns:
    - Amplified data (numpy array)
    """
    # Detrend the data
    y_detrended = detrend(y)
    # Amplify the detrended data
    y_amplified = y_detrended * amplification_factor
    # Add the trend back
    y_amplified_with_trend = y_amplified + (y - y_detrended)
    return y_amplified_with_trend

    import numpy as np


def calculate_metrics(timeline_true, timeline_predicted):
    # Ensure inputs are NumPy arrays for consistency
    timeline_true = np.array(timeline_true)
    timeline_predicted = np.array(timeline_predicted)
    # Calculate Mean Absolute Error (MAE)
    MAE = np.mean(np.abs(timeline_predicted - timeline_true))
    # Calculate Mean Squared Error (MSE)
    MSE = np.mean(np.square(timeline_predicted - timeline_true))
    # Calculate Root Mean Squared Error (RMSE)
    RMSE = np.sqrt(MSE)
    # Calculate Root Mean Squared Logarithmic Error (RMSLE)
    RMSLE = np.sqrt(
        np.mean(np.square(np.log1p(timeline_predicted) - np.log1p(timeline_true)))
    )
    # Calculate R-squared (R²)
    SS_res = np.sum(np.square(timeline_true - timeline_predicted))
    SS_tot = np.sum(np.square(timeline_true - np.mean(timeline_true)))
    R_squared = 1 - (SS_res / SS_tot)
    return MAE, MSE, RMSE, RMSLE, R_squared


def emergence_indication(d_true, threshold, sust_time):
    d_true = smooth_with_numpy(d_true)
    indicator = np.zeros(d_true.shape)  # Initialize with 0s (green)
    # Populate the indicator array
    for j in range(len(d_true)):
        if d_true[j] >= threshold:
            indicator[j] = 1  # Mark as red
    # Enforce the sustained condition
    sustained = True
    if sustained:
        start_idx = None
        for i in range(len(indicator)):
            if indicator[i] == 1 and start_idx is None:
                start_idx = i  # Start of a red sequence
            elif (
                indicator[i] == 0 and start_idx is not None
            ):  # End of a red sequence, check its length
                if i - start_idx < sust_time:
                    indicator[start_idx:i] = 0  # Sequence too short, revert to green
                start_idx = None  # Reset start index for the next sequence
        # Check for a sequence that goes till the end of the array
        if start_idx is not None and len(indicator) - start_idx < sust_time:
            indicator[start_idx:] = 0
    return indicator


def emergence_indication2(d_true):
    indicator = np.zeros(d_true.shape)  # Initialize with 1s (green)
    min_index = np.argmin(d_true)  # Find the index of the minimum value in d_true
    indicator[min_index] = 1  # Mark only the lowest value with 0 (red)
    indicator[min_index + 1] = 1
    return indicator


def smooth_with_numpy(d_true, window_size=5):
    if window_size <= 1:
        return d_true
    pad_width = window_size // 2  # Calculate the number of elements to pad on each side
    padded_d_true = np.pad(
        d_true, pad_width, mode="edge"
    )  # Pad the beginning and end of d_true with its first and last values, respectively
    window = np.ones(window_size) / window_size  # Create the smoothing window
    smoothed_d_true = np.convolve(
        padded_d_true, window, mode="same"
    )  # Apply convolution on the padded data
    return (
        smoothed_d_true[pad_width:-pad_width] if pad_width else smoothed_d_true
    )  # Remove the padding to return the smoothed array to its original length


def recalibrate(pred, previous_value):
    trend = pred - pred[0]
    new_pred = trend + previous_value
    return new_pred


def find_closest_fits_frame_to_NOAA_record(filenames, target_date):
    """
    Finds the file with timestamp closest to the target_date.
    Parameters:
    - filenames: List of strings, paths to files with timestamps in their names.
    - target_date: Datetime object, the target date to compare the file timestamps against.
    Returns:
    - The filename from filenames that is closest in time to target_date.
    """

    def extract_datetime_from_filename(filename):
        basename = os.path.basename(filename)  # Get the base name of the file
        timestamp_str = basename.split(".")[2]  # Extract the timestamp part
        file_datetime = datetime.strptime(
            timestamp_str, "%Y%m%d_%H%M%S_TAI"
        )  # Adjusted to include "_TAI"
        return file_datetime

    def get_data(
        file,
    ):  # Function designed to read FITS image data and headers from a specified file, and return the data and associated headers as output
        with fits.open(file) as file_read:
            int_grams = file_read[1].data  # get the images
            headers = file_read[1].header  # get the headers associated with the images
        return int_grams, headers

    closest_file = None
    min_difference = timedelta.max  # Initialize with maximum timedelta
    for file in filenames:
        file_datetime = extract_datetime_from_filename(file)
        difference = abs(
            file_datetime - target_date
        )  # Calculate difference as a timedelta object
        if difference < min_difference:
            min_difference = difference
            closest_file = file
    int_grams, int_headers = get_data(closest_file)
    NOAA_first_int_map = int_grams[int(np.shape(int_grams)[0] / 2 - 1), :, :]
    return NOAA_first_int_map


def add_grid_lines(ax, divisions=9, color="w", linewidth=1):
    """
    Adds grid lines to an image plot to visually divide it into a matrix.

    Parameters:
    - ax: The axes object to add grid lines to.
    - divisions: Number of divisions along each axis (default is 9 for a 9x9 grid).
    - color: Color of the grid lines.
    - linewidth: Width of the grid lines.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_spacing = np.linspace(xlim[0], xlim[1], divisions + 1)
    y_spacing = np.linspace(ylim[0], ylim[1], divisions + 1)

    for x in x_spacing[1:-1]:
        ax.axvline(x=x, color=color, linewidth=linewidth)
    for y in y_spacing[1:-1]:
        ax.axhline(y=y, color=color, linewidth=linewidth)


def highlight_tile(ax, tile_number, divisions=9, color="r", linewidth=2):
    """
    Highlights a specific tile in the grid with a colored box.

    Parameters:
    - ax: The axes object on which the grid and image are plotted.
    - tile_number: The number of the tile to highlight, in row-major order.
    - divisions: The number of divisions along each axis (assumes a square grid).
    - color: Color of the highlight box.
    - linewidth: Width of the highlight box lines.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate width and height of each tile
    tile_width = (xlim[1] - xlim[0]) / divisions
    tile_height = (ylim[1] - ylim[0]) / divisions

    # Calculate row and column index of the tile (0-indexed)
    row_idx = (tile_number - 1) // divisions
    col_idx = (tile_number - 1) % divisions

    # Calculate coordinates for the bottom-left corner of the tile
    x = xlim[0] + col_idx * tile_width
    y = ylim[1] - (row_idx + 1) * tile_height  # y coordinates go top-to-bottom

    # Create a rectangle patch to highlight the tile
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x, y),
        tile_width,
        tile_height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)


def calculate_extended_metrics(
    model, timeline_true, timeline_predicted, training_time=None
):
    """Calculate extended metrics including RMSE@1, RMSE@5, parameter count, and training time."""
    # Basic metrics
    MAE, MSE, RMSE, RMSLE, R_squared = calculate_metrics(
        timeline_true, timeline_predicted
    )

    # Calculate RMSE@1 (RMSE for 1-hour prediction)
    if len(timeline_true.shape) > 1:
        RMSE_1 = np.sqrt(
            np.mean(np.square(timeline_predicted[:, 0] - timeline_true[:, 0]))
        )
        # Calculate RMSE@5 (RMSE for 5-hour prediction)
        if timeline_true.shape[1] >= 5:
            RMSE_5 = np.sqrt(
                np.mean(np.square(timeline_predicted[:, 4] - timeline_true[:, 4]))
            )
        else:
            RMSE_5 = None
    else:
        RMSE_1 = RMSE
        RMSE_5 = None

    # Calculate parameter count
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "MAE": MAE,
        "RMSE": RMSE,
        "R2": R_squared,
        "params": param_count,
        "train_time": training_time,
        "RMSE@1": RMSE_1,
        "RMSE@5": RMSE_5,
    }


class PlateauStopper(tune.stopper.Stopper):
    """Stops trials when the metric has plateaued."""

    def __init__(
        self,
        metric: str,
        min_epochs: int = 20,
        patience: int = 10,
        min_improvement_percent: float = 1e-5,
    ):
        """
        Args:
            metric: The metric to monitor.
            min_epochs: Minimum number of epochs to run before stopping is considered.
            patience: Number of recent epochs to check for improvement.
            min_improvement: The minimum improvement required to not stop the trial.
        """
        self._metrics = metric
        self._min_epochs = min_epochs
        self._patience = patience
        self._min_improvement = min_improvement_percent
        self._trial_history = {}  # To store the history of each trial

    def __call__(self, trial_id: str, result: dict) -> bool:
        """This is called after each tune.report() call."""
        # Initialize history for a new trial
        if trial_id not in self._trial_history:
            self._trial_history[trial_id] = []

        history = self._trial_history[trial_id]
        history.append(result[self._metrics])

        # Don't stop if we haven't reached the minimum number of epochs
        if len(history) <= self._min_epochs:
            return False

        # Check for improvement over the patience window
        # We look at the best value in the last `patience` epochs
        # and compare it to the best value before that window.
        window = history[-self._patience :]
        previous_best = min(history[: -self._patience])
        current_best = min(window)

        # If there's no meaningful improvement, stop the trial
        improvement_needed = previous_best * self._min_improvement / 100
        if previous_best - current_best < improvement_needed:
            print(
                f"Stopping trial {trial_id}: "
                f"No improvement of {improvement_needed} in the last {self._patience} epochs."
            )
            return True

        return False

    def stop_all(self) -> bool:
        """This function is used to stop all trials at once. We don't need it here."""
        return False
