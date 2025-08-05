import pandas as pd
import pandas_ta as pdt
import re, os, numpy as np
import torch


l = re.split(r"[\\/]", os.path.abspath(os.getcwd()))
BASE_PATH = "/".join(l[:-1]) + "/"

DATA_PATH = BASE_PATH + "stock-predictor/data"
RESULTS_PATH = BASE_PATH + "stock-predictor/results"
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists


def load_data(stock):
    """Loads and preprocesses data for a single Active Region (AR)."""
    try:
        data = pd.read_csv(
            f"{DATA_PATH}/{stock}.csv"
        )  # "ticker","name","date","open","high","low","close","adjusted close","volume"
        data["low"] = data["low"].pct_change()
        data["high"] = data["high"].pct_change()
        data["close"] = data["close"].pct_change()
        data["open"] = data["open"].pct_change()

        data["rsi_14"] = pdt.rsi(data["close"], length=14)
        data["atr"] = pdt.atr(data["high"], data["low"], data["close"], length=14)
        data["volume_ratio"] = data["volume"] / data["volume"].rolling(30).mean()

        macd_df = pdt.macd(data["close"], fast=12, slow=26, signal=9)
        data["macd"] = macd_df["MACD_12_26_9"]
        data["macd_signal"] = macd_df["MACDs_12_26_9"]
        

        # remove unnecessary data
        del data["name"]
        del data["date"]
        del data["ticker"]
        del data["adjusted close"]
        del data["volume"]

        return data
    except FileNotFoundError:
        print(f"Warning: Data file for {stock} not found. Skipping.")
        return None


def prepare_dataset(stock, num_in, num_pred):
    # Load data
    data = load_data(stock)
    data.dropna(inplace=True)

    columns = list(data.columns)
    train_size = int(len(data) * 0.8)
    train_data = data[columns].iloc[:train_size]
    test_data = data[columns].iloc[train_size:]

    for column in columns:
        mean = np.mean(train_data[column])
        std = np.std(train_data[column])
        train_data[column] = (train_data[column] - mean) / std
        test_data[column] = (test_data[column] - mean) / std

    # Create sequences for the LSTM
    x_train, y_train = [], []
    for i in range(len(train_data) - num_in - num_pred):
        x_train.append(train_data[columns].iloc[i : i + num_in].values)
        y_train.append(
            train_data["close"].iloc[i + num_in : i + num_in + num_pred].values
        )

    x_test, y_test = [], []
    for i in range(len(test_data) - num_in - num_pred):
        x_test.append(test_data[columns].iloc[i : i + num_in].values)
        y_test.append(
            test_data["close"].iloc[i + num_in : i + num_in + num_pred].values
        )

    X_train = torch.from_numpy(
        np.array(x_train, dtype=np.float32)
    )  # shape: [N_train, num_in, n_features]
    y_train = torch.from_numpy(
        np.array(y_train, dtype=np.float32)
    )  # shape: [N_train, num_pred]
    X_test = torch.from_numpy(np.array(x_test, dtype=np.float32))
    y_test = torch.from_numpy(np.array(y_test, dtype=np.float32))

    print(X_train)
    print(X_test)
    return X_train, y_train, X_test, y_test, len(columns)
