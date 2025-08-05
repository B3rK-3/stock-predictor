import os
import sys
import time
import warnings

import numpy as np
import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from functions import (
    lstm_ready,
    min_max_scaling,
    training_loop_w_stats,
    PlateauStopper,
)
from ray import tune
import ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import re
import pandas as pd
import pandas_ta as pdt
from eval import eval_tune

isVanillaLSTM = True

if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM

    model_type = "VanillaLSTM"
else:
    from functions import LSTM as LSTM

    model_type = "LSTM"


# Assume these are defined in a 'functions.py' file or similar
# from functions import LSTM, lstm_ready, min_max_scaling

warnings.filterwarnings("ignore")

# --- Configuration ---
# Define constants and configurations at the top level for clarity.

l = re.split(r"[\\/]", os.path.abspath(os.getcwd()))
BASE_PATH = "/".join(l[:-1]) + "/"

DATA_PATH = BASE_PATH + "stock-predictor/data"
RESULTS_PATH = BASE_PATH + "stock-predictor/results"
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists

# --- Data Loading & Preparation ---


def load_data(stock):
    """Loads and preprocesses data for a single Active Region (AR)."""
    try:
        data = pd.read_csv(
            f"{DATA_PATH}/{stock}.csv"
        )  # "ticker","name","date","open","high","low","close","adjusted close","volume"
        # data["sma_5"] = pdt.sma(data["close"], length=5)
        data["ema_5"] = pdt.ema(data["close"], length=5)
        data["rsi_14"] = pdt.rsi(data["close"], length=14)
        data["macd"] = pdt.macd(data["close"])["MACD_12_26_9"]
        data["atr"] = pdt.atr(data["high"], data["low"], data["close"])
        # BBM_20 = pdt.bbands(data["close"], length=20)
        # data["BBU_20"] = BBM_20["BBU_20_2.0"]
        # data["BBM_20"] = BBM_20["BBM_20_2.0"]
        # data["BBL_20"] = BBM_20["BBL_20_2.0"]
        data["close_roc"] = data["close"].diff()
        data["volume_roc"] = data["volume"].diff()
        data["ema_diff"] = data["ema_5"].diff()

        # remove unnecessary data
        del data["name"]
        del data["date"]
        del data["ticker"]
        del data["adjusted close"]

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
        mn = np.min(train_data[column])
        mx = np.max(train_data[column])
        train_data[column] = min_max_scaling(train_data[column], mn, mx)
        test_data[column] = min_max_scaling(test_data[column], mn, mx)

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

    return X_train, y_train, X_test, y_test, len(columns)


# --- Model Training & Evaluation ---
def train(model, dataloader, loss_fn, optimizer, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss


def eval_model(model, dataloader, loss_fn, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item() * x.size(0)

    return total_loss


def training_loop(
    model,
    n_epochs,
    num_in,
    num_layers,
    num_pred,
    hidden_size,
    stock,
    optimizer,
    loss_fn,
    train_loader,
    test_loader,
    scheduler,
    device,
):
    results = []
    for epoch in range(n_epochs):
        training_loss = train(model, train_loader, loss_fn, optimizer, device)
        test_loss = eval_model(model, test_loader, loss_fn, device)
        train_loss = training_loss / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)

        # if epoch % int(n_epochs/100) == 0:
        learning_rate = scheduler.get_last_lr()[0]
        print(
            "Epoch: %d, train loss: %1.8f, test loss: %1.8f, learning rate: %1.8f"
            % (epoch, train_loss, test_loss, learning_rate)
        )
        RMSE = eval_tune(
            device, model.state_dict(), num_in, num_layers, num_pred, hidden_size, stock
        )
        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "lr": learning_rate,
            "RMSE": RMSE,
        }
        results.append(result)  # Collect results for saving

        tune.report(result)
        wandb.log(result)
        scheduler.step(test_loss)

    return results


# --- Main Execution ---
def main(config):
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # --- Data Loading ---
    print("Batch size:", config["batch_size"])
    print("Loading and preparing training data...")
    stocks = ["A"]
    for stock in stocks:
        # Initialize wandb
        wandb.init(
            project="LSTM_ADAM",
            entity=os.environ.get("WANDB_ENTITY"),
            config=config,
            name=f"{stock}_p{config['num_pred']}_in{config['num_in']}_l{config['num_layers']}_h{config['hidden_size']}",
            notes=f"LSTM training with lr={config['learning_rate']}, dropout={config['dropout']}",
        )
        x_train, y_train, x_test, y_test, input_size = prepare_dataset(
            stock,
            config["num_in"],
            config["num_pred"],
        )

        # --- Model & Optimizer ---
        model = LSTM(
            input_size,
            config["hidden_size"],
            config["num_layers"],
            config["num_pred"],
            dropout=config["dropout"],
        ).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=5)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=config["batch_size"],
            shuffle=False,
        )

        # --- Training Loop ---
        print("Starting training...")
        results = training_loop(
            model,
            config["n_epochs"],
            config["num_in"],
            config["num_layers"],
            config["num_pred"],
            config["hidden_size"],
            stock,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            scheduler=scheduler,
            device=device,
        )

        # print(results)
        # --- Save Model & Artifacts ---
        model_name = f"{stock}_p{config['num_pred']}_in{config['num_in']}_l{config['num_layers']}_h{config['hidden_size']}.pth"
        model_path = os.path.join(RESULTS_PATH, model_name)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        model_artifact = wandb.Artifact(
            name=f"lstm-model-{wandb.run.id}",
            type="model",
            description="LSTM Model for SAR emergence prediction",
            metadata=config,
        )
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")
        wandb.finish()


def parse_args():
    """Parses command-line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python train.py <grid_search sample_size>")
        sys.exit(1)

    try:
        config = {
            "sample_size": int(sys.argv[1]),
        }
        return config
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For this refactoring to be fully functional, you must provide
    # the implementations for these functions from your 'functions.py' file.
    config = parse_args()

    search_space = {
        "num_pred": tune.choice([2]),
        "num_in": tune.choice([7]),
        "num_layers": tune.choice([1]),
        "hidden_size": tune.choice([10, 30, 50, 100, 150, 200]),
        "n_epochs": tune.choice([500]),
        "learning_rate": tune.choice([0.01, 0.0001, 0.001, 0.00001]),
        "dropout": tune.choice([0, 0.01, 0.1]),
        "batch_size": tune.choice([4, 8, 16]),
    }
    algo = OptunaSearch()
    scheduler = ASHAScheduler(max_t=500, grace_period=10, reduction_factor=3)

    custom_stopper = PlateauStopper(
        "RMSE", min_epochs=21, patience=10, min_improvement_percent=0.5
    )

    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
    tuner = tune.Tuner(  # ③
        tune.with_resources(main, {"gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="RMSE",
            mode="min",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=config["sample_size"],
            trial_dirname_creator=lambda trial: str(trial.trial_id),
        ),
        run_config=tune.RunConfig(
            stop=custom_stopper,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
