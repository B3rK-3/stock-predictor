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
from eval import eval_AR_emergence as eval
import re

isVanillaLSTM = False

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

DATA_PATH = BASE_PATH + "SAR_EMERGENCE_RESEARCH/data"
RESULTS_PATH = BASE_PATH + "SAR_EMERGENCE_RESEARCH/lstm/results"
os.makedirs(RESULTS_PATH, exist_ok=True)  # Ensure the results directory exists

# --- Data Loading & Preparation ---


def load_ar_data(ar_num, size, rid_of_top):
    """Loads and preprocesses data for a single Active Region (AR)."""
    try:
        # Load data from .npz files
        pm_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_pmdop{ar_num}_flat.npz")
        mag_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_mag{ar_num}_flat.npz")
        int_path = os.path.join(DATA_PATH, f"AR{ar_num}", f"mean_int{ar_num}_flat.npz")

        pm_and_int = np.load(pm_path, allow_pickle=True)
        mag_flux_data = np.load(mag_path, allow_pickle=True)
        intensities_data = np.load(int_path, allow_pickle=True)

        # Unpack arrays
        power_maps = [pm_and_int[f"arr_{i}"] for i in range(4)]
        mag_flux = mag_flux_data["arr_0"]
        intensities = intensities_data["arr_0"]

        # Trim, stack, and handle NaNs
        trim_slice = slice(
            rid_of_top * size, -rid_of_top * size if rid_of_top > 0 else None
        )
        power_maps = [pm[trim_slice, :] for pm in power_maps]
        mag_flux = mag_flux[trim_slice, :]
        intensities = intensities[trim_slice, :]

        stacked_maps = np.stack(power_maps, axis=1)
        stacked_maps[np.isnan(stacked_maps)] = 0
        mag_flux[np.isnan(mag_flux)] = 0
        intensities[np.isnan(intensities)] = 0

        return stacked_maps, mag_flux, intensities

    except FileNotFoundError:
        print(f"Warning: Data files for AR {ar_num} not found. Skipping.")
        return None, None, None


def prepare_dataset(ar_list, size, rid_of_top, num_in, num_pred):
    """Builds a complete dataset (X, y) for a list of ARs."""
    all_inputs_list, all_flux_list = [], []

    # Load data for all ARs
    for ar in ar_list:
        pm, flux, intensity = load_ar_data(ar, size, rid_of_top)
        if pm is not None:
            # Scale data (example uses per-AR scaling; consider global scaling)
            pm = min_max_scaling(pm, np.min(pm), np.max(pm))
            flux = min_max_scaling(flux, np.min(flux), np.max(flux))
            intensity = min_max_scaling(intensity, np.min(intensity), np.max(intensity))

            # Combine power maps and intensity
            intensity_reshaped = np.expand_dims(intensity, axis=1)
            combined_inputs = np.concatenate([pm, intensity_reshaped], axis=1)

            all_inputs_list.append(combined_inputs)
            all_flux_list.append(flux)

    if not all_inputs_list:
        print("all_inputs_list does not exist")
        return None, None, 0

    # Create sequences for the LSTM
    x_list, y_list = [], []
    tiles = size**2 - 2 * size * rid_of_top

    for inputs, flux in zip(all_inputs_list, all_flux_list):
        for tile in range(tiles):
            x_seq, y_seq = lstm_ready(tile, size, inputs, flux, num_in, num_pred)
            if x_seq.shape[0] > 0:
                x_seq = torch.reshape(x_seq, (x_seq.shape[0], num_in, x_seq.shape[2]))
                x_list.append(x_seq)
                y_list.append(y_seq)

    if not x_list:
        print("X_list does not exist")
        return None, None, 0

    x_all = torch.cat(x_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    input_feature_size = x_all.shape[2]

    return x_all, y_all, input_feature_size


# --- Model Training & Evaluation ---


def train_epoch(model, dataloader, loss_fn, optimizer, device):
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

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# --- Main Execution ---


def main(config):
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Initialize wandb
    # wandb.init(
    #     project="LSTM,Future_11,NUM_IN_110,pred_12",
    #     entity=os.environ.get("WANDB_ENTITY"),
    #     config=config,

    #     name=f"LSTM_pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_l{config['learning_rate']:.5f}_d{config['dropout']:.2f}",
    #     notes=f"LSTM training with lr={config['learning_rate']}, dropout={config['dropout']}",
    # )

    # --- Data Loading ---
    print("Batch size:", config["batch_size"])
    print("Loading and preparing training data...")
    train_ars = [
        11130,
        11149,
        11158,
        11162,
        11199,
        11327,
        11344,
        11387,
        11393,
        11416,
        11422,
        11455,
        11619,
        11640,
        11660,
        11678,
        11682,
        11765,
        11768,
        11776,
        11916,
        11928,
        12036,
        12051,
        12085,
        12089,
        12144,
        12175,
        12203,
        12257,
        12331,
        12494,
        12659,
        12778,
        12864,
        12877,
        12900,
        12929,
        13004,
        13085,
        13098,
    ]
    x_train, y_train, input_size = prepare_dataset(
        train_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
    )

    print("Loading and preparing test data...")
    test_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
    x_test, y_test, _ = prepare_dataset(
        test_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
    )

    if x_train is None or x_test is None:
        print("Could not create datasets. Exiting.")
        return

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=config["batch_size"], shuffle=False
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
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10)

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss = evaluate_model(model, test_loader, loss_fn, device)

        lr = scheduler.get_last_lr()[0]
        scheduler.step(test_loss)

        # Evaluate every 10 epochs and on the last epoch
        scores = []
        for AR in [11698, 11726, 13165, 13179, 13183]:
            score = eval(device, AR, False, BASE_PATH, model.state_dict(), **config)
            scores.append(score)
        val_rmse = float(np.mean(scores))

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
        }
        print(log_metrics)
        # wandb.log(log_metrics)

    # --- Save Model & Artifacts ---
    model_name = f"pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"
    model_path = os.path.join(RESULTS_PATH, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # model_artifact = wandb.Artifact(
    #     name=f"lstm-model-{wandb.run.id}",
    #     type="model",
    #     description="LSTM Model for SAR emergence prediction",
    #     metadata=config,
    # )
    # model_artifact.add_file(model_path)
    # wandb.log_artifact(model_artifact)

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")
    # wandb.finish()


def main_w_tune(config):
    """Main function to run the experiment."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Initialize wandb
    wandb.init(
        project=f"{model_type},Plateau_percent",
        entity=os.environ.get("WANDB_ENTITY"),
        config=config,
        name=f"{model_type}_pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_l{config['learning_rate']:.5f}_d{config['dropout']:.2f}",
        notes=f"{model_type} training with lr={config['learning_rate']}, dropout={config['dropout']}",
    )

    # --- Data Loading ---
    print("Loading and preparing training data...")
    train_ars = [
        11130,
        11149,
        11158,
        11162,
        11199,
        11327,
        11344,
        11387,
        11393,
        11416,
        11422,
        11455,
        11619,
        11640,
        11660,
        11678,
        11682,
        11765,
        11768,
        11776,
        11916,
        11928,
        12036,
        12051,
        12085,
        12089,
        12144,
        12175,
        12203,
        12257,
        12331,
        12494,
        12659,
        12778,
        12864,
        12877,
        12900,
        12929,
        13004,
        13085,
        13098,
    ]
    x_train, y_train, input_size = prepare_dataset(
        train_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
    )

    print("Loading and preparing test data...")
    test_ars = [11462, 11521, 11907, 12219, 12271, 12275, 12567]
    x_test, y_test, _ = prepare_dataset(
        test_ars,
        9,
        config["rid_of_top"],
        config["num_in"],
        config["num_pred"],
    )

    if x_train is None or x_test is None:
        print("Could not create datasets. Exiting.")
        return

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=config["batch_size"], shuffle=False
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
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.2, patience=10)
    model_name = f"{model_type}pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(config["n_epochs"]):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss = evaluate_model(model, test_loader, loss_fn, device)

        lr = scheduler.get_last_lr()[0]
        scheduler.step(test_loss)

        # Evaluate every 10 epochs and on the last epoch
        scores = []
        for AR in [11698, 11726, 13165, 13179, 13183]:
            score = eval(device, AR, False, BASE_PATH, model.state_dict(), **config)
            scores.append(score)
        val_rmse = float(np.mean(scores))

        log_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "learning_rate": float(lr),
            "RMSE": val_rmse,
        }
        if val_rmse <= 0.07:
            result = BASE_PATH + f"SAR_EMERGENCE_RESEARCH/lstm/results/{val_rmse:.8f}"
            os.makedirs(result, exist_ok=True)  # Ensure the results directory exists
            model_path = os.path.join(result, model_name)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            model_artifact = wandb.Artifact(
                name=f"{model_type}-model-{wandb.run.id}-RMSE-{val_rmse:.8f}",
                type="model",
                description=f"{model_type} Model for SAR emergence prediction",
                metadata=config,
            )
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)
        print(log_metrics)
        wandb.log(log_metrics)
        tune.report(log_metrics)

    # --- Save Model & Artifacts ---
    model_name = f"pred{config['num_pred']}_r{config['rid_of_top']}_i{config['num_in']}_n{config['num_layers']}_h{config['hidden_size']}_e{config['n_epochs']}_lr{config['learning_rate']:.8f}_d{config['dropout']}.pth"
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
    if len(sys.argv) < 9 or len(sys.argv) > 10:
        print(
            "Usage: python train_one_epoch.py <num_pred> <rid_of_top> <num_in> <num_layers> <hidden_size> <n_epochs> <learning_rate> <dropout> <grid_search sample_size>"
        )
        sys.exit(1)

    try:
        if len(sys.argv) == 9:
            config = {
                "num_pred": int(sys.argv[1]),
                "rid_of_top": int(sys.argv[2]),
                "num_in": int(sys.argv[3]),
                "num_layers": int(sys.argv[4]),
                "hidden_size": int(sys.argv[5]),
                "n_epochs": int(sys.argv[6]),
                "learning_rate": float(sys.argv[7]),
                "dropout": float(sys.argv[8]),
                "batch_size": 64,
            }
        else:
            config = {
                "num_pred": int(sys.argv[1]),
                "rid_of_top": int(sys.argv[2]),
                "num_in": int(sys.argv[3]),
                "num_layers": int(sys.argv[4]),
                "hidden_size": int(sys.argv[5]),
                "n_epochs": int(sys.argv[6]),
                "learning_rate": float(sys.argv[7]),
                "dropout": float(sys.argv[8]),
                "sample_size": int(sys.argv[9]),
            }
        return config
    except (ValueError, IndexError) as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For this refactoring to be fully functional, you must provide
    # the implementations for these functions from your 'functions.py' file.
    config = parse_args()
    if "sample_size" in config.keys():
        search_space = {
            "num_pred": tune.choice([12]),
            "rid_of_top": tune.choice([4]),
            "num_in": tune.choice([110]),
            "num_layers": tune.choice([2, 3, 4]),
            "hidden_size": tune.choice([32, 64, 128, 256]),
            "n_epochs": tune.choice([500]),
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "dropout": tune.choice([0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "batch_size": tune.choice([32, 64, 128, 256]),
        }
        algo = OptunaSearch()
        scheduler = ASHAScheduler(max_t=500, grace_period=10, reduction_factor=3)

        custom_stopper = PlateauStopper(
            "RMSE", min_epochs=21, patience=10, min_improvement_percent=0.5
        )

        ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
        tuner = tune.Tuner(  # ③
            tune.with_resources(main_w_tune, {"gpu": 1}),
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
    else:
        main(config)
