import os
import sys
import time
import warnings
import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
import ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import re
from eval import eval_tune
from functions import (
    prepare_dataset,
    PlateauStopper,
    DATA_PATH,
    RESULTS_PATH,
    BASE_PATH,
    isVanillaLSTM,
)


if isVanillaLSTM:
    from functions import VanillaLSTM as LSTM

    model_type = "VanillaLSTM"
else:
    from functions import LSTM as LSTM

    model_type = "LSTM"

warnings.filterwarnings("ignore")


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
    stocks = ["AAPL"]
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
