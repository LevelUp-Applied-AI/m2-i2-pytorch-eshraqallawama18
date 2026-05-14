import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import time
import matplotlib.pyplot as plt


# ─── Model ──────────────────────────────────────────────
class HousingModel(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x


# ─── Experiment Function ────────────────────────────────
def run_experiment(config, X_train, X_test, y_train, y_test):

    lr = config["learning_rate"]
    hidden_size = config["hidden_size"]
    epochs = config["epochs"]

    model = HousingModel(hidden_size)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    start_time = time.time()

    # Training
    for epoch in range(epochs):

        predictions = model(X_train)

        loss = criterion(predictions, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    training_time = time.time() - start_time

    # Evaluation
    with torch.no_grad():

        test_predictions = model(X_test)

        test_loss = criterion(
            test_predictions,
            y_test
        ).item()

    preds_np = test_predictions.numpy().flatten()

    actual_np = y_test.numpy().flatten()

    # MAE
    mae = float(np.mean(np.abs(actual_np - preds_np)))

    # R²
    ss_res = float(np.sum((actual_np - preds_np) ** 2))

    ss_tot = float(np.sum(
        (actual_np - np.mean(actual_np)) ** 2
    ))

    r2 = float(1 - (ss_res / ss_tot))

    return {
        "learning_rate": float(lr),
        "hidden_size": int(hidden_size),
        "epochs": int(epochs),
        "train_loss": float(loss.item()),
        "test_loss": float(test_loss),
        "mae": float(mae),
        "r2": float(r2),
        "training_time": float(training_time)
    }


# ─── Main ───────────────────────────────────────────────
def main():

    # 1. Load Data
    df = pd.read_csv("data/housing.csv")

    # 2. Features & Target
    feature_cols = [
        "area_sqm",
        "bedrooms",
        "floor",
        "age_years",
        "distance_to_center_km"
    ]

    X = df[feature_cols]

    y = df[["price_jod"]]

    # 3. Standardization
    X_mean = X.mean()

    X_std = X.std()

    X_scaled = (X - X_mean) / X_std

    # 4. Convert to tensors
    X_tensor = torch.tensor(
        X_scaled.values,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y.values,
        dtype=torch.float32
    )

    print(f"X shape: {X_tensor.shape}")

    print(f"y shape: {y_tensor.shape}")

    # 5. Train/Test Split
    torch.manual_seed(42)

    indices = torch.randperm(len(X_tensor))

    X_shuffled = X_tensor[indices]

    y_shuffled = y_tensor[indices]

    split = int(0.8 * len(X_tensor))

    X_train = X_shuffled[:split]

    X_test = X_shuffled[split:]

    y_train = y_shuffled[:split]

    y_test = y_shuffled[split:]

    # 6. Hyperparameter Grid
    learning_rates = [0.1, 0.01, 0.001]

    hidden_sizes = [16, 32, 64]

    epochs_list = [50, 100, 200, 300]

    results = []

    experiment_num = 1

    total_experiments = (
        len(learning_rates)
        * len(hidden_sizes)
        * len(epochs_list)
    )

    # 7. Experiment Loop
    for lr in learning_rates:

        for hidden_size in hidden_sizes:

            for epochs in epochs_list:

                print(
                    f"\nRunning experiment "
                    f"{experiment_num}/{total_experiments}"
                )

                config = {
                    "learning_rate": lr,
                    "hidden_size": hidden_size,
                    "epochs": epochs
                }

                result = run_experiment(
                    config,
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )

                results.append(result)

                print(result)

                experiment_num += 1

    # 8. Save experiments.json
    with open("experiments.json", "w") as f:

        json.dump(results, f, indent=4)

    print("\nSaved experiments.json")

    # 9. Leaderboard
    sorted_results = sorted(
        results,
        key=lambda x: x["mae"]
    )

    print("\nTOP 10 CONFIGURATIONS\n")

    print(
        "Rank | LR | Hidden | Epochs | "
        "Test MAE | Test R² | Time"
    )

    for i, r in enumerate(sorted_results[:10], start=1):

        print(
            f"{i:2} | "
            f"{r['learning_rate']} | "
            f"{r['hidden_size']} | "
            f"{r['epochs']} | "
            f"{r['mae']:.2f} | "
            f"{r['r2']:.4f} | "
            f"{r['training_time']:.2f}s"
        )

    # 10. Visualization
    lrs = [r["learning_rate"] for r in results]

    maes = [r["mae"] for r in results]

    plt.figure(figsize=(8, 6))

    plt.scatter(lrs, maes)

    plt.xscale("log")

    plt.xlabel("Learning Rate")

    plt.ylabel("Test MAE")

    plt.title("Experiment Results")

    plt.savefig("experiment_summary.png")

    print("\nSaved experiment_summary.png")


# ─── Entry Point ────────────────────────────────────────
if __name__ == "__main__":
    main()