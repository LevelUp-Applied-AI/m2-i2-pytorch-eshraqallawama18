"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn


def main():
    # ─── 1. Load Data ─────────────────────────────────────
    df = pd.read_csv('data/housing.csv')

    # ─── 2. Features & Target ─────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # ─── 3. Standardization ───────────────────────────────
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    # ─── 4. Convert to tensors ────────────────────────────
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    print(f"X shape: {X_tensor.shape}")
    print(f"y shape: {y_tensor.shape}")

    # ─── 5. Define Model ──────────────────────────────────
    class HousingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(5, 32)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(32, 1)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    # ─── 6. Setup training ────────────────────────────────
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ─── 7. Training Loop ─────────────────────────────────
    num_epochs = 100

    for epoch in range(num_epochs):
        # Forward
        predictions = model(X_tensor)

        # Loss
        loss = criterion(predictions, y_tensor)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ─── 8. Save predictions ──────────────────────────────
    with torch.no_grad():
        predictions_tensor = model(X_tensor)

    predictions_np = predictions_tensor.numpy().flatten()
    actuals_np = y_tensor.numpy().flatten()

    results_df = pd.DataFrame({
        'actual': actuals_np,
        'predicted': predictions_np
    })

    results_df.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv")


# ─── Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    main()