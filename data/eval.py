# File: test.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from models.nhits.nhits import AdaptedNHITS  

class InferenceDataset(Dataset):
    def __init__(self, csv_path: str, input_size: int, forecast_size: int, device: torch.device):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.X_cols = [f"X{i}" for i in range(input_size)]
        self.Y_cols = [f"Y{i}" for i in range(forecast_size)]
        self.data = df
        self.device = device  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = row[self.X_cols].values.astype("float32")  
        y = row[self.Y_cols].values.astype("float32") 
        return torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)


def run_inference(
    checkpoint_path: str,
    csv_path: str,
    input_size: int,
    forecast_size: int,
    stack_types,
    n_blocks,
    n_layers,
    n_theta_hidden,
    n_pool_kernel_size,
    pooling_mode="max",
    dropout_prob_theta=0.1,
    batch_normalization=True,
    shared_weights=False,
    learning_rate=1e-3,
    weight_decay=1e-5,
    loss_name="MSE",
    random_seed=42,
    batch_size=64,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model = AdaptedNHITS.load_from_checkpoint(
        checkpoint_path,
        input_size=input_size,
        forecast_size=forecast_size,
        stack_types=stack_types,
        n_blocks=n_blocks,
        n_layers=n_layers,
        n_theta_hidden=n_theta_hidden,
        n_pool_kernel_size=n_pool_kernel_size,
        pooling_mode=pooling_mode,
        dropout_prob_theta=dropout_prob_theta,
        batch_normalization=batch_normalization,
        shared_weights=shared_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss_name=loss_name,
        random_seed=random_seed,
    ).to(device)
    model.eval()

    infer_ds = InferenceDataset(csv_path, input_size, forecast_size, device)
    infer_loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False)

    mse_criterion = torch.nn.MSELoss()
    mae_criterion = torch.nn.L1Loss()

    preds = []
    total_mse, total_mae = 0, 0
    num_batches = 0

    for batch_x, batch_y in infer_loader:
        with torch.no_grad():
            batch_y_hat = model(batch_x)
            preds.append(batch_y_hat.cpu().numpy())
            
            # Calculate errors
            mse = mse_criterion(batch_y_hat, batch_y).item()
            mae = mae_criterion(batch_y_hat, batch_y).item()
            total_mse += mse
            total_mae += mae
            num_batches += 1

            print(f"Batch {num_batches}: MSE = {mse:.4f}, MAE = {mae:.4f}")

    # Aggregate results
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    print(f"Overall: MSE = {avg_mse:.4f}, MAE = {avg_mae:.4f}")

    preds = np.concatenate(preds, axis=0)
    return preds, avg_mse, avg_mae


if __name__ == "__main__":
    checkpoint_path = "ckpt/h720.ckpt"
    inference_csv = "test_dataset/processed_LD2011_2014_trimmed/H=720/test.csv"
    input_size = 3600
    forecast_size = 720
    stack_types = ["identity", "identity", "identity"]
    n_blocks = [1, 1, 1]
    n_layers = [2, 2, 2]
    n_theta_hidden = [[512, 512], [512, 512], [512, 512]]
    n_pool_kernel_size = [16, 8, 1]

    predictions, avg_mse, avg_mae = run_inference(
        checkpoint_path=checkpoint_path,
        csv_path=inference_csv,
        input_size=input_size,
        forecast_size=forecast_size,
        stack_types=stack_types,
        n_blocks=n_blocks,
        n_layers=n_layers,
        n_theta_hidden=n_theta_hidden,
        n_pool_kernel_size=n_pool_kernel_size,
        pooling_mode="max",
        dropout_prob_theta=0.1,
        batch_normalization=True,
        shared_weights=False,
        learning_rate=1e-3,
        weight_decay=1e-5,
        loss_name="MSE",
        random_seed=42,
        batch_size=64,
    )

    print("Done with inference!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}")
