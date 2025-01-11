from torch.utils.data import DataLoader
from models.nhits.nhits import AdaptedNHITS
from models.nhits.nhits import AdaptedNHITSModel
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
from typing import List, Tuple, Optional
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class WeatherWindowDataset():
    def __init__(self, csv_path, L, H):
        df = pd.read_csv(csv_path)
        # each row => [X0..X_{L-1}, Y0..Y_{H-1}]
        self.X_cols = [f"X{i}" for i in range(L)]
        self.Y_cols = [f"Y{i}" for i in range(H)]
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = row[self.X_cols].values.astype("float32")  # shape (L,)
        y = row[self.Y_cols].values.astype("float32")  # shape (H,)
        return torch.tensor(x), torch.tensor(y)


# Suppose L=480, H=96
train_ds = WeatherWindowDataset("train_merged/train_merged.csv", L=480, H=96)

val_ds   = WeatherWindowDataset("validation_merged_final.csv",   L=480, H=96)
test_ds  = WeatherWindowDataset("processed_data2/H=96/test.csv",  L=480, H=96)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)
test_loader  = DataLoader(test_ds,  batch_size=256)

model = AdaptedNHITS(
    input_size=480,
    forecast_size=96,
    stack_types=["identity", "identity", "identity"],
    n_blocks=[1,1,1],         # 2 blocks
    n_layers=[2,2,2],         # each block has 2 linear layers
    n_theta_hidden=[[512,512],[512,512],[512,512]], 
    n_pool_kernel_size=[16,8,1],
    pooling_mode="max",
    dropout_prob_theta=0.1,
    batch_normalization=True,
    shared_weights=False,
    learning_rate=1e-3,
    weight_decay=1e-5,
    loss_name="MSE",
    random_seed=42,
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",             # monitor val_loss from validation_step
    dirpath="./checkpoints",        # directory to save the checkpoints
    filename="nhits-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,                   # keep only the best checkpoint
    mode="min",                     # we want to minimize val_loss
    save_last=True,                 # also save a 'last.ckpt'
)

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",  # same metric
    patience=5,          # how many epochs with no improvement before stopping
    verbose=True,
    mode="min"           # lower val_loss is better
)


trainer = pl.Trainer(max_epochs=30,accelerator="cuda",devices="auto",callbacks=[checkpoint_callback, early_stopping])
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

train_losses = model.train_loss_list
val_losses = model.val_loss_list

val_x = list(range(len(train_losses) // len(val_losses), len(train_losses) + 1, len(train_losses) // len(val_losses)))

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")  
plt.plot(val_x[:len(val_losses)], val_losses, label="Validation Loss", linestyle="--")  
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")
plt.savefig("loss_curves720.png")
plt.show()