# File: models/nhits/nhits_adapted.py

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial
from typing import List, Tuple, Optional

# ------------------------------------------------------------------
# Interpolation helpers (optional if you want to upsample forecast)
# ------------------------------------------------------------------

def upsample_linear(knots: torch.Tensor, out_size: int) -> torch.Tensor:
    """1D linear interpolation from [B, K] -> [B, out_size]."""
    tmp = knots.unsqueeze(1)  # shape (B, 1, K)
    up = F.interpolate(tmp, size=out_size, mode="linear", align_corners=True)
    return up.squeeze(1)      # shape (B, out_size)



# ------------------------------------------------------------------
# One N-HiTS block, operating on (B, L)
# ------------------------------------------------------------------

class NHiTSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_layers: List[int],
        pool_size: int = 1,
        pooling_mode: str = "max",
        dropout_prob: float = 0.0,
        batch_normalization: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.pool_size = pool_size
        self.pooling_mode = pooling_mode

        self.n_theta = input_size + forecast_size
        

        # Optional pooling
        if pool_size > 1:
            if pooling_mode == "max":
                self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)
            else:
                raise ValueError(f"Unsupported pooling_mode '{pooling_mode}'")
        else:
            self.pool = None

        # Build MLP
       
        mlp_input_dim = input_size // pool_size
        

        dims = [mlp_input_dim] + hidden_layers
        mlp_layers = []
        for i in range(len(hidden_layers)):
            mlp_layers.append(nn.Linear(dims[i], dims[i+1]))
            mlp_layers.append(nn.ReLU())
            if batch_normalization:
                mlp_layers.append(nn.BatchNorm1d(dims[i+1]))
            if dropout_prob > 0:
                mlp_layers.append(nn.Dropout(p=dropout_prob))

        # final linear -> n_theta
        forecast_size_tr = forecast_size // pool_size
        #mlp_layers.append(nn.Linear(dims[-1], self.n_theta))
        self.mlp = nn.Sequential(*mlp_layers)
        self.backcast_fc=nn.Linear(dims[-1],mlp_input_dim)
        self.forecast_fc=nn.Linear(dims[-1],forecast_size_tr)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x => shape (B, L)
        Returns (backcast, forecast), each shape (B, L) or (B, H).
        """
        B = x.size(0)
        L = x.size(1)

        if self.pool is not None:
            tmp = x.unsqueeze(1)  # => (B,1,L)
            pooled = self.pool(tmp)  # => (B,1,L//pool_size)
            mlp_input = pooled.squeeze(1)  # => (B, L//pool_size)
        else:
            mlp_input = x  # => (B, L)

        theta = self.mlp(mlp_input)  # => shape (B, n_theta)
        theta_back = self.backcast_fc(theta)
        theta_fore = self.forecast_fc(theta)
        forecast = upsample_linear(theta_fore, self.forecast_size)
        backcast = upsample_linear(theta_back, self.input_size)
        
        return backcast, forecast

# ------------------------------------------------------------------
# The adapted N-HiTS model: multiple blocks stacked in "doubly residual" fashion
# ------------------------------------------------------------------

class AdaptedNHITSModel(nn.Module):
    """
    A multi-block N-HiTS that directly operates on:
      - x (B, L) => (B, H)
    No exogenous or static features.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        stack_types: List[str],           # e.g. ["identity"]
        n_blocks: List[int],             # e.g. [2, 1, ...]
        n_layers: List[int],             # e.g. [2, 2, ...]
        n_theta_hidden: List[List[int]], # e.g. [[256,256], [512,512]]
        n_pool_kernel_size: List[int],
        pooling_mode: str = "max",
        dropout_prob_theta: float = 0.0,
        batch_normalization: bool = False,
        shared_weights: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size

        self.blocks = nn.ModuleList()
        # build blocks for each "stack_type"
        # though we likely only have "identity" here
        for i, stype in enumerate(stack_types):
            for b_id in range(n_blocks[i]):
                if shared_weights and b_id > 0:
                    # reuse last block
                    block = self.blocks[-1]
                else:
                    # # The n_theta is input_size + forecast_knots
                    # but we handle it in the block's basis
                    block = NHiTSBlock(
                        input_size=input_size,
                        forecast_size=forecast_size,
                        hidden_layers=n_theta_hidden[i],
                        
                        pool_size=n_pool_kernel_size[i],
                        pooling_mode=pooling_mode,
                        dropout_prob=dropout_prob_theta,
                        
                        batch_normalization=(batch_normalization if len(self.blocks) == 0 else False),
                    )
                self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (B, L)
        returns => (B, H)
        """

        # residual "doubly"
        # 1) Start with res = x
        # 2) Each block => (backcast, forecast)
        # 3) res = res - backcast
        # 4) accumulate forecast
        res = x
        forecast_sum = torch.zeros(x.size(0), self.forecast_size, device=x.device)

        for block in self.blocks:
            backcast, f_block = block(res)
            res = res - backcast
            forecast_sum = forecast_sum + f_block

        return forecast_sum

# ------------------------------------------------------------------
#  PyTorch Lightning wrapper
# ------------------------------------------------------------------

import pytorch_lightning as pl

class AdaptedNHITS(pl.LightningModule):
    """
    High-level LightningModule that expects:
      - a batch (x, y)
      - x => shape (B, L)
      - y => shape (B, H)
    then trains the multi-block N-HiTS model to minimize an MSE or other loss.
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        stack_types: List[str],
        n_blocks: List[int],
        n_layers: List[int],
        n_theta_hidden: List[List[int]],
        n_pool_kernel_size: List[int],
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        dropout_prob_theta: float = 0.0,
        activation: str = "ReLU",
        batch_normalization: bool = False,
        shared_weights: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        loss_name: str = "MSE",
        random_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.forecast_size = forecast_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self.train_loss_list = []
        self.val_loss_list = []

        # Build NHiTS internal model
        self.model = AdaptedNHITSModel(
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
        )

        # For reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # simple MSE
        self.criterion = nn.MSELoss()
        self.criterion2 = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => shape (B, L)
        returns => shape (B, H)
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch  # x => (B, L), y => (B, H)
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.train_loss_list.append(loss.item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.val_loss_list.append(loss.item())
        
        self.log("val_loss", loss, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        loss2 = self.criterion2(y_hat, y)
        self.log("test_loss_MSE", loss, prog_bar=True)
        self.log("test_loss_MAE", loss2, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
