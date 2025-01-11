# N-HiTS Time Series Forecasting

A **PyTorch Lightning** implementation of the *N-HiTS* architecture for time series forecasting. This repository provides scripts for data preprocessing, model definition, training, validation, and inference. Adapted N-HiTS uses a stack of multi-block modules with linear upsampling, allowing flexible modeling of time series with various frequencies and horizons.

---

## Table of Contents

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Data Preparation](#data-preparation)  
6. [Model Training & Validation](#model-training--validation)  
7. [Inference](#inference)  
8. [Usage Example](#usage-example)  
9. [Extending to Other Datasets](#extending-to-other-datasets)  


---

## Features

- **Adapted N-HiTS Blocks**  
  Each block handles a portion of the residual series, enabling powerful hierarchical forecasting.  
- **Configurable Architecture**  
  Easily adjust stack types, pooling modes, dropout rates, and more to fit various time series tasks.  
- **PyTorch Lightning Integration**  
  Simplifies training loops, validation steps, checkpointing, and multi-GPU training.  
- **Window-Based Data Preprocessing Scripts**  
  Automatically generate `(X, Y)` pairs from long time series CSVs, splitting into train/val/test.  
- **Additional Data Utilities**  
  Scripts for merging CSV files, converting various dataset formats, and normalizing signals.

---

## Requirements

- Python 3.10+  
- [PyTorch Lightning 2.5.0.post0](https://pytorch-lightning.readthedocs.io/en/stable/)  
- [PyTorch 2.5.1](https://pytorch.org/)  
- [NumPy 2.2.1](https://numpy.org/)  
- [Pandas 2.2.3](https://pandas.pydata.org/)  
- [scikit-learn 1.4.2](https://scikit-learn.org/stable/)  
- [matplotlib 3.8.4](https://matplotlib.org/)

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/aminekhelif/N-HITS.git
   cd NHiTS
    ```
2. **Create a virtual environment (recommended)**
```
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3.	**Install required packages**
```
pip install -r requirements.txt
```
or install the packages individually:
```
pip install torch==2.5.1 pytorch_lightning==2.5.0.post0 numpy==2.2.1 pandas==2.2.3 scikit_learn==1.4.2 matplotlib==3.8.4
```
## Project Structure
```
NHiTS/
├── README.md
├── __init__.py
├── data
│   ├── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   └── dataloader.py
│   ├── eval.py
│   ├── merging.py
│   ├── parsing.py
│   └── preprocess_data.py
├── main.py
├── models
│   ├── __init__.py
│   └── nhits
│       ├── __init__.py
│       └── nhits.py
└── requirements.txt
```
	•	models/nhits: Contains the main AdaptedNHITS class (LightningModule) and NHiTSBlock definitions.
	•	data/: A placeholder directory for raw and processed data.
	•	scripts/: Utility scripts for data handling.
	•	train.py: Illustrates how to set up the training loop with the Trainer.
	•	inference.py: Shows how to load a trained checkpoint and run inference.
	•	test.py: Example script for debugging or integration testing (optional).

# Data Preparation
1.	Have a CSV with time series
>	The first column is assumed to be timestamps (if available), and the remaining columns are numeric signals.
2.	Use the provided preprocessing script
```
python scripts/data_preprocessing.py \
    --file_path daily.csv \
    --output_folder processed_daily \
    --h_values 96 192 336 720 \
    --multiplier 5 \
    --train_split 0.7 \
    --validation_split 0.1
```
This generates windowed (X, Y) pairs for each signal, normalized, and saves them into train.csv, validation.csv, and test.csv .

3.	Adjust parameters
```
	•	--h_values: The forecast horizons to generate data for.
	•	--multiplier: L = multiplier * H.
	•	--train_split & --validation_split: Control the time-based splitting for training, validation, and testing.
```
You can also merge multiple CSVs or convert other datasets using scripts in the scripts/ folder.

## Model Training & Validation

An example using PyTorch Lightning:
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Example dataset from scripts/data_preprocessing.py
from scripts.data_preprocessing import WeatherWindowDataset
from models.nhits.nhits import AdaptedNHITS  # Our LightningModule

# Create PyTorch Datasets and DataLoaders:
train_dataset = WeatherWindowDataset("processed_daily/H=96/train.csv", L=480, H=96)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = WeatherWindowDataset("processed_daily/H=96/validation.csv", L=480, H=96)
val_loader = DataLoader(val_dataset, batch_size=256)

# Initialize the model
model = AdaptedNHITS(
    input_size=480,
    forecast_size=96,
    stack_types=["identity", "identity", "identity"],
    n_blocks=[1,1,1],
    n_layers=[2,2,2],
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

# Define callbacks (checkpointing, early stopping)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="nhits-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    mode="min"
)

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

# Create the Trainer
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",  # or "cpu", "cuda"
    devices=1,
    callbacks=[checkpoint_callback, early_stopping]
)

trainer.fit(model, train_loader, val_loader)
```
The best checkpoint is automatically saved in ./checkpoints.

## Inference

Use the best checkpoint to load the model and run inference on new or test data:
```python
from models.nhits.nhits import AdaptedNHITS
import torch
from torch.utils.data import DataLoader

# Example dataset from scripts/data_preprocessing.py
from scripts.data_preprocessing import WeatherWindowDataset

checkpoint_path = "checkpoints/nhits-epoch=XX-val_loss=YYYY.ckpt"
test_csv = "processed_daily/H=96/test.csv"

model = AdaptedNHITS.load_from_checkpoint(checkpoint_path)

# Prepare the test DataLoader
test_dataset = WeatherWindowDataset(test_csv, L=480, H=96)
test_loader = DataLoader(test_dataset, batch_size=256)

trainer = torch.compile(torch.compile(pl.Trainer(
    accelerator="gpu",
    devices=1
)))

test_results = trainer.test(model, test_loader)
print(test_results)

(Note: torch.compile is optional and requires recent versions of PyTorch for performance tuning.)

Alternatively, you can write a custom inference script (e.g., inference.py) to collect predictions batch-by-batch, compute metrics, and save results.
```
## Usage Example
1.	Data Preprocessing
```
python scripts/data_preprocessing.py \
    --file_path data/daily.csv \
    --output_folder data/processed_daily \
    --h_values 96 \
    --multiplier 5 \
    --train_split 0.7 \
    --validation_split 0.1
```

2.	Train
```
python train.py \
    --train_csv data/processed_daily/H=96/train.csv \
    --val_csv data/processed_daily/H=96/validation.csv \
    --epochs 30 \
    --batch_size 256
```

3.	Inference
```
python inference.py \
    --checkpoint ./checkpoints/nhits-best.ckpt \
    --test_csv data/processed_daily/H=96/test.csv
```
## Extending to Other Datasets
	•	Custom CSVs: Ensure the first column is timestamps (if available), then numeric signals in subsequent columns.
	•	Normalization: The scripts use StandardScaler. Adjust or replace scaling methods as needed.
	•	Window Generation: Modify the logic in preprocess_weather_data if you want overlapping windows with a stride less than H, or to incorporate other custom logic.
	•	Additional Exogenous Variables: Extend the NHiTSBlock class to handle exogenous inputs (e.g., weather data, calendar features).
	•	Multiple Frequencies: The flexible architecture allows modeling multiple sampling rates if you carefully design your windows.
