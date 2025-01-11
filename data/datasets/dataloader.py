import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_weather_data(file_path, 
                            output_folder, 
                            h_values, 
                            multiplier, 
                            train_split, 
                            validation_split):
    """
    Preprocess weather data into windows of length L (= multiplier * H) 
    plus an output horizon of H, for each signal individually.
    
    Each row in the final CSV corresponds to one window from a single signal:
      [ x_{t}, ..., x_{t+L-1}, y_{t+L}, ..., y_{t+L+H-1} ]
    where L = multiplier*H.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the data.
        The first column is assumed to be timestamps, and the rest are signals.
    output_folder : str
        Folder where the processed data will be stored.
    h_values : list
        A list of horizon sizes, e.g. [96, 192, 336, 720].
    multiplier : int
        L = multiplier * H.
    train_split : float
        Fraction of data (in time) for training (e.g. 0.7).
    validation_split : float
        Fraction of data (in time) for validation (e.g. 0.1).
        Remaining fraction goes to test.
    """

    # 1. Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 2. Load the CSV (drop first column if it's timestamps)
    data = pd.read_csv(file_path)
    signals = data.columns[1:]           # Skip the timestamp column
    data_values = data[signals].values   # shape: (T, D) for T timesteps, D signals

    # 3. Normalize each signal across the entire dataset
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_values)  # shape: (T, D)

    # 4. Time-based splitting into train, val, test
    total_points = len(normalized_data)
    train_end = int(total_points * train_split)
    val_end   = int(total_points * (train_split + validation_split))

    train_data = normalized_data[:train_end]    # shape: (train_end, D)
    val_data   = normalized_data[train_end:val_end]
    test_data  = normalized_data[val_end:]

    # 5. Helper function to create (X, Y) for a 1D signal
    #    X: shape (num_windows, L)
    #    Y: shape (num_windows, H)
    def generate_pairs_1d(signal_1d, H, L, stride):
        """
        signal_1d: (T,) array for one signal
        H: forecast horizon
        L: number of past points
        stride: how many points to jump each step (usually H)
        """
        X_list, Y_list = [], []
        T = len(signal_1d)
        # We need i + L + H <= T
        for i in range(0, T - L - H + 1, stride):
            X_list.append(signal_1d[i : i + L])        # Past L points
            Y_list.append(signal_1d[i + L : i + L + H])# Next H points
        return np.array(X_list), np.array(Y_list)

    # 6. Loop over each horizon H
    for H in h_values:
        L = multiplier * H
        stride = H  # Typically, we step by H to avoid overlap

        # Create a folder for this horizon
        horizon_folder = os.path.join(output_folder, f"H={H}")
        os.makedirs(horizon_folder, exist_ok=True)

        # We'll collect windows (rows) from each signal, then concatenate
        train_rows = []
        val_rows   = []
        test_rows  = []

        # 7. Process each signal individually
        #    Each row => single signal, single window
        D = normalized_data.shape[1]  # number of signals
        for d in range(D):
            # Extract the d-th signal from each split
            train_signal = train_data[:, d]  # shape: (train_end,)
            val_signal   = val_data[:, d]
            test_signal  = test_data[:, d]

            # Generate (X, Y) windows
            train_X, train_Y = generate_pairs_1d(train_signal, H, L, stride)
            val_X,   val_Y   = generate_pairs_1d(val_signal,   H, L, stride)
            test_X,  test_Y  = generate_pairs_1d(test_signal,  H, L, stride)

            # Concatenate horizontally: (N, L + H)
            # Each row: [X0,...,X_{L-1}, Y0,...,Y_{H-1}]
            train_combined = np.hstack([train_X, train_Y])  # shape: (num_windows, L+H)
            val_combined   = np.hstack([val_X,   val_Y])
            test_combined  = np.hstack([test_X,  test_Y])

            # Convert to DataFrame so we can concatenate with others
            train_rows.append(pd.DataFrame(train_combined))
            val_rows.append(pd.DataFrame(val_combined))
            test_rows.append(pd.DataFrame(test_combined))

        # Combine all signals’ windows (row-wise)
        # Now each row is a single time-series window from some signal
        train_df = pd.concat(train_rows, ignore_index=True)
        val_df   = pd.concat(val_rows,   ignore_index=True)
        test_df  = pd.concat(test_rows,  ignore_index=True)

        # Assign column names: X0..X(L-1), Y0..Y(H-1)
        columns = [f"X{i}" for i in range(L)] + [f"Y{i}" for i in range(H)]
        train_df.columns = columns
        val_df.columns   = columns
        test_df.columns  = columns

        # 8. Save to CSV
        train_df.to_csv(os.path.join(horizon_folder, "train.csv"), index=False)
        val_df.to_csv(os.path.join(horizon_folder,   "validation.csv"), index=False)
        test_df.to_csv(os.path.join(horizon_folder,  "test.csv"), index=False)

        print(f"Processed data for H={H} -> {horizon_folder}")
