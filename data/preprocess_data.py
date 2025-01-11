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
    """

    # 1. Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 2. Load the CSV (drop first column if it's timestamps)
    data = pd.read_csv(file_path)
    signals = data.columns[1:]  # skip the first column as timestamp
    data_values = data[signals].values 

    # 3. Normalize
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_values)

    # 4. Splits
    total_points = len(normalized_data)
    train_end = int(total_points * train_split)
    val_end   = int(total_points * (train_split + validation_split))

    train_data = normalized_data[:train_end]
    val_data   = normalized_data[train_end:val_end]
    test_data  = normalized_data[val_end:]

    # Helper function
    def generate_pairs_1d(signal_1d, H, L, stride):
        X_list, Y_list = [], []
        T = len(signal_1d)
        # We need i + L + H <= T
        for i in range(0, T - L - H + 1, stride):
            X_list.append(signal_1d[i : i + L])
            Y_list.append(signal_1d[i + L : i + L + H])
        if len(X_list) == 0:
            return np.array([]), np.array([])  # empty
        return np.array(X_list), np.array(Y_list)

    # 6. Process each horizon
    for H in h_values:
        L = multiplier * H
        stride = H  # step by H
        horizon_folder = os.path.join(output_folder, f"H={H}")
        os.makedirs(horizon_folder, exist_ok=True)

        train_rows, val_rows, test_rows = [], [], []
        D = normalized_data.shape[1]

        for d in range(D):
            train_signal = train_data[:, d]
            val_signal   = val_data[:, d]
            test_signal  = test_data[:, d]

            # Generate
            train_X, train_Y = generate_pairs_1d(train_signal, H, L, stride)
            val_X,   val_Y   = generate_pairs_1d(val_signal,   H, L, stride)
            test_X,  test_Y  = generate_pairs_1d(test_signal,  H, L, stride)

            # If empty, skip
            if train_X.size > 0:
                train_rows.append(pd.DataFrame(np.hstack([train_X, train_Y])))
            else:
                print(f"[Warning] No TRAIN windows for signal={d}, H={H}")

            if val_X.size > 0:
                val_rows.append(pd.DataFrame(np.hstack([val_X, val_Y])))
            else:
                print(f"[Warning] No VAL windows for signal={d}, H={H}")

            if test_X.size > 0:
                test_rows.append(pd.DataFrame(np.hstack([test_X, test_Y])))
            else:
                print(f"[Warning] No TEST windows for signal={d}, H={H}")

        # Combine
        train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        val_df   = pd.concat(val_rows, ignore_index=True)   if val_rows   else pd.DataFrame()
        test_df  = pd.concat(test_rows, ignore_index=True)  if test_rows  else pd.DataFrame()

        # Column names
        columns = [f"X{i}" for i in range(L)] + [f"Y{i}" for i in range(H)]

        # Write only if not empty
        if not train_df.empty and train_df.shape[1] == (L + H):
            train_df.columns = columns
            train_df.to_csv(os.path.join(horizon_folder, "train.csv"), index=False)
        else:
            print(f"[Warning] train.csv is empty for H={H}.")

        if not val_df.empty and val_df.shape[1] == (L + H):
            val_df.columns = columns
            val_df.to_csv(os.path.join(horizon_folder, "validation.csv"), index=False)
        else:
            print(f"[Warning] validation.csv is empty for H={H}.")

        if not test_df.empty and test_df.shape[1] == (L + H):
            test_df.columns = columns
            test_df.to_csv(os.path.join(horizon_folder, "test.csv"), index=False)
        else:
            print(f"[Warning] test.csv is empty for H={H}.")

        print(f"Processed data for H={H} -> {horizon_folder}")

# Example usage
if __name__ == "__main__":
    FILE_PATH = "daily.csv"          
    OUTPUT_FOLDER = "processed_daily"
    H_VALUES = [96, 192, 336, 720]
    MULTIPLIER = 5
    TRAIN_SPLIT = 0.7
    VALIDATION_SPLIT = 0.1

    preprocess_weather_data(
        file_path=FILE_PATH,
        output_folder=OUTPUT_FOLDER,
        h_values=H_VALUES,
        multiplier=MULTIPLIER,
        train_split=TRAIN_SPLIT,
        validation_split=VALIDATION_SPLIT,
    )
