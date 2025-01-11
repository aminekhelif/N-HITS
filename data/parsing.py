import os
import pandas as pd
import numpy as np
from typing import List, Dict

class DatasetConverter:
    """
    A robust class to parse different datasets (LD2011, traffic, Jena climate, ILI files, exchange rates)
    and then convert them to CSV with all numeric columns except 'timestamp'.
    """
    
    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Orchestrates the correct loader by inspecting the filename.
        Returns a DataFrame with 'timestamp' + numeric columns.
        """
        fname = os.path.basename(path).lower()
        
        # Decide which specialized loader to call
        if "ld2011" in fname or "ld2011_2014" in fname:
            return self._load_ld2011(path)
        elif "traffic" in fname and fname.endswith(".tsf"):
            return self._load_traffic_file(path)
        elif "jena" in fname:
            return self._load_jena_climate(path)
        elif "ili" in fname or "ilin" in fname or "who_nrevss" in fname:
            return self._load_ili_file(path)
        elif "exchange_rate_to" in fname or "exchange_rate_usd_to" in fname:
            return self._load_exchange_rate_usd_csv(path)
        elif "exchange-rates-main" in path.replace("\\","/") and any(x in fname for x in ["annual", "daily", "monthly"]):
            # annual.csv, daily.csv, monthly.csv have the same format: "Date,Country,Exchange rate"
            return self._load_exchange_rates_main_file(path)
        elif "yearly.csv" in fname:
            # yearly has "Date,Country,Value"
            return self._load_exchange_rates_yearly(path)
        else:
            # fallback
            return self._load_generic_csv(path)

    def convert_to_csv(self, input_path: str, output_path: str):
        """
        1) Load the dataset from input_path
        2) Ensure columns except 'timestamp' are numeric
        3) Save the final DataFrame as a CSV to output_path
        """
        df = self.load_dataset(input_path)
        
        # Convert all non-timestamp columns to numeric (coerce invalid values to NaN)
        numeric_cols = [c for c in df.columns if c != 'timestamp']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Finally, save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}, shape={df.shape}")



    # Specila loaders for this file
    def _load_ld2011(self, path: str) -> pd.DataFrame:
        """
        Loads 'LD2011_2014.txt' or similar:
            "";"MT_001";"MT_002";"MT_003"...
            "2011-01-01 00:15:00";0;0;0 ...
        Uses sep=';' and quotechar='"'.
        """
        df = pd.read_csv(
            path, 
            sep=';', 
            decimal='.',  # If decimal is comma, change to decimal=','
            quotechar='"', 
            engine='python', 
            header=0
        )
        
        # The first column is the timestamp (often unnamed or "")
        first_col = df.columns[0]
        df.rename(columns={first_col: 'timestamp'}, inplace=True)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Keep only numeric + timestamp
        numeric_cols = [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]
        df = df[['timestamp'] + numeric_cols].copy()
        
        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    def _load_traffic_file(self, path: str) -> pd.DataFrame:
        """
        Reads 'traffic_hourly_dataset.tsf' or similar.
        This is placeholder logic; adjust for your actual .tsf format.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Filter lines
        data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('@')]
        
        # Example parse: if lines are "some_label:YYYY-MM-DD HH-MM-SS:value1,value2,..."
        timestamps = []
        all_values = []
        
        for line in data_lines:
            parts = line.split(':')
            if len(parts) >= 3:
                # e.g., parts[1] = date/time, parts[2] = comma-sep values
                ts_str = parts[1].strip()
                # do any needed replacements to parse the time
                # ex: "2015-01-01 00-00-01" -> "2015-01-01 00:00:01"
                ts_str = ts_str.replace('-', ':', 2)
                ts_str = ts_str.replace(' ', 'T')  # just an example if needed
                values_str = parts[2]
                values_list = values_str.split(',')
                
                timestamps.append(ts_str)
                all_values.append(values_list)
        
        df = pd.DataFrame(all_values)
        
        # Insert the timestamp as a new column
        df['timestamp'] = pd.to_datetime(timestamps, errors='coerce', exact=False)
        
        # Rename sensor columns
        sensor_cols = [f'sensor_{i}' for i in range(df.shape[1] - 1)]
        rename_map = dict(zip(df.columns[:-1], sensor_cols))
        df.rename(columns=rename_map, inplace=True)
        
        # Reorder so 'timestamp' is first
        df = df[['timestamp'] + sensor_cols]
        
        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _load_jena_climate(self, path: str) -> pd.DataFrame:
        """
        Jena climate dataset with columns like "Date Time","p (mbar)","T (degC)"...
        Dates in format DD.MM.YYYY HH:MM:SS
        """
        df = pd.read_csv(path, quotechar='"')
        
        if 'Date Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date Time'], errors='coerce', dayfirst=True)
        else:
            if len(df) < 60000:
                df['timestamp'] = pd.date_range('2009-01-01', periods=len(df), freq='10min')
            else:
                df['timestamp'] = range(len(df))
        
        numeric_cols = [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]
        df = df[['timestamp'] + numeric_cols].copy()
        
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _load_ili_file(self, path: str) -> pd.DataFrame:
        """
        For ILI / WHO_NREVSS data with columns:
          REGION TYPE,REGION,YEAR,WEEK,...
        We'll parse weekly dates from YEAR+WEEK if no direct date col.
        """
        df = pd.read_csv(path)
        
        # Potential date columns
        possible_date_cols = [c for c in df.columns if 'date' in c.lower() or 'week_end' in c.lower()]
        if possible_date_cols:
            df['timestamp'] = pd.to_datetime(df[possible_date_cols[0]], errors='coerce')
        else:
            # Build date from YEAR/WEEK
            if 'YEAR' in df.columns and 'WEEK' in df.columns:
                def year_week_to_date(row):
                    try:
                        return pd.to_datetime(f"{int(row['YEAR'])}-W{int(row['WEEK']):02}-1", format="%G-W%V-%u")
                    except:
                        return pd.NaT
                df['timestamp'] = df.apply(year_week_to_date, axis=1)
        
        # If still no valid timestamp, fallback
        if 'timestamp' not in df.columns or df['timestamp'].isnull().all():
            if len(df) < 60000:
                df['timestamp'] = pd.date_range('2002-01-01', periods=len(df), freq='W')
            else:
                df['timestamp'] = range(len(df))
        
        numeric_cols = [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]
        df = df[['timestamp'] + numeric_cols].copy()
        
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _load_exchange_rate_usd_csv(self, path: str) -> pd.DataFrame:
        """
        For 'exchange_rate_to_usd.csv' or 'exchange_rate_usd_to.csv':
        has a 'date' column + many currency columns.
        """
        df = pd.read_csv(path)
        
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['timestamp'] = range(len(df))
        
        numeric_cols = [c for c in df.columns if c not in ['date','timestamp'] and pd.api.types.is_numeric_dtype(df[c])]
        df = df[['timestamp'] + numeric_cols].copy()
        
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _load_exchange_rates_main_file(self, path: str) -> pd.DataFrame:
        """
        For annual.csv, daily.csv, monthly.csv:
        Each has: Date,Country,Exchange rate
        We'll pivot to wide format: columns = unique Countries, values = Exchange rate
        """
        df = pd.read_csv(path)
        
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df['timestamp'] = range(len(df))
        
        # Pivot if columns exist
        if {'Date', 'Country', 'Exchange rate'}.issubset(df.columns):
            pivoted = df.pivot(index='Date', columns='Country', values='Exchange rate')
            pivoted.reset_index(inplace=True)
            pivoted.rename(columns={'Date': 'timestamp'}, inplace=True)
            pivoted.sort_values('timestamp', inplace=True)
            pivoted.reset_index(drop=True, inplace=True)
            return pivoted
        else:
            numeric_cols = [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]
            df = df[['timestamp'] + numeric_cols].copy()
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    
    def _load_exchange_rates_yearly(self, path: str) -> pd.DataFrame:
        """
        For yearly.csv with columns: Date,Country,Value
        We'll pivot similarly.
        """
        df = pd.read_csv(path)
        
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df['timestamp'] = range(len(df))
        
        # Pivot if columns exist
        if {'Date','Country','Value'}.issubset(df.columns):
            pivoted = df.pivot(index='Date', columns='Country', values='Value')
            pivoted.reset_index(inplace=True)
            pivoted.rename(columns={'Date': 'timestamp'}, inplace=True)
            pivoted.sort_values('timestamp', inplace=True)
            pivoted.reset_index(drop=True, inplace=True)
            return pivoted
        else:
            numeric_cols = [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]
            df = df[['timestamp'] + numeric_cols].copy()
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    
    def _load_generic_csv(self, path: str) -> pd.DataFrame:
        """
        Fallback loader if we cannot detect the dataset.
        We'll parse any date/time column or create a dummy index.
        """
        df = pd.read_csv(path)
        
        possible_date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if possible_date_cols:
            df['timestamp'] = pd.to_datetime(df[possible_date_cols[0]], errors='coerce')
        
        if 'timestamp' not in df.columns or df['timestamp'].isnull().all():
            if len(df) < 60000:
                df['timestamp'] = pd.date_range('1900-01-01', periods=len(df), freq='D')
            else:
                df['timestamp'] = range(len(df))
        
        numeric_cols = [c for c in df.columns if c != 'timestamp' and pd.api.types.is_numeric_dtype(df[c])]
        df = df[['timestamp'] + numeric_cols].copy()
        
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df


def main():
    converter = DatasetConverter()
    

    dataset_paths = ["/Data/exchange-rates-main/data/annual.csv",
                    "/Data/exchange-rates-main/data/daily.csv",
                    "/Data/exchange-rates-main/data/monthly.csv",
                    "/Data/exchange-rates-main/data/yearly.csv",  
                    "/Data/Exchange/exchange_rate_to_usd.csv",
                    "/Data/Exchange/exchange_rate_usd_to.csv",
                    "/Data/ILI/ILINet.csv",
                    "/Data/ILI/WHO_NREVSS_Clinical_Labs.csv",
                    "/Data/ILI/WHO_NREVSS_Combined_prior_to_2015_16.csv",
                    "/Data/ILI/WHO_NREVSS_Public_Health_Labs.csv",
                    "/Data/jena_climate_2009_2016.csv",
                    "/Data/LD2011_2014.txt",
                    "/Data/traffic_hourly_dataset.tsf",
]
    output_dir = "converted"
    os.makedirs(output_dir, exist_ok=True)
    for path in dataset_paths:
        file_name = path.split("/")[-1].split(".")[0]
        output_path = os.path.join(output_dir, file_name + "_converted.csv")
        
        print(f"Converting {path} -> {output_path}")
        converter.convert_to_csv(path, output_path)

if __name__=='__main__':
    main()