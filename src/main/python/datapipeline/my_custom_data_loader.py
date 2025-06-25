import os
import pandas as pd
import typing
import datapipeline.base_data_loader as base_data_loader

class MyCustomDataLoader(base_data_loader.BaseDataLoader):
    """This loader loads your custom contrastive pairs DataFrame."""
    def __init__(self, data_path: str, data_file: str = "my_data.parquet", **kwargs):
        super().__init__(data_path, **kwargs)
        self._data_file_path = os.path.join(data_path, data_file)
        
    def load(self) -> pd.DataFrame:
        print(f"Loading data from: {self._data_file_path}")
        if self._data_file_path.endswith(".parquet"):
            df = pd.read_parquet(self._data_file_path)
        elif self._data_file_path.endswith(".csv"):
            df = pd.read_csv(self._data_file_path)
        elif self._data_file_path.endswith(".pickle")
            df = pd.read_pickle(self._data_file_path)
        else:
            raise ValueError("Unsupported file format. Please use .parquet or .csv")
        
        df = df.reset_index(drop=True) # Use the existing 'index' column if available or create one.
        print(f"Loaded DataFrame with {len(df)} rows.")
        return df