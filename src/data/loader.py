import pandas as pd

class DataLoader:
    def __init__(self, file_path: str, index_col=None):
        self.file_path = file_path
        self.index_col = index_col

    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path, index_col=self.index_col)
            
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")

            print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file")
