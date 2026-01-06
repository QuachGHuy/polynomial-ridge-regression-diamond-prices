import pandas as pd

class DataCleaner:
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data.copy()

    def clean(self) -> pd.DataFrame:
        print(f"Data size: {self.df.shape[0]} rows x {self.df.shape[1]} cols")
        self._remove_outliers()
        self._drop_missing_rows()
        print(f"Data size: {self.df.shape[0]} rows x {self.df.shape[1]} cols")

        return self.df

    def _remove_outliers(self):
        self.df = self.df[
            (self.df["x"] > 0) & (self.df["y"] > 0) & (self.df["z"] > 0) &
            (self.df["y"] < 30) & 
            (self.df["z"] < 10) & (self.df["z"] > 2) &
            (self.df["depth"].between(50, 75)) &
            (self.df["table"].between(40, 80))
        ]

    def _drop_missing_rows(self):
        self.df.dropna(inplace=True)

    def save(self, file_path: str) -> None:
        self.df.to_csv(file_path, index=False)