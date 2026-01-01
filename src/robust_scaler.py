import pandas as pd
import numpy as np

class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, data: pd.DataFrame):
        self.median = data.median()

        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)

        self.iqr = q75 - q25
        self.iqr = self.iqr.replace(0, 1)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.median is None or self.iqr is None:
            raise Exception(".fit() must be called before .transform()!")
        
        scaled_data = (data - self.median) / self.iqr
        return scaled_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)