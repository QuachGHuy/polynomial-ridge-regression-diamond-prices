import pandas as pd

class RobustScaler:
    def __init__(self, features: list[str]):
        self.features = features
        self.median = None
        self.iqr = None

    def fit(self, df: pd.DataFrame):
        self.median = df[self.features].median()

        q25 = df[self.features].quantile(0.25)
        q75 = df[self.features].quantile(0.75)

        self.iqr = (q75 - q25).replace(0, 1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.median is None or self.iqr is None:
            raise RuntimeError("fit() must be called before transform()")

        result = df.copy()

        result[self.features] = (
            result[self.features] - self.median
        ) / self.iqr

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
