import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame) -> None:
        self.df = data.copy()
        self.categorical_features = ["cut", "color", "clarity"]
        self.log_features = ["price", "carat"]

        # Categorical mapping dict
        self.mappings = {
            "cut": {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5},
            "color": {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7},
            "clarity": {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
        }

        # Interaction term
        self.inter_term = {
            "primary": "carat",
            "features": ["cut", "color", "clarity", "density"],
            "degree": 3
        }

    def transform(self) -> pd.DataFrame:
        print(f"DataFrame: {self.df.shape[0]} rows x {self.df.shape[1]} cols \n")
        print(f"List cols: {self.df.columns}")

        self._ordinal_encoding()
        self._create_geometry_features()
        self._log_transform()
        self._add_interaction_term()
        
        print("Feature Engineering completed")
        print(f"DataFrame: {self.df.shape[0]} rows x {self.df.shape[1]} cols \n")
        print(f"List cols: {self.df.columns}")
        
        return self.df

    def _ordinal_encoding(self):
        for feature in self.categorical_features:
            if feature in self.mappings:
                self.df[feature] = self.df[feature].map(self.mappings[feature])
    
    def _create_geometry_features(self):
        self.df["volume"] = self.df["x"] * self.df["y"] * self.df["z"]
        self.df["density"] = self.df["carat"] / self.df["volume"]

        self.df.drop(columns=["x", "y", "z", "volume"], inplace=True)
    
    def _log_transform(self):
        for feature in self.log_features:
            if feature in self.df.columns:
                self.df[feature] = np.log1p(self.df[feature])

    def _add_interaction_term(self):
        primary_col = self.inter_term["primary"]
        tar_col = self.inter_term["features"]
        degree = self.inter_term["degree"]

        if primary_col not in self.df.columns:
            return
        
        # For Linear Interaction
        for inter in tar_col:
            if inter in self.df.columns and inter != primary_col:
                self.df[f"{primary_col}_x_{inter}"] = self.df[primary_col] * self.df[inter]

        # For Polynomial Interaction
        for i in range(2, degree + 1):
            poly_name = f"{primary_col}_{i}"
            self.df[poly_name] = self.df[primary_col] ** i
            
            for inter in tar_col:
                if inter in self.df.columns and inter != primary_col:
                    self.df[f"{poly_name}_x_{inter}"] = self.df[poly_name] * self.df[inter]
                
                
    
    
   