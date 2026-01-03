import pandas as pd
import numpy as np

class DataSplitter: 
    def __init__(self, X: pd.DataFrame, y: pd.Series, 
                 test_size: float = 0.2, 
                 valid_size: float = 0.2, 
                 shuffle: bool = True, 
                 random_state: int = 42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.data_size = self.X.shape[0]

    def split_data(self) -> tuple:
        if self.test_size + self.valid_size >= 1.0:
            raise ValueError("Total sum of test size & valid size must be less than 1.0")


        # Shuffle data
        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(self.data_size)
        
            X_shuffled = self.X.iloc[indices]
            y_shuffled = self.y.iloc[indices]

        else:
            X_shuffled = self.X
            y_shuffled = self.y

        # Split data
        train_size = int(self.data_size * (1 - (self.test_size + self.valid_size)))
        test_size = int(self.data_size * self.test_size)
        valid_size = self.data_size - train_size - test_size 

        X_train = X_shuffled.iloc[:train_size]
        y_train = y_shuffled.iloc[:train_size]

        X_valid = X_shuffled.iloc[train_size : train_size + valid_size]
        y_valid = y_shuffled.iloc[train_size : train_size + valid_size]

        X_test = X_shuffled.iloc[train_size + valid_size:]
        y_test = y_shuffled.iloc[train_size + valid_size:]

        print(f"Train: {X_train.shape[0]} | Valid: {X_valid.shape[0]} | Test: {X_test.shape[0]}")
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test

