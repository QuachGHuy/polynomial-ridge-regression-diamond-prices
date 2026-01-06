import pandas as pd
import numpy as np


def split_data(
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2, 
        valid_size: float = 0.2, 
        shuffle: bool = True, 
        random_state: int = 42
) -> tuple:

    if test_size + valid_size >= 1.0:
        raise ValueError("Total sum of test size & valid size must be less than 1.0")

    data_size = X.shape[0]

    # Shuffle data
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(data_size)
    
        X_shuffled = X.iloc[indices]
        y_shuffled = y.iloc[indices]

    else:
        X_shuffled = X
        y_shuffled = y

    # Split data
    train_size = int(data_size * (1 - (test_size + valid_size)))
    test_size = int(data_size * test_size)
    valid_size = data_size - train_size - test_size 

    X_train = X_shuffled.iloc[:train_size]
    y_train = y_shuffled.iloc[:train_size]

    X_valid = X_shuffled.iloc[train_size : train_size + valid_size]
    y_valid = y_shuffled.iloc[train_size : train_size + valid_size]

    X_test = X_shuffled.iloc[train_size + valid_size:]
    y_test = y_shuffled.iloc[train_size + valid_size:]

    print(f"Train: {X_train.shape[0]} | Valid: {X_valid.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

