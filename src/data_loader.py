import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        df = pd.read_csv(self.file_path, index_col=0)
        return df