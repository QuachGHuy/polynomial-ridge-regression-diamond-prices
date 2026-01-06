import pandas as pd

def load(file_path: str, index_col: bool = False) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, index_col=index_col)
        
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file")
