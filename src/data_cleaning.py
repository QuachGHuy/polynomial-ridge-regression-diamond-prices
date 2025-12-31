import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw_data_dir = os.path.join(project_root, "data", "raw", "diamonds.csv")
cleand_data_dir = os.path.join(project_root, "data", "cleaned", "cleaned_diamonds.csv")

# Load data
df = pd.read_csv(raw_data_dir, index_col=0)

# Remove outliers
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
df = df[(df["depth"]<75)&(df["depth"]>50)]
df = df[(df["table"]<80)&(df["table"]>40)]
df = df[(df["y"]<30)]
df = df[(df["z"]<10)&(df["z"]>2)]

# Drop NaN
df = df.dropna()

# Save to
df.to_csv(cleand_data_dir, index=False)