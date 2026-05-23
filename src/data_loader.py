import pandas as pd
import os

def load_raw_data(file_path="data/raw/comments_data.csv"):
    return pd.read_csv(file_path)

def save_processed_data(df, save_path="data/processed/processed_comments.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
