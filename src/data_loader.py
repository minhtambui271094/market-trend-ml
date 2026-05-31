import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)

    df.columns = [c.lower() for c in df.columns]

    df["time"] = pd.to_datetime(df["time"])

    return df