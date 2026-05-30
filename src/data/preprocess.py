import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Column names
INT_COLS = [f'int_{i}' for i in range(1, 14)]
CAT_COLS = [f'cat_{i}' for i in range(1, 27)]
COLS = ['label'] + INT_COLS + CAT_COLS

def load_data(path: str, nrows: int = None) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, sep='\t', header=None, names=COLS, nrows=nrows)
    print(f"Loaded {len(df)} rows")
    return df

def preprocess(df: pd.DataFrame) -> tuple:
    print("Preprocessing...")

    # Fill missing integers with median
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categoricals with 'unknown'
    for col in CAT_COLS:
        df[col] = df[col].fillna('unknown')

    # Encode categoricals
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('label', axis=1)
    y = df['label']

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data('data/raw/dac/train_sample.txt')
    X_train, X_test, y_train, y_test = preprocess(df)

    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Click rate: {y_train.mean():.3f}")
    print("Done!")
