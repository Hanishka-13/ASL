import pandas as pd

df = pd.read_csv("dataset/landmarks.csv")

print("Shape:", df.shape)
print("\nColumn names (first 10):", list(df.columns[:10]))
print("\nFirst row of feature values:")
print(df.iloc[0, :10])

feature_cols = [c for c in df.columns if c not in ("label", "class")]

print("\nOverall min:", df[feature_cols].min().min())
print("Overall max:", df[feature_cols].max().max())
print("Overall mean:", df[feature_cols].mean().mean())