import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle

import pandas as pd

from src.rag_pipeline import rag_search

query = "What did the court say about stray dogs?"
results = rag_search(query, top_k=5)

df = pd.DataFrame(results)
df.to_csv("rag_results.csv", index=False)
df.to_parquet("rag_results.parquet", index=False)

with open("rag_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Saved successfully!")
