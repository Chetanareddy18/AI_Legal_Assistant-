import json
import os

import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
EMB_DIR = os.path.join(ROOT, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_text(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "text" in data:
        return data["text"]
    if "full_text" in data:
        return data["full_text"]
    return ""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

all_chunks = []
refs = []

for file in os.listdir(RESULTS):
    if file.endswith("_full.json"):
        path = os.path.join(RESULTS, file)
        text = load_text(path)
        if text.strip():
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            refs.extend([file] * len(chunks))

emb = embedder.embed_documents(all_chunks)
emb = np.array(emb)

np.save(os.path.join(EMB_DIR, "chunks.npy"), all_chunks, allow_pickle=True)
np.save(os.path.join(EMB_DIR, "embeddings.npy"), emb)
np.save(os.path.join(EMB_DIR, "refs.npy"), refs, allow_pickle=True)

df = pd.DataFrame({
    "file": refs,
    "chunk": all_chunks,
    "embedding": [",".join(map(str, e)) for e in emb]
})
df.to_csv(os.path.join(EMB_DIR, "embeddings.csv"), index=False)
df.to_parquet(os.path.join(EMB_DIR, "embeddings.parquet"), index=False)

with open(os.path.join(EMB_DIR, "embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, indent=4)

print("Chunking and embedding completed.")
