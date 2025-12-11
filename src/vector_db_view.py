import os
import csv
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-assistant-index")

ids = [str(i) for i in range(5000)]
batch = 100

rows = []

for i in range(0, len(ids), batch):
    batch_ids = ids[i:i+batch]
    result = index.fetch(ids=batch_ids)

    for vid, vector in result.vectors.items():
        rows.append([
            vid,
            vector.metadata.get("text", ""),
            vector.metadata.get("source", ""),
            vector.values[:10]
        ])

with open("vector_db_preview.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "text", "source", "embedding_preview"])
    writer.writerows(rows)

print("Exported vector_db_preview.csv")
