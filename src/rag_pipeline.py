import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document   

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-assistant-index")
index = pc.Index(INDEX_NAME)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class PineconeRetriever:
    def __init__(self, index, embedder, text_key="text"):
        self.index = index
        self.embedder = embedder
        self.text_key = text_key

    def get_relevant_documents(self, query, k=5):
        qvec = self.embedder.encode(query).tolist()
        resp = self.index.query(
            vector=qvec,
            top_k=k,
            include_metadata=True
        )

        docs = []
        for m in resp.matches:
            text = m.metadata.get(self.text_key, "")
            meta = dict(m.metadata)
            meta["score"] = m.score  

            docs.append(Document(
                page_content=text,
                metadata=meta
            ))

        return docs

    def similarity_search(self, query, k=5):
        return self.get_relevant_documents(query, k=k)



retriever = PineconeRetriever(index=index, embedder=embed_model)


def rag_search(query, top_k=5):
    docs = retriever.get_relevant_documents(query, k=top_k)
    out = []

    for d in docs:
        out.append({
            "score": d.metadata.get("score"),
            "text": d.page_content,
            "source": d.metadata.get("source", "")
        })

    return out


if __name__ == "__main__":
    q = "What did the Supreme Court direct regarding stray dogs in NCR?"
    res = rag_search(q, top_k=5)

    for r in res:
        print("Score:", r["score"])
        print("Source:", r["source"])
        print("Text:", r["text"][:500])
        print()
