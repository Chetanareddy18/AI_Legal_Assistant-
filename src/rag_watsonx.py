import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from rag_pipeline import retriever
load_dotenv()
creds = Credentials(
    api_key=os.getenv("WATSONX_API_KEY"),
    url=os.getenv("WATSONX_URL")
)
model = Model(
    model_id=os.getenv("WATSONX_MODEL_ID", "mistralai/mistral-medium-2505"),
    credentials=creds,
    project_id=os.getenv("WATSONX_PROJECT_ID")
)
def build_prompt(query, docs):
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = f"""
You are a legal expert AI assistant.
Use ONLY the context below. 
If the answer is not in the context, say:
"The information is not available in the provided documents."
Context:
{context}
Question:
{query}

Answer:
"""
    return prompt
def generate_answer(query, top_k=3):
    docs = retriever.get_relevant_documents(query, k=top_k)
    prompt = build_prompt(query, docs)
    params = {
        GenParams.MAX_NEW_TOKENS: 350,
        GenParams.TEMPERATURE: 0.2
    }
    result = model.generate_text(prompt=prompt, params=params)
    return {
        "answer": result,
        "passages": [
            {
                "text": d.page_content,
                "source": d.metadata.get("source", ""),
                "score": d.metadata.get("score", None)
            }
            for d in docs
        ]
    }
def run_rag(question: str):
    output = generate_answer(question)
    return output["answer"]   

if __name__ == "__main__":
    out = run_rag("What did the court order about stray dogs?")
    print("\nANSWER:\n", out)
