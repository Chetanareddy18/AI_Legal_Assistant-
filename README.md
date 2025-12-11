AI Legal Document Assistant (RAG System)

An AI-powered legal assistant that allows users to upload court case PDFs and ask questions strictly based on the document content.
This project uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Pinecone, WatsonX, FastAPI, and Streamlit to ensure fully grounded, context-based answers.

🚀 Features

PDF text extraction

Text cleaning & chunking

Embedding generation using Sentence Transformers

Vector storage & retrieval using Pinecone

RAG pipeline orchestration using LangChain

Answer generation using IBM WatsonX.ai

FastAPI backend for document processing & retrieval

Streamlit frontend for user interaction

🔗 Why LangChain, Pinecone, and WatsonX Together?
📌 Pinecone – Vector Storage & Retrieval

Stores embeddings of document chunks

Performs fast similarity search

Returns the most relevant context for answering queries

📌 LangChain – RAG Pipeline Orchestration

Loads & processes PDFs

Splits text into chunks

Generates embeddings

Sends embeddings to Pinecone

Retrieves relevant chunks

Builds the final prompt for the LLM

📌 WatsonX – Final Answer Generation

Receives retrieved context

Generates structured, grounded, accurate answers

Minimizes hallucinations by relying only on provided text

⚙️ How the System Works

LangChain extracts, cleans, and chunks PDF text

Sentence Transformers generate embeddings

Pinecone stores and retrieves embeddings

LangChain constructs the RAG prompt

WatsonX produces the final grounded answer

🧩 Tech Stack
Core Components

IBM WatsonX.ai

LangChain

Pinecone

FastAPI

Streamlit

Sentence Transformers

Python

Supporting Libraries

pdfplumber

python-dotenv

uvicorn

requests

os / json

📁 Project Structure
project/
│── backend/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── embeddings.py
│   ├── pinecone_client.py
│   └── watsonx_client.py
│
│── frontend/
│   └── app.py
│
│── data/
│   └── uploaded_pdfs/
│
│── requirements.txt
│── README.md

🛠️ How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

2. Set Environment Variables

Create a .env file in the project root:

WATSONX_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX=your_index

3. Start FastAPI Backend
uvicorn backend.main:app --reload

4. Start Streamlit Frontend
streamlit run frontend/app.py

🔄 Workflow Summary

User uploads a legal PDF

System extracts & cleans text

Text is chunked

Embeddings generated

Embeddings stored in Pinecone

User asks a question

Pinecone retrieves relevant chunks

LangChain forms the final prompt

WatsonX generates a grounded answer

🧠 Challenges Faced

Cleaning poorly formatted legal PDFs

Maintaining consistent Pinecone metadata

Fixing FastAPI routing issues

Connecting Streamlit with backend API

Debugging multi-component RAG pipeline

📚 What I Learned

Building full RAG systems end-to-end

Semantic search using Pinecone

Working with IBM WatsonX APIs

Creating APIs using FastAPI

Developing UI with Streamlit

Debugging multi-tool distributed workflows

🚧 Future Improvements

Deploy using Docker & cloud hosting

Add multimodal RAG (OCR for scanned PDFs)

Improve prompt templates

Add support for multiple documents

User authentication & history

📘 References

LangChain Documentation

IBM WatsonX.ai Documentation

Sentence Transformers Documentation

Pinecone Documentation
