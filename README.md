AI Legal Document Assistant

Retrieval-Augmented Generation (RAG) system using LangChain, Pinecone, WatsonX, FastAPI, and Streamlit.

Overview

This project is an AI-powered legal assistant that allows users to upload court case PDFs and ask questions based strictly on the document content. The system uses a Retrieval-Augmented Generation (RAG) pipeline to ensure responses are grounded in the provided text.

Features

PDF text extraction

Text cleaning and chunking

Embedding generation using Sentence Transformers

Vector storage and retrieval using Pinecone

RAG pipeline management using LangChain

Answer generation using IBM WatsonX LLM

FastAPI backend for processing and retrieval

Streamlit front-end for user interaction

Why We Use LangChain, Pinecone, and WatsonX Together
Pinecone – Vector Storage & Retrieval

Stores all embeddings of document chunks

Performs fast similarity search

Returns top relevant text for answering queries

LangChain – RAG Pipeline Orchestration

Loads and processes PDFs

Splits text into chunks

Generates embeddings

Sends embeddings to Pinecone

Retrieves relevant chunks

Builds the final prompt sent to the LLM

WatsonX – Final Answer Generation

Receives context retrieved from Pinecone

Generates structured, grounded, and accurate answers

Avoids hallucinations by relying on provided context

How They Work Together

LangChain extracts, cleans, and chunks PDF text.

Embeddings are generated using Sentence Transformers.

Pinecone stores and retrieves the embeddings.

LangChain forms the final RAG prompt.

WatsonX generates the final answer.

Tech Stack
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

Project Structure
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

How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

2. Set Environment Variables

Create .env file:

WATSONX_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX=your_index

3. Start FastAPI Backend
uvicorn backend.main:app --reload

4. Start Streamlit Frontend
streamlit run frontend/app.py

Workflow Summary

Upload a legal PDF.

System extracts and cleans text.

Text is chunked and converted to embeddings.

Embeddings stored in Pinecone.

When user asks a question, Pinecone retrieves matching chunks.

LangChain prepares the final prompt.

WatsonX generates a grounded answer.

Challenges Faced

Cleaning broken legal PDF formatting

Managing consistent Pinecone metadata

Fixing FastAPI request routing issues

Connecting Streamlit with backend

Debugging multi-component pipeline

What I Learned

Building end-to-end RAG systems

Using Pinecone for high-quality semantic search

Working with IBM WatsonX APIs

Building API services with FastAPI

Creating simple UI using Streamlit

Debugging multi-tool workflows

Future Improvements

Deploy on cloud using Docker

Add multimodal RAG (OCR, scanned PDFs)

Improve prompt templates

Add multi-document support

References

LangChain Docs

IBM WatsonX.ai Docs

Sentence Transformers Docs

Pinecone Docs