# AI Legal Document Assistant (RAG System)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-0A66C2)

## Project Overview

AI Legal Document Assistant is a Retrieval-Augmented Generation (RAG) application for legal document analysis. Users can upload legal PDFs and ask questions, and the system responds using only retrieved document context to minimize hallucinations.

The project integrates:

- LangChain for orchestration
- Pinecone for vector search
- IBM WatsonX for grounded answer generation
- FastAPI for backend services
- Streamlit for interactive user interface

## Key Features

- Legal PDF text extraction and preprocessing
- Chunking and semantic embedding generation
- Pinecone-based vector storage and retrieval
- Context-grounded answer generation with WatsonX
- FastAPI backend for pipeline endpoints
- Streamlit UI for query interaction

## Why LangChain + Pinecone + WatsonX

### Pinecone (Vector Retrieval)

- Stores chunk embeddings from uploaded documents
- Performs fast semantic similarity search
- Returns top relevant context for each user query

### LangChain (RAG Orchestration)

- Handles document loading and chunking
- Integrates embedding and retrieval steps
- Constructs the final prompt from retrieved context

### WatsonX (Answer Generation)

- Produces final responses from retrieved context
- Improves answer reliability for legal Q and A
- Reduces hallucinations by grounding outputs in source text

## System Workflow

1. User uploads a legal PDF.
2. Text is extracted, cleaned, and chunked.
3. Embeddings are generated and stored in Pinecone.
4. User submits a legal query.
5. Relevant chunks are retrieved via semantic search.
6. LangChain builds a grounded prompt.
7. WatsonX returns the final context-aware response.

## Tech Stack

### Core

- Python
- LangChain
- Pinecone
- IBM WatsonX.ai
- FastAPI
- Streamlit
- Sentence Transformers

### Supporting Libraries

- pdfplumber
- python-dotenv
- uvicorn
- requests

## Project Structure

```text
ai-legal-assistant/
|-- .env
|-- .gitignore
|-- README.md
|-- requirements.txt
`-- src/
	|-- app_fastapi.py
	|-- chunk_and_embed.py
	|-- preprocess_nlu.py
	|-- rag_pipeline.py
	|-- rag_watsonx.py
	|-- save_rag_results.py
	|-- streamlit.py
	|-- streamlit_app.py
	|-- vector_db_build.py
	`-- vector_db_view.py
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Chetanareddy18/AI_Legal_Assistant-.git
cd AI_Legal_Assistant-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
WATSONX_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX=your_index
```

## Run the Application

### Start FastAPI Backend

```bash
uvicorn src.app_fastapi:app --reload
```

### Start Streamlit Frontend

```bash
streamlit run src/streamlit_app.py
```

## Challenges Addressed

- Cleaning noisy and inconsistently formatted legal PDFs
- Preserving reliable metadata for retrieval quality
- Stabilizing FastAPI routing and backend integration
- Coordinating end-to-end RAG behavior across multiple components

## Learning Outcomes

- Building end-to-end production-style RAG workflows
- Practical semantic retrieval with Pinecone
- Integrating IBM WatsonX generation APIs
- Exposing ML pipelines via FastAPI
- Building rapid interactive interfaces in Streamlit

## Future Improvements

- Dockerized deployment and cloud hosting
- OCR support for scanned legal PDFs
- Enhanced prompt templates and retrieval tuning
- Multi-document comparative querying
- User authentication and query history

## References

- LangChain Documentation
- Pinecone Documentation
- IBM WatsonX.ai Documentation
- Sentence Transformers Documentation
