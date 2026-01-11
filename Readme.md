# WebRAG FastAPI

A Web-based Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, Chroma, and Groq.

## Features
- Load any website URL
- Store content in a vector database
- Ask questions grounded in the website
- Chat history per user

## Run
```bash
pip install -r requirements.txt
uvicorn api_endpoints:app --reload
```

## Endpoints
- POST /load-url
- POST /ask
