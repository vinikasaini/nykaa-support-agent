# nykaa-support-agent

A Retrieval-Augmented Generation (RAG) based customer support agent for answering common Nykaa order and support-related questions using a custom knowledge base.

The project demonstrates dataset generation, knowledge-base creation, document chunking, local embeddings, ChromaDB vector search, similarity-based fallback, grounded generation, and retrieval evaluation.

## Project Overview

The goal of this project is to build a simple customer support agent that can answer questions using information available in its knowledge base.

The system retrieves relevant information from the knowledge base before generating an answer. If the retrieved information is not sufficiently similar to the customer's question, the system returns an "I don't know" response instead of answering without supporting information.

## Project Structure

```text
nykaa-support-agent/
│
├── data/
│   ├── knowledge_base/
│   └── chroma_db/
│
├── src/
│   ├── dataset.py
│   ├── knowledge_base.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── threshold_calibration.py
│   ├── rag_pipeline.py
│   └── evaluate_rag.py
│
├── tests/
│
├── docs/
│
├── .gitignore
├── requirements.txt
└── README.md
