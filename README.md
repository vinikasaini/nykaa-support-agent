# Nykaa Support Agent

A RAG-based customer support agent for answering Nykaa order and support-related queries using a custom knowledge base.

## Features

- Knowledge-base creation and document chunking
- Local HuggingFace embeddings with ChromaDB
- Cosine-similarity based retrieval and fallback
- Grounded answers using retrieved context
- PII masking and prompt-injection guardrails
- LangGraph-based agent workflow
- RAG Triad evaluation on 15 test queries
- FastAPI deployment with `/ask` and `/add-document` endpoints
- JSONL structured logging with trace IDs and timing
- MCP server exposing the order lookup tool
- Separate MCP client for testing order lookups

## Project Structure

```text
nykaa-support-agent/
├── data/
│   ├── knowledge_base/
│   ├── chroma_db/
│   └── orders.csv
├── src/
│   ├── dataset.py
│   ├── knowledge_base.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── threshold_calibration.py
│   ├── rag_pipeline.py
│   ├── evaluate_rag.py
│   ├── agent.py
│   ├── api.py
├── mcp_server.py
|── mcp_client.py
├── tests/
├── docs/
├── requirements.txt
└── README.md

## Tech Stack

Python, LangChain, LangGraph, ChromaDB, HuggingFace Embeddings, FastAPI, Pydantic, FastMCP, JSON Schema

## How It Works

Customer query → Guardrails → Intent Detection → RAG / Order Lookup → Grounded Response → Structured JSON Response

For unsupported questions or insufficiently relevant retrieved context, the agent returns an appropriate fallback instead of generating an unsupported answer.

## Evaluation

The RAG pipeline is evaluated using Context Relevance, Groundedness, and Answer Relevance across 15 queries covering the required knowledge-base topics, including out-of-scope and edge-case queries.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt