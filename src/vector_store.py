from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from knowledge_base import (
    load_documents,
    split_documents,
    split_documents_by_sentence
)


#  LOAD DOCUMENTS
documents = load_documents()

print(f"Loaded {len(documents)} documents")


#  CREATE BOTH CHUNKING STRATEGIES
fixed_chunks = split_documents(documents)

sentence_chunks = split_documents_by_sentence(documents)

print(f"Fixed-size chunks: {len(fixed_chunks)}")
print(f"Sentence-based chunks: {len(sentence_chunks)}")


#  CREATE EMBEDDING MODEL

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. PREPARE FIXED-SIZE DATA
fixed_texts = [
    chunk["content"]
    for chunk in fixed_chunks
]

fixed_metadatas = [
    {
        "source": chunk["filename"],
        "chunking_strategy": "fixed_size"
    }
    for chunk in fixed_chunks
]


#  CREATE FIXED-SIZE CHROMA COLLECTION
fixed_vector_store = Chroma.from_texts(
    texts=fixed_texts,
    embedding=embedding_model,
    metadatas=fixed_metadatas,
    collection_name="nykaa_fixed_size",
    persist_directory="data/chroma_db"
)

print(
    f"\nStored {len(fixed_texts)} chunks "
    "in fixed-size ChromaDB collection"
)


#  PREPARE SENTENCE-BASED DATA
sentence_texts = [
    chunk["content"]
    for chunk in sentence_chunks
]

sentence_metadatas = [
    {
        "source": chunk["filename"],
        "chunking_strategy": "sentence"
    }
    for chunk in sentence_chunks
]


#  CREATE SENTENCE-BASED CHROMA COLLECTION
sentence_vector_store = Chroma.from_texts(
    texts=sentence_texts,
    embedding=embedding_model,
    metadatas=sentence_metadatas,
    collection_name="nykaa_sentence",
    persist_directory="data/chroma_db"
)

print(
    f"Stored {len(sentence_texts)} chunks "
    "in sentence-based ChromaDB collection"
)


print("VECTOR DATABASE CREATION COMPLETE")

print("Collection 1: nykaa_fixed_size")
print("Collection 2: nykaa_sentence")