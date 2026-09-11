from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Load the same embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to our existing ChromaDB
vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)


# Customer question
query = "Can I return a product?"


# Search for the 3 most relevant chunks
results = vector_store.similarity_search(
    query,
    k=3
)


print("\nCustomer Question:")
print(query)

print("\nRelevant Information:")

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {document.metadata['source']}")
    print(document.page_content)