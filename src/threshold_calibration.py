import numpy as np

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="nykaa_sentence",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)


in_scope_queries = [
    "Can I return a product?",
    "How long does a refund take?",
    "Can I cancel my order after it has been shipped?",
    "What should I do if my product arrives damaged?",
    "How can I track my order?"
]


out_of_scope_queries = [
    "What is the weather today?",
    "Who is the CEO of Nykaa?"
]


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def measure_similarity(query):
    query_vector = embedding_model.embed_query(query)

    results = vector_store.get(
        include=["documents", "embeddings", "metadatas"]
    )

    best_similarity = -1
    best_document = None

    for document, embedding, metadata in zip(
        results["documents"],
        results["embeddings"],
        results["metadatas"]
    ):
        similarity = cosine_similarity(
            query_vector,
            embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_document = {
                "content": document,
                "metadata": metadata
            }

    return best_similarity, best_document


print("In-scope query results")

in_scope_scores = []

for query in in_scope_queries:

    similarity, document = measure_similarity(query)

    in_scope_scores.append(similarity)

    print("\nQuery:", query)
    print("Top-1 cosine similarity:", round(similarity, 4))
    print("Source:", document["metadata"]["source"])


print("\nOut-of-scope query results")

out_of_scope_scores = []

for query in out_of_scope_queries:

    similarity, document = measure_similarity(query)

    out_of_scope_scores.append(similarity)

    print("\nQuery:", query)
    print("Top-1 cosine similarity:", round(similarity, 4))
    print("Source:", document["metadata"]["source"])


print("\nSummary")

print(
    "In-scope scores:",
    [round(score, 4) for score in in_scope_scores]
)

print(
    "Out-of-scope scores:",
    [round(score, 4) for score in out_of_scope_scores]
)