import numpy as np

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


THRESHOLD = 0.465
TOP_K = 3
MOCK_LLM = True


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="nykaa_sentence",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def retrieve_context(query):
    query_vector = embedding_model.embed_query(query)

    results = vector_store.get(
        include=["documents", "embeddings", "metadatas"]
    )

    scored_documents = []

    for document, embedding, metadata in zip(
        results["documents"],
        results["embeddings"],
        results["metadatas"]
    ):
        similarity = cosine_similarity(
            query_vector,
            embedding
        )

        scored_documents.append({
            "content": document,
            "source": metadata["source"],
            "similarity": similarity
        })

    scored_documents.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return scored_documents[:TOP_K]


def mock_llm(query, retrieved_chunks):
    context = "\n\n".join(
        chunk["content"]
        for chunk in retrieved_chunks
    )

    answer = (
        "MOCK_LLM ANSWER\n\n"
        f"Question: {query}\n\n"
        "Answer based only on retrieved context:\n"
        f"{context}"
    )

    return answer


def answer_query(query):

    retrieved_chunks = retrieve_context(query)

    top_similarity = retrieved_chunks[0]["similarity"]

    print("\nCustomer Question:")
    print(query)

    print("\nTop-1 Cosine Similarity:")
    print(round(top_similarity, 4))

    print("\nRetrieved Context:")

    for index, chunk in enumerate(retrieved_chunks, start=1):

        print(f"\n--- Result {index} ---")
        print("Source:", chunk["source"])
        print(
            "Similarity:",
            round(chunk["similarity"], 4)
        )
        print(chunk["content"])

    if top_similarity < THRESHOLD:

        print("\nFinal Answer:")
        print(
            "I don't know. "
            "I could not find sufficiently relevant information "
            "in the knowledge base."
        )

        return

    if MOCK_LLM:

        answer = mock_llm(
            query,
            retrieved_chunks
        )

        print("\nFinal Answer:")
        print(answer)


query = input("\nEnter your question: ")

answer_query(query)