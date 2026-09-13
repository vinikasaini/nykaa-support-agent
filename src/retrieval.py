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


def retrieve_context(query, k=3):
    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


if __name__ == "__main__":

    query = "Can I return a product?"

    results = retrieve_context(query)

    print("\nCustomer Question:")
    print(query)

    print("\nRelevant Information:")

    for i, document in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print(
            f"Source: {document.metadata['source']}"
        )

        print(document.page_content)