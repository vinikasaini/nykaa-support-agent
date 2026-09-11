from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


TOP_K = 3


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


fixed_store = Chroma(
    collection_name="nykaa_fixed_size",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)


sentence_store = Chroma(
    collection_name="nykaa_sentence",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model
)


queries = [
    "How can I return a product?",
    "How long does a refund take?",
    "How can I track my order?",
    "What should I do if my product is damaged?",
    "Can I cancel my order?"
]


def inspect_collection(store, strategy, query):

    results = store.similarity_search(
        query,
        k=TOP_K
    )

    print("\n" + "-" * 70)
    print("Strategy:", strategy)
    print("Query:", query)

    for rank, document in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print("Source:", document.metadata["source"])
        print("Chunk:")
        print(document.page_content)


for query in queries:

    inspect_collection(
        fixed_store,
        "Fixed-size",
        query
    )

    inspect_collection(
        sentence_store,
        "Sentence-based",
        query
    )