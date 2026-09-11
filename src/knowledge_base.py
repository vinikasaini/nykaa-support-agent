from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base")


def load_documents():
    documents = []

    for file_path in KNOWLEDGE_BASE_PATH.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "filename": file_path.name,
            "content": text
        })

    return documents


# Load all 12 documents
documents = load_documents()

print(f"Loaded {len(documents)} documents")

for document in documents:
    print(document["filename"])



#  FIXED-SIZE CHUNKING
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:

        document_chunks = splitter.split_text(
            document["content"]
        )

        for chunk in document_chunks:

            chunks.append({
                "filename": document["filename"],
                "content": chunk
            })

    return chunks


#  SENTENCE-BASED CHUNKING
def split_documents_by_sentence(documents):

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". "],
        chunk_size=500,
        chunk_overlap=0
    )

    chunks = []

    for document in documents:

        document_chunks = splitter.split_text(
            document["content"]
        )

        for chunk in document_chunks:

            chunks.append({
                "filename": document["filename"],
                "content": chunk
            })

    return chunks


# CREATE BOTH TYPES OF CHUNKS
fixed_chunks = split_documents(documents)

sentence_chunks = split_documents_by_sentence(documents)

print("CHUNKING RESULTS")

print(f"Fixed-size chunks: {len(fixed_chunks)}")
print(f"Sentence-based chunks: {len(sentence_chunks)}")



# DISPLAY SOME FIXED-SIZE CHUNKS

print("FIRST 3 FIXED-SIZE CHUNKS")

for chunk in fixed_chunks[:3]:

    print("\n--- CHUNK ---")
    print("Source:", chunk["filename"])
    print(chunk["content"])



# DISPLAY SOME SENTENCE-BASED CHUNKS
print("FIRST 5 SENTENCE-BASED CHUNKS")

for chunk in sentence_chunks[:5]:

    print("\n--- CHUNK ---")
    print("Source:", chunk["filename"])
    print(chunk["content"])


#  CREATE EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



#  CREATE EMBEDDINGS FOR FIXED-SIZE CHUNKS
fixed_texts = [
    chunk["content"]
    for chunk in fixed_chunks
]

fixed_embeddings = embedding_model.embed_documents(
    fixed_texts
)

print("FIXED-SIZE EMBEDDINGS")

print(
    f"Created embeddings for {len(fixed_embeddings)} chunks"
)

print(
    f"Embedding dimensions: {len(fixed_embeddings[0])}"
)



# 10. CREATE EMBEDDINGS FOR SENTENCE-BASED CHUNKS

sentence_texts = [
    chunk["content"]
    for chunk in sentence_chunks
]

sentence_embeddings = embedding_model.embed_documents(
    sentence_texts
)

print("SENTENCE-BASED EMBEDDINGS")

print(
    f"Created embeddings for {len(sentence_embeddings)} chunks"
)

print(
    f"Embedding dimensions: {len(sentence_embeddings[0])}"
)