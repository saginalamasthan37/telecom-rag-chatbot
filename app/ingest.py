import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_PATH = "app/vectorstore"
DATA_PATH = "app/data"


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks):
    print("Creating embeddings using HuggingFace (free, runs locally)...")
    print("First time will download the model (~90MB) — please wait...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old vectorstore")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} chunks to ChromaDB at {CHROMA_PATH}")
    return vectorstore


def test_retrieval(vectorstore):
    print("\n--- Testing retrieval ---")
    query = "How do I reset my router?"
    results = vectorstore.similarity_search(query, k=3)

    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200])
        print("---")


def main():
    print("=== Starting data ingestion ===")
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    test_retrieval(vectorstore)
    print("\n=== Ingestion complete! ===")


if __name__ == "__main__":
    main()