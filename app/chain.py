import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "app/vectorstore"


def load_vectorstore():
    """Load the existing ChromaDB vectorstore with HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    print("Vectorstore loaded successfully!")
    return vectorstore


if __name__ == "__main__":
    vs = load_vectorstore()
    results = vs.similarity_search("How do I reset my router?", k=2)
    for i, r in enumerate(results):
        print(f"\nResult {i+1}: {r.page_content[:100]}")