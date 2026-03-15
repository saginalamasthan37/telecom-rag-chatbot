import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

CHROMA_PATH = "app/vectorstore"

# Custom prompt — gives the LLM its personality and rules
TELECOM_PROMPT = PromptTemplate(
    template="""You are a helpful Verizon customer support AI assistant.
Use ONLY the information provided in the context below to answer the customer's question.
If you don't know the answer from the context, say "I don't have that information. Please call 1-800-VERIZON for further assistance."
Always be polite, clear, and professional.

Context:
{context}

Customer Question: {question}

Your Answer:""",
    input_variables=["context", "question"]
)


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


def create_rag_chain():
    """Connect vectorstore + LLM into a full RAG chain"""
    
    # Step 1 — Load vectorstore
    vectorstore = load_vectorstore()

    # Step 2 — Create retriever (finds top 3 relevant chunks)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Step 3 — Load Groq LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=500,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Step 4 — Build the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": TELECOM_PROMPT}
    )

    print("RAG chain created successfully!")
    return chain


if __name__ == "__main__":
    print("Testing RAG chain...")
    chain = create_rag_chain()
    print("Chain is ready!")