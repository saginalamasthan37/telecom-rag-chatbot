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
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=500,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": TELECOM_PROMPT}
    )

    print("RAG chain created successfully!")
    return chain


def ask(chain, question: str) -> dict:
    """Takes a question, returns answer + sources"""
    result = chain.invoke({"query": question})

    answer = result["result"]
    sources = []

    # Fix: check both possible keys for source documents
    source_docs = result.get("source_documents") or result.get("context") or []

    for doc in source_docs:
        sources.append({
            "content": doc.page_content[:150],
            "source": doc.metadata.get("source", "unknown")
        })

    return {
        "answer": answer,
        "sources": sources,
        "source_count": len(sources)
    }


if __name__ == "__main__":
    print("Loading RAG chain...")
    chain = create_rag_chain()

    test_questions = [
        "How do I reset my router?",
        "What is the difference between 4G and 5G?",
        "How do I pay my bill?",
        "Can I get a free pizza?"
    ]

    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Q: {question}")
        result = ask(chain, question)
        print(f"A: {result['answer']}")
        print(f"Sources used: {result['source_count']} chunks")
        if result['sources']:
            print("First source preview:")
            print(result['sources'][0]['content'])