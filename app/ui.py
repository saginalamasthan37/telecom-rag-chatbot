import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Verizon AI Support",
    page_icon="📱",
    layout="centered"
)

# API URL
API_URL = "http://localhost:8000/chat"

# Header
st.title("📱 Verizon AI Customer Support")
st.caption("Powered by RAG - Ask anything about your Mobile service")

# Quick question buttons
st.markdown("**Try asking:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Reset router"):
        st.session_state.quick_q = "How do I reset my router?"

with col2:
    if st.button("💳 Pay my bill"):
        st.session_state.quick_q = "How do I pay my bill?"

with col3:
    if st.button("📶 4G vs 5G"):
        st.session_state.quick_q = "What is the difference between 4G and 5G?"

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if available
        if message.get("sources"):
            with st.expander("📄 Sources used"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(src["content"])
                    st.caption(f"From: {src['source']}")

# Handle quick question buttons
prompt = None
if "quick_q" in st.session_state:
    prompt = st.session_state.quick_q
    del st.session_state.quick_q

# Chat input
user_input = st.chat_input("Ask your question here...")
if user_input:
    prompt = user_input

# Process the question
if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": prompt},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    source_count = data["source_count"]

                    # Show answer
                    st.markdown(answer)

                    # Show sources
                    with st.expander(f"📄 Sources used ({source_count})"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(src["content"])
                            st.caption(f"From: {src['source']}")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This chatbot answers questions about Verizon services using RAG.
    
    **Tech Stack:**
    - 🦜 LangChain
    - 🗄️ ChromaDB
    - 🤖 Groq LLM
    - ⚡ FastAPI
    - 🎨 Streamlit
    """)

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()