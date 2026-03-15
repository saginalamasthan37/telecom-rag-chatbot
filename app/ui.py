import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="AI Support",
    page_icon="📱",
    layout="centered"
)

# API URL
API_URL = "http://localhost:8000/chat"

# Header
st.title("📱 AI Customer Support")
st.caption("Powered by RAG — Ask anything about your Mobile service")

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask your question here...")

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
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    st.error(f"API error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")