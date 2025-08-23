import streamlit as st
from dotenv import load_dotenv
import os
from rag_utils import answer_question

# Load environment variables from .env
load_dotenv()

st.set_page_config(page_title="📘 PDF RAG Chatbot", page_icon="🤖", layout="wide")

st.title("📘 PDF Question Answering Chatbot (RAG)")
st.write("Upload PDFs to the `docs/` folder, build vectorstore with `run_once.py`, then ask questions here.")

# Show which model is being used (for debugging)
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ Hugging Face token not found. Please set HF_TOKEN in your .env file.")
else:
    st.success("✅ Hugging Face token loaded successfully.")

# User input
query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Thinking..."):
        answer = answer_question(query, HF_TOKEN)
    st.markdown("### ✅ Answer:")
    st.write(answer)
