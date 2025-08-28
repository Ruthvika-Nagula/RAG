import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face token (must be set in .env)
HF_TOKEN = os.getenv("HF_TOKEN")

# Paths
VECTORSTORE_FOLDER = "vectorstore/"
INDEX_FILE = VECTORSTORE_FOLDER + "faiss_index.bin"
CHUNKS_FILE = VECTORSTORE_FOLDER + "chunks.pkl"

# Load FAISS index and chunks
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")



def retrieve(query, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    results = [chunks[i] for i in I[0]]
    return results


def answer_question(query: str, H) -> str:
    retrieved_chunks = retrieve(query)
    context = "\n".join(retrieved_chunks)
    
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    # Hugging Face inference client (secured with token)
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        
        if response.choices and len(response.choices) > 0:
            return  response.choices[0].message["content"]
        else:
            return "No answer found."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error during inference: {str(e)}"
    
