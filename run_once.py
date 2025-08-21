import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader   # safer than PyPDF2

# Paths
DOCS_FOLDER = "docs/"
VECTORSTORE_FOLDER = "vectorstore/"
INDEX_FILE = os.path.join(VECTORSTORE_FOLDER, "faiss_index.bin")
CHUNKS_FILE = os.path.join(VECTORSTORE_FOLDER, "chunks.pkl")

# Make sure vectorstore exists
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ✅ Load all PDFs safely
def load_pdfs(folder):
    texts = []
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("⚠️ docs/ folder created. Please add PDFs inside it and rerun.")
        return texts

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            filepath = os.path.join(folder, file)
            try:
                reader = PdfReader(filepath)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")
    return texts


# ✅ Chunk text into smaller pieces
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# Step 1: Load PDFs
all_texts = load_pdfs(DOCS_FOLDER)

if not all_texts:
    print("❌ No valid PDF content found in docs/. Add PDFs and rerun.")
    exit()

# Step 2: Chunk texts
all_chunks = []
for txt in all_texts:
    all_chunks.extend(chunk_text(txt))

# Step 3: Generate embeddings
embeddings = model.encode(all_chunks, convert_to_numpy=True)

# Step 4: Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Vectorstore created successfully!")
print(f"📂 Saved FAISS index at: {INDEX_FILE}")
print(f"📂 Saved chunks at: {CHUNKS_FILE}")
