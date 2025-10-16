from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

PDF_PATH = "data/sample.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(EMBEDDINGS_DIR, "chunks.pkl")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_text(text)
    return chunks

def create_and_save_embeddings(chunks):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Embeddings created and saved successfully.")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found at {PDF_PATH}. Please place a file named 'sample.pdf' in the data/ folder.")
        exit()

    print("Reading PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(text)} characters of text.")

    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and saving index...")
    create_and_save_embeddings(chunks)
    print("All steps completed.")
