import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(EMBEDDINGS_DIR, "chunks.pkl")

def load_index_and_chunks():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print("No embeddings found. Please run app.py first to create them.")
        exit()
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_similar_chunks(query, index, chunks, model, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

def generate_answer(context, question, generator):
    prompt = f"Answer the question using the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    result = generator(prompt, max_new_tokens=250, temperature=0.5, do_sample=True)
    return result[0]["generated_text"]

if __name__ == "__main__":
    index, chunks = load_index_and_chunks()
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading language model (this may take a moment)...")
    
    
    #model_name = "microsoft/phi-2"  # CPU Friendly Small Model || Ram required 16 GB+ 
    #model_name = "facebook/opt-350m" # More smaller model 
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    
    #---------------------------------------------
    #Only for TinyLlama/TinyLlama-1.1B-Chat-v1.0
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # CPU Friendly Smaller Model ||  Ram Required ~4 GB 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    #---------------------------------------------
    
    while True:
        import time
        start = time.time() 
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        retrieved_chunks = retrieve_similar_chunks(query, index, chunks, emb_model)
        context = "\n\n".join(retrieved_chunks)
        answer = generate_answer(context, query, generator)
        end = time.time()
        print("Time taken:", round(end - start, 2), "seconds")
        print("\nAnswer:\n")
        print(answer)
