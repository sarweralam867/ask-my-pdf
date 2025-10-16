# Ask My PDF – Local RAG Chatbot

This project lets you chat with your own PDF documents — completely offline and free.
It uses Python, FAISS, Sentence Transformers, and a small local LLM (like TinyLlama) to build a simple RAG (Retrieval-Augmented Generation) system.

## What It Does

- Reads and processes your PDF files.

- Splits the text into smaller chunks for better retrieval.

- Creates and stores embeddings locally using FAISS.

- Lets you ask natural language questions about your document.

- Generates answers using a small open-source model that runs on your CPU — no APIs or paid services.

## Project Structure
```
ask-my-pdf/
│
├── data/
│   └── sample.pdf               # Your input PDF
│
├── embeddings/
│   ├── index.faiss              # Saved vector index
│   └── chunks.pkl               # Stored text chunks
│
├── app.py                       # Reads PDF & builds FAISS index
├── query.py                     # Runs question-answering locally
├── requirements.txt             # Dependencies list
└── README.md                    # This file
```
## Setup Instructions

Clone or download this folder. 
Make sure Python 3.10+ (3.10.11 prefered) is installed.

Create a virtual environment (optional but recommended)
```
python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate  # on Mac/Linux
```

Install dependencies
```
pip install -r requirements.txt
```

## Add your PDF file
Place the file you want to query inside the data folder and name it sample.pdf.

### Build the embeddings
```
python app.py
```

This step extracts text, splits it into chunks, and saves embeddings in the embeddings/ folder.

### Run the chatbot
```
python query.py

```
The first time you run it, the model will be downloaded (around 2GB for TinyLlama).
After that, it loads instantly from cache.

Example Usage
```
Ask a question (or type 'exit' to quit): What is the main topic of this pdf?
Time taken: 45.12 seconds

Answer:
The main topic of this PDF is the academic and research contributions of Brac University.
```
## Performance Notes

Everything runs fully offline once the model is downloaded.

On CPU, each answer may take 30–90 seconds depending on your hardware.

If you need faster responses:

- Use a smaller model like facebook/opt-350m.

- Limit token generation in query.py (reduce max_new_tokens).

- Run on a machine with more CPU cores or GPU.

## Key Dependencies

- PyPDF2 – extract text from PDFs

- sentence-transformers – create text embeddings

- faiss-cpu – vector database for fast retrieval

- transformers – run the local language model

- torch – backend for inference

All dependencies and versions are pinned in requirements.txt.

## Future Improvements

- Add a Streamlit UI for easier interaction.

- Support multiple PDFs.

- Include context highlighting in responses.

## Author: H.M. Sarwer Alam
## License: MIT 
