# Rag-Ollma-Chat-bot

````markdown
# Retrieval-Augmented Generation (RAG) with Ollama and LangChain

This Python project implements a Retrieval-Augmented Generation (RAG) system that uses LangChain, Ollama LLM (Mistral model), Hugging Face embeddings, and Chroma vector database to answer questions based on PDF documents.

---

## Features

- Load PDF documents from a specified directory
- Split documents into chunks for better embedding and retrieval
- Generate embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model
- Store and query document embeddings in Chroma vector database
- Query the Ollama Mistral LLM to generate answers based on retrieved context
- Supports clearing and updating the vector database to keep documents up to date

---

## Requirements

- Python 3.8+
- The following Python packages (see `requirements.txt`):

  - langchain
  - langchain-ollama
  - langchain-community
  - langchain-huggingface
  - sentence-transformers
  - chromadb
  - PyPDF2

---

## Usage

1. Place your PDF files in the `data/` directory.

2. Run the script:

   ```bash
   python main.py
````

3. When prompted, choose whether to index the PDF files. Indexing splits the documents, generates embeddings, and stores them in the vector database.

4. After indexing, you can ask questions interactively. The system will retrieve relevant document chunks and query the Ollama Mistral model to answer.

5. Type `quit` to exit the program.

---

## How it works

* **Loading documents**: PDF files are loaded using `PyPDFDirectoryLoader`.
* **Splitting text**: Documents are split into overlapping chunks for efficient retrieval.
* **Embeddings**: Each chunk is embedded using Hugging Face's MiniLM model.
* **Chroma DB**: Embeddings are stored and queried in Chroma vector database.
* **Querying LLM**: The retrieved context is passed to the Ollama Mistral LLM to generate answers.
* **Persistence**: The vector database is persisted per user to avoid re-indexing unchanged documents.

---

## Notes

* The Ollama server with Mistral model should be running locally or accessible to the script.
* The `dbs_<user_id>` folders store the vector database files and should be preserved between runs.
* You can customize chunk size, overlap, and other parameters in the code.

---

## Author

Hayati Yurtoğlu

---

Feel free to contribute or raise issues.

```

---

İstersen bunu senin proje klasörüne `README.md` olarak eklemen için `.md` formatında da hazırlayabilirim.
```
