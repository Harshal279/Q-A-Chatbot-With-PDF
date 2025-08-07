
# 🧠 Conversational RAG with PDF Upload using Groq & Langchain

This is a **Conversational RAG (Retrieval-Augmented Generation)** web app built with **Streamlit**, powered by **Groq's LLaMA3**, **Langchain**, and **HuggingFace Embeddings**.

> 📄 Upload a PDF and start chatting with its content! It supports contextual memory, intelligent reformulations, and ultra-fast response powered by Groq.

---

## 🚀 Features

- 📁 Upload and parse PDF files
- 🧠 Contextual question answering (history-aware retriever)
- 💬 Chat interface with LLaMA3 running on Groq API
- 🔄 Conversational memory using session state
- ⚡ Fast, reliable vector retrieval using Chroma DB & HuggingFace embeddings

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) - Web Interface
- [Langchain](https://www.langchain.com/) - Retrieval & RAG framework
- [Groq API](https://console.groq.com/) - LLM backend (LLaMA3)
- [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Text embeddings
- [Chroma Vector Store](https://www.trychroma.com/) - In-memory vector database
- [PyPDFLoader](https://docs.langchain.com/docs/modules/data_connection/document_loaders/pdf) - PDF parsing

---

## 🖼️ Demo Screenshot

![Screenshot](screenshot.png)

---

## 🧑‍💻 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/conversational-rag-groq.git
cd conversational-rag-groq
```

### 2. Install dependencies

Make sure you have Python 3.9+ and pip installed.

```bash
pip install -r requirements.txt
```

### 3. Get Your Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Create an account or log in
3. Go to **API Keys** tab
4. Click **Create API Key**, give it a name, and copy the key

Then, create a `.env` file in the root directory and add your key:

```env
Paste your Groq api key in place
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will launch in your browser. Upload any PDF file and start chatting!

---

---

## 🧠 Acknowledgements

- [Langchain](https://www.langchain.com/)
- [Groq](https://groq.com/)
- [Streamlit](https://streamlit.io/)
- [Chroma](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/)

---
