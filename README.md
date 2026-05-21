# Hindi Document Q&A System

An intelligent Streamlit-based application that allows users to upload Hindi PDF documents and ask questions in Hindi or English. The app uses Retrieval-Augmented Generation (RAG) to extract relevant information and generate accurate, context-aware answers. It also supports comparison between two documents.

---

## 🚀 Features

- 📄 Upload one or two Hindi PDF documents
- 🔍 Ask questions in Hindi or English
- 🧠 Get answers using semantic search and language generation
- 📊 Compare content between two documents
- 🖋️ Handles Mangal font and scanned Hindi text using OCR

---

## 🛠️ Tech Stack

- **Streamlit** – Interactive web interface
- **PyMuPDF** – PDF text extraction
- **EasyOCR** – OCR for scanned Hindi documents
- **SentenceTransformers** – Multilingual embeddings
- **FAISS** – Fast vector similarity search
- **Transformers (GPT2)** – Language generation
- **Torch** – Deep learning backend

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/hindi-doc-qa.git
cd hindi-doc-qa
pip install -r requirements.txt
