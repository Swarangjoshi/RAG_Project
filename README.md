# 📚 PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) based Question Answering System that allows users to ask questions about a PDF document and get answers from the document.

## 🚀 Project Overview

This project demonstrates a complete RAG pipeline using a Python programming notes PDF.

The system extracts text from the PDF, divides the text into chunks, creates embeddings, stores them in FAISS, retrieves relevant chunks for a user question, and uses FLAN-T5 to generate the final answer.

## 🧠 RAG Pipeline

PDF
↓
Text Extraction
↓
Text Chunking
↓
Sentence Transformer Embeddings
↓
FAISS Vector Database
↓
Question Embedding
↓
Similarity Search
↓
Relevant Context
↓
FLAN-T5
↓
Final Answer

## 🛠️ Technologies Used

- Python
- PyPDF
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5
- Streamlit
- NumPy

## ✨ Features

- 📄 PDF document processing
- ✂️ Text chunking
- 🔢 Text embeddings
- 🔍 FAISS similarity search
- 🤖 Local FLAN-T5 model
- 💬 Question answering
- 📌 Retrieved source chunks
- 🌐 Streamlit web interface
- 🔄 Multiple questions

## 📁 Project Structure

```text
RAG_Project/
│
├── data/
│   └── PYTHON PROGRAMMING NOTES.pdf
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```
