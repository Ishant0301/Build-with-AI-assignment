📄 Build with AI – RAG Chatbot
A Retrieval-Augmented Generation (RAG) based chatbot that leverages LangChain, FAISS, and Google's Gemini model to answer user questions by retrieving relevant information from uploaded files (CSV, PDF, TXT).

🔍 Overview
This application allows users to upload a file and ask questions about its contents. Behind the scenes, the system processes the document, breaks it into chunks, embeds those chunks using HuggingFace embeddings, and indexes them using FAISS for efficient retrieval. When a question is asked, the chatbot retrieves the most relevant chunks and generates a context-aware answer using LangChain with Google's Gemini language model.

🚀 Features
✅ Supports CSV, PDF, and TXT files

✅ Utilizes LangChain and FAISS for document retrieval

✅ Employs Google Gemini for natural language answers

✅ Interactive UI built with Streamlit

✅ Displays source documents for transparency

🛠️ Tech Stack
Python

LangChain

FAISS

Google Generative AI (Gemini)

Streamlit

HuggingFace Transformers

dotenv for environment variable management

