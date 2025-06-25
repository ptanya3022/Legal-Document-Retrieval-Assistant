# ⚖️ Legal Document Retrieval Assistant  
*A Streamlit-based legal Q&A app using Google Gemini and LangChain (RAG-powered)*

[![Live Demo](https://img.shields.io/badge/Streamlit-Demo-brightgreen)](https://legal-document-retrieval-assistant-uauwpqyjknvxg3pqelandx.streamlit.app/)

---

## 📌 Overview

The **Legal Document Retrieval Assistant** is a Retrieval-Augmented Generation (RAG) application that allows users to ask legal questions and get accurate answers based on official Indian legal documents such as the Indian Penal Code and the RTI Act, 2005.

The system uses **Google Gemini 1.5 Flash** as the LLM and **LangChain** for chaining document retrieval with prompt generation. It provides grounded responses based on actual legal PDFs uploaded to the app.

---

## ✅ Features

- 🔍 Ask questions based on Indian law (IPC, RTI)
- 🧠 Powered by Gemini 1.5 Flash (via Google AI Studio)
- 📄 Parses and embeds PDF legal documents
- 📁 Uses FAISS for vector storage and semantic search
- 🖥️ Simple and fast UI using Streamlit
- 💡 Retrieval-Augmented Generation (RAG) architecture

---

## 🧱 Tech Stack Used

| Component          | Description                            |
|-------------------|----------------------------------------|
| 🧠 LLM             | Google Gemini 1.5 Flash via API Key    |
| 🔗 Framework       | LangChain                              |
| 📄 PDF Parsing     | PyMuPDF + OCR (Tesseract)              |
| 📊 Vector Store    | FAISS                                  |
| 🌐 Frontend        | Streamlit                              |
| 🧪 Development     | Python 3.x, Conda                      |
| 🔐 API Handling    | .env file to manage Gemini API keys    |

---

## ⚙️ How It Works (Simplified RAG Flow)

1. **PDFs are uploaded manually** into the `data/` folder.
2. `vector_store.py` reads and parses the PDFs.
3. It converts the chunks into embeddings and stores them using FAISS inside `embeddings/`.
4. `rag_pipeline.py` handles:
   - fetching user queries
   - retrieving top relevant chunks
   - generating responses via Gemini
5. `app.py` (Streamlit) runs the entire pipeline in a user-friendly interface.

---

## 📂 Project Structure

```
legal-document-retrieval-assistant/
│
├── app.py                  # Streamlit frontend
├── rag_pipeline.py         # RAG pipeline logic
├── vector_store.py         # PDF reading and embedding generation
├── requirements.txt        # Python dependencies
├── .env                    # Gemini API Key
├── data/                   # Folder containing legal PDFs
├── embeddings/             # Stores FAISS vector data
├── venv/                   # Conda virtual environment (not pushed to Git)
```

---

## 💼 Real-World Impact

This project helps:
- 📚 Law students explore Indian laws section-wise  
- 👨‍💼 Professionals search the RTI Act or IPC quickly  
- 👩‍⚖️ Common citizens ask legal questions in plain English  
- 🏢 Can be extended for legal firms, compliance docs, HR policies  

---

## 🧪 How to Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/ptanya3022/legal-document-retrieval-assistant.git
cd legal-document-retrieval-assistant

# 2. Create and activate virtual environment
conda create -p venv python=3.10
conda activate ./venv

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🔮 Future Scope

- Upload custom PDFs from UI  
- Show source snippets with highlighted text  
- Add references or section numbers in responses  
- Support for multilingual queries  
- Improve OCR for scanned legal docs  

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/)  
- [Google AI Studio](https://makersuite.google.com/)  
- [PyMuPDF](https://pymupdf.readthedocs.io/)  
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)  
- [Streamlit](https://streamlit.io/)
