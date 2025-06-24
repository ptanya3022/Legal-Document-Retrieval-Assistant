import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import fitz  # PyMuPDF


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

DATA_PATH = "data"
EMBEDDING_PATH = "embeddings"
file_path= DATA_PATH

os.makedirs(EMBEDDING_PATH, exist_ok=True)

def load_pdf_flexibly(file_path):
    try:
        print(f"🔍 Trying PyPDFLoader: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        if pages:
            return pages
    except Exception as e:
        print(f"⚠️ PyPDFLoader failed: {e}")
    
    print(f"🧾 Using OCR fallback for: {file_path}")
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)

    return [Document(page_content=text)]

def load_all_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_PATH, filename)
            docs = load_pdf_flexibly(path)
            documents.extend(docs)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

def main():
    print("📂 Loading PDFs from /data...")
    docs = load_all_documents()
    print(f"✅ Total documents loaded: {len(docs)}")

    chunks = split_documents(docs)
    print(f"✂️ Total chunks: {len(chunks)}")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    vectordb.save_local(EMBEDDING_PATH)
    print("✅ FAISS vector store saved to /embeddings")

if __name__ == "__main__":
    main()
