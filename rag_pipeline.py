import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env file and Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the FAISS vector store
db = FAISS.load_local(
    folder_path="embeddings",
    embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Set up the retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# Initialize Gemini Pro model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Prompt template 
prompt_template = """
You are a legal assistant. Use the following context to answer the user's question.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# Setup RAG pipeline with prompt
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Ask a test question
query = "What is Section 302 of the Indian Penal Code?"
response = rag_chain(query)

# Print the answer
print("\n🧠 Answer:")
print(response["result"])
