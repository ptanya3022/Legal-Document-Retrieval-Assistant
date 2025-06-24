import os
from dotenv import load_dotenv
import streamlit as st

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Load FAISS vector store
db = FAISS.load_local(
    folder_path="embeddings",
    embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 5})

# Prompt Template
prompt_template = """
You are a legal assistant. Answer the user question only using the context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Streamlit UI
st.set_page_config(page_title="Legal Document Retrieval Assistant", layout="wide")
st.title("📄 Legal Document Retrieval Assistant")

st.caption("📘 This assistant understands content from:")
st.markdown("- **Indian Penal Code (IPC)**")
st.markdown("- **Right to Information Act, 2005 (RTI Act)**")

query = st.text_input("💬 Ask your legal question:")

if query:
    with st.spinner("Thinking..."):
        response = rag_chain(query)
        st.subheader("🧠 Answer:")
        st.write(response["result"])

        with st.expander("📚 Source Chunks"):
            for doc in response["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                st.markdown(doc.page_content[:500] + "...")
