import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Groq

# Set Streamlit page config
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

# Title and description
st.title("🤖 Hybrid RAG Chatbot with Groq + LangChain")
st.markdown("Ask anything or upload a PDF to ask document-specific questions.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- PDF Upload Section ---
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            loader = PyPDFLoader(uploaded_file.name)
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Embeddings and FAISS
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.success("✅ PDF processed and indexed.")

# --- User Input Section ---
query = st.text_input("💬 Ask a question")

# --- LLM Setup ---
llm = Groq(
    model="llama3-8b-8192",
    api_key=os.environ.get("GROQ_API_KEY")
)

# --- Chat Logic ---
if query:
    if st.session_state.vectorstore:
        # RAG over PDF
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True
        )
        with st.spinner("Answering from PDF..."):
            result = qa_chain(query)
            st.write("📚 Answer from document:")
            st.markdown(result["result"])
    else:
        # General Chat
        with st.spinner("Thinking..."):
            response = llm.invoke(query)
            st.write("🌐 General Answer:")
            st.markdown(response)




