import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Title
st.set_page_config(page_title="Hybrid RAG Chatbot 💬", layout="wide")
st.title("🤖 Hybrid RAG Chatbot - PDF + General Q&A")

# Load GROQ API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("🚫 GROQ_API_KEY not found in environment. Please set it in Streamlit Secrets.")
    st.stop()

# Initialize LLM
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="LLaMA3-8b-8192")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

# Load and split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Initialize Vector Store
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# Main chat interface
if uploaded_file:
    with st.spinner("🔄 Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        chunks = load_and_split_pdf("temp.pdf")
        vectorstore = create_vectorstore(chunks)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        st.success("✅ PDF processed. Ask questions below!")

        # Chat UI
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.chat_input("Ask anything about your PDF or in general...")
        if user_query:
            response = qa_chain.run(user_query)
            st.session_state.chat_history.append((user_query, response))

        # Display history
        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("🧑‍💻"):
                st.markdown(user_msg)
            with st.chat_message("🤖"):
                st.markdown(bot_msg)

else:
    st.info("📂 Please upload a PDF from the sidebar to get started.")






