import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Groq
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser

# Set up the page
st.set_page_config(page_title="Hybrid Chatbot", layout="centered")
st.title("🤖 Hybrid RAG Chatbot (Groq + LangChain)")

# Get Groq API key
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set your GROQ_API_KEY in Streamlit secrets or environment.")
    st.stop()

# Set up LLM
llm = Groq(
    api_key=groq_api_key,
    model="LLaMA3-8b-8192",
)

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load and split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Create FAISS vector store
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process PDF and initialize retrieval chain
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    chunks = load_and_split_pdf("temp.pdf")
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever()
    rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
else:
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant. Engage in conversation and answer questions naturally.

        Chat history:
        {chat_history}

        User: {input}
        Assistant:"""
    )
    chain = prompt | llm | StrOutputParser()

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message in chat
    st.chat_message("user").markdown(user_input)

    if uploaded_file:
        # PDF-based RAG response
        response = rag_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
        answer = response["answer"]
    else:
        # General chat response
        answer = chain.invoke({"input": user_input, "chat_history": memory.buffer})

    # Display answer
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append((user_input, answer))








