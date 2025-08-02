import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


def load_llm():
    load_dotenv()
    api_key=os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("Please set your GROQ_API_KEY in the .env file.")
        st.stop()

    return ChatGroq(api_key=api_key, model="llama3-8b-8192")


def build_chain(llm):
    prompt=ChatPromptTemplate.from_template("User: {question}\nAI:")
    output_parser=StrOutputParser()
    return prompt | llm | output_parser


def build_context_prompt(user_input, retriever=None):
    """Build full conversation history and optionally add PDF context."""
    prompt= ""

    # Add past messages
    for entry in st.session_state.history:
        role= "User" if entry["role"] == "user" else "AI"
        prompt+=f"{role}: {entry['content']}\n"

    # Add PDF context if retriever is available
    if retriever:
        docs= retriever.get_relevant_documents(user_input)
        context="\n".join([doc.page_content for doc in docs])
        if context.strip():
            prompt+=f"\nUse the following context to assist:\n{context}\n"

    # Add current question
    prompt+=f"User: {user_input}\nAI:"
    return prompt


def chat_with_bot(chain, full_prompt):
    return chain.invoke({"question": full_prompt})


def main():
    st.set_page_config(page_title="🤖 Hybrid Chatbot", page_icon="📄")
    st.title("🤖 Chat with AI (Groq + RAG)")
    st.caption("Ask anything! Upload a PDF to answer from your own docs.")

    llm=load_llm()
    chain=build_chain(llm)

    if "history" not in st.session_state:
        st.session_state.history=[]

    uploaded_file=st.file_uploader("📄 Upload PDF for RAG", type="pdf")
    retriever=None

    if uploaded_file:
        with st.spinner("📄 Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path=tmp.name

            loader=PyPDFLoader(tmp_path)
            docs=loader.load()

            splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks=splitter.split_documents(docs)

            embeddings=HuggingFaceEmbeddings()
            vectorstore=FAISS.from_documents(chunks, embeddings)
            retriever=vectorstore.as_retriever(search_type="similarity", k=3)

        st.success("✅ PDF processed and ready for questions!")

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    user_input=st.chat_input("Ask your question:")

    if user_input:
        st.chat_message("user").markdown(user_input)

        full_prompt=build_context_prompt(user_input, retriever)

        with st.spinner("💬 Thinking..."):
            response=chat_with_bot(chain, full_prompt)

        st.chat_message("ai").markdown(response)

        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "ai", "content": response})

    if st.button("🗑️ Clear Chat"):
        st.session_state.history=[]
        st.rerun()


if __name__ == "__main__":
    main()

