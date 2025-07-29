import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load vectorstore
vectorstore = FAISS.load_local("vectorstore/changi_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="Changi Chatbot")
st.title("🛫 Changi Airport Chatbot")

query = st.text_input("Ask something about Changi Airport")

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke({"query": query})
        st.success(result["result"])
