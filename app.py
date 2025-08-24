import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os

st.set_page_config(page_title="🛫 Changi Airport Chatbot")

st.title("🛫 Changi Airport Chatbot")
query = st.text_input("Ask something about Changi Airport")

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    folder_path="vectorstore",
    index_name="index",  # not index.faiss
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)


# Use Hugging Face model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["hf_token"]

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",   # small & reliable model
    temperature=0.4,
    max_new_tokens=512,
)


qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

if query:
    with st.spinner("Thinking..."):
        response = qa.invoke({"query": query})
        st.write("💬", response["result"])

