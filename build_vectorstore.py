from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import glob
from pathlib import Path
import os

docs = []
for filename in glob.glob("Data/*.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        docs.append(Document(page_content=content))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("vectorstore/changi_index")
print("Vectorstore built and saved.")