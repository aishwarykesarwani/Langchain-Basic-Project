import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# Load hostel data
loader = TextLoader("../data/hostel_info.txt")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embed & store in vector DB
embedding = OllamaEmbeddings(model="mistral")
db = Chroma.from_documents(docs, embedding)

# Load LLM
llm = Ollama(model="mistral")

# Retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False
)

# Streamlit UI
st.title("🏠 Hostel Chatbot (Offline)")
user_q = st.text_input("Ask me anything about the hostel:")

if user_q:
    with st.spinner("Thinking..."):
        response = qa.run(user_q)
    st.markdown("**Answer:**")
    st.write(response)
