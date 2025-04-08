import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# Avoid reloading everything on every run
@st.cache_resource(show_spinner=False)
def setup_chain():
    # Path setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(BASE_DIR, "../data/hostel_info.txt")

    # Load & split docs
    loader = TextLoader(FILE_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    # Vector DB
    embedding = OllamaEmbeddings(model="mistral")
    vectordb = Chroma.from_documents(chunks, embedding)

    # LLM + QA Chain
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    return qa_chain

# UI
st.set_page_config(page_title="Sunrise Hostel Bot", page_icon="🏠")
st.title("🏠 Sunrise Hostel Q&A")

query = st.text_input("Ask something about the hostel 🏫:")

if query:
    with st.spinner("Answering..."):
        qa_chain = setup_chain()
        response = qa_chain.run(query)
    st.success("Done")
    st.write(response)
