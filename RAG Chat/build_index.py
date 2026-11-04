from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

docs = PyPDFLoader("document loaders/genai_notes.pdf").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(len(docs), "pages ->", len(chunks), "chunks")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db_local",
)
print("Index saved to chroma_db_local")