from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()   

docs = [
    Document(page_content="RAG retrieves relevant text and gives it to the model as context.", metadata={"topic": "rag"}),
    Document(page_content="An embedding turns text into a vector of numbers that captures meaning.", metadata={"topic": "embeddings"}),
    Document(page_content="A vector store saves embeddings and finds the most similar ones.", metadata={"topic": "vector store"}),
    Document(page_content="Pizza is best with extra cheese.", metadata={"topic": "food"}),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",   
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db_gemini",
)

results = vectorstore.similarity_search("What does an embedding do?", k=2)
for r in results:
    print(r.metadata, "->", r.page_content)