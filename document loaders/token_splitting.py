from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("document loaders/genai_notes.pdf")

docs = data.load()

splitter = TokenTextSplitter(
    chunk_size=10,
    chunk_overlap=1
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)
print(chunks[1].page_content)