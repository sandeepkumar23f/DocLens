from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/genai_notes.pdf")
docs = data.load()

print(docs)
print(len(docs)) #it tells the number of pages
