from langchain_community.document_loaders import TextLoader

data = TextLoader("document loaders/notes.txt", encoding="utf-8").load() #location of notes.txt file
print(data)