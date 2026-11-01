from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)
data = TextLoader("document loaders/notes.txt", encoding="utf-8");
docs = data.load()

chunks = text_splitter.split_documents(docs)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()