from langchain_community.document_loaders import WebBaseLoader

url = "https://my-portfolio12s.netlify.app/"

data = WebBaseLoader(url)
docs = data.load()
print(docs)

