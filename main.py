from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

data = TextLoader("document loaders/notes.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes the texts"),
    ("human", "{data}")
])

# using groq api key
model = init_chat_model(
    "groq:openai/gpt-oss-120b",
    max_tokens=200,   # hard limit on the reply length
    
)

prompt = template.format_messages(data = docs[0].page_content)
resp = model.invoke(prompt)
print(resp.content)