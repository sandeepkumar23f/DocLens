from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model

load_dotenv()
data = PyPDFLoader("document loaders/genai_notes.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system","You are an AI pdf summarizer who summarizes the pdf"),
    ("human","{data}")
])

model = init_chat_model(
    model="groq:openai/gpt-oss-120b",
    max_tokens=500
)

prompt = template.format_messages(data = docs[0].page_content)
respo = model.invoke(prompt)
print(respo.content)