import os
os.environ["HF_HUB_OFFLINE"] = "0"
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize Hugging Face model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.7,
    }
)

model = ChatHuggingFace(llm=llm)

# Load PDF
loader = PyPDFLoader("C:/Users/sande/Downloads/sandeep_resume.pdf")
docs = loader.load()

# Create prompt template
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI PDF summarizer. Summarize the given content in simple, clear language."),
    ("human", "Summarize the following PDF content:\n\n{data}")
])

# Generate prompt
prompt = template.format_messages(
    data=docs[0].page_content
)

# Get response
response = model.invoke(prompt)

print(response.content)