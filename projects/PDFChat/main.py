# With Streamlit UI

import os
import tempfile

os.environ["HF_HUB_OFFLINE"] = "0"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader


# Initialize LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.7,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)


# Prompt template
template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI PDF assistant.

        Answer the user's question using ONLY the information
        provided in the PDF.

        Give short, accurate and relevant answers.
        Do not make up information.

        If the answer is not available in the PDF, say:
        'The information is not available in the provided PDF.'
        """
    ),
    (
        "human",
        """
        PDF Content:
        {data}

        User Question:
        {question}
        """
    )
])


# Function to process PDF and answer questions
def answer_question(uploaded_file, user_question):

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        pdf_content = "\n".join(
            doc.page_content for doc in docs
        )

        prompt = template.format_messages(
            question=user_question,
            data=pdf_content
        )

        response = model.invoke(prompt)

        return response.content

    finally:
        os.remove(pdf_path)