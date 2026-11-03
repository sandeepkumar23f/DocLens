# Created for terminal interface

import os
os.environ["HF_HUB_OFFLINE"]="0"
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.7,
        }
)

model = ChatHuggingFace(llm = llm)

loader = PyPDFLoader("C:/Users/sande/Downloads/sandeep_resume.pdf")
docs = loader.load()

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

print("--------- Type 0 to exit ------------")
while True:
    user_question = input("\n Ask your question: ")
    if user_question == '0':
        break

    prompt = template.format_messages(
        question=user_question,
        data="\n".join(doc.page_content for doc in docs)
    )


    response = model.invoke(prompt)
    print("\n Answer: ",response.content)