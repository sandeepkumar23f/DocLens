from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db_gemini",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lamda_mult": 0.5
    }
)

llm = init_chat_model("groq:openai/gpt-oss-120b", temperature=0, max_tokens=500)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """ You are a helpful AI assistant
Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
 """
         ),
         (
             "human",
             """ Context:
{context}

Question: 
{question}
"""
         )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You: ")
    if query == 0:
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)
    print(f" \n AI: {response.content}")