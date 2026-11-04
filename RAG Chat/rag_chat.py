from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma

load_dotenv()

# must be the SAME model that built the index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db_local",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
)

llm = init_chat_model("groq:openai/gpt-oss-120b", temperature=0, max_tokens=500)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
    ("human",
     """Context:
{context}

Question:
{question}
"""),
])

print("RAG system created")
print("Press 0 to exit")

while True:
    query = input("\nYou: ").strip()
    if query == "0":
        break
    if not query:
        continue

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
    print("Sources (page):", sorted({d.metadata.get("page") for d in docs}))