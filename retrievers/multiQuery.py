import logging
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# show the generated query variants
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

docs = [
    Document(page_content="Gradient descent is an optimization algorithm that minimizes a loss function by moving in the direction of steepest descent."),
    Document(page_content="Gradient descent minimizes a loss function by repeatedly stepping in the direction of the negative gradient."),
    Document(page_content="Gradient descent is an algorithm that reduces the loss by updating parameters opposite to the gradient."),
    Document(page_content="The learning rate controls how big each gradient descent step is. Too large can overshoot, too small is slow."),
    Document(page_content="Stochastic gradient descent updates parameters using a small random batch instead of the full dataset."),
    Document(page_content="Backpropagation computes the gradients of the loss for each weight in a neural network."),
    Document(page_content="Pizza is best with extra cheese."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = init_chat_model("groq:openai/gpt-oss-120b", temperature=0, max_tokens=500)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

query = "What is gradient descent"

print("\nRetrieved Documents:\n")

for doc in multi_query_retriever.invoke(query):
    print("-", doc.page_content)



