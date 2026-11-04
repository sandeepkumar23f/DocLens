from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
    # three near-duplicates about gradient descent
    Document(page_content="Gradient descent is an optimization algorithm that minimizes a loss function by moving in the direction of steepest descent."),
    Document(page_content="Gradient descent minimizes a loss function by repeatedly stepping in the direction of the negative gradient."),
    Document(page_content="Gradient descent is an algorithm that reduces the loss by updating parameters opposite to the gradient."),
    # different but related ideas
    Document(page_content="The learning rate controls how big each gradient descent step is. Too large can overshoot, too small is slow."),
    Document(page_content="Stochastic gradient descent updates parameters using a small random batch instead of the full dataset."),
    Document(page_content="Backpropagation computes the gradients of the loss for each weight in a neural network."),
    # unrelated
    Document(page_content="Pizza is best with extra cheese."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)   # in memory, so no duplicate-folder problem

query = "What is gradient descent"

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

print("Similarity retriever result")
for doc in similarity_retriever.invoke(query):
    print("-", doc.page_content)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 6, "lambda_mult": 0.5},
)

print("\nMMR result")
for doc in mmr_retriever.invoke(query):
    print("-", doc.page_content)