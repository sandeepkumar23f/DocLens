import os
import hashlib
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma

load_dotenv()

# =====================================================
# PART 1: LOGIC (your RAG code, unchanged in behavior)
# =====================================================
@st.cache_resource   # load heavy models once, not on every click
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = init_chat_model("groq:openai/gpt-oss-120b", temperature=0, max_tokens=500)
    return embeddings, llm


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


def build_index(pdf_bytes, file_id, embeddings):
    """Load the uploaded PDF, split it, embed it, and store it in memory."""
    # PyPDFLoader needs a file path, so write the upload to a temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        return None, len(pages), 0      # no readable text (probably a scanned PDF)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{file_id}",   # unique name, so uploads never mix
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    return (vectorstore, retriever), len(pages), len(chunks)


def ask(query, retriever, llm):
    """One turn of your loop. Returns the answer and the retrieved docs."""
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


# =====================================================
# PART 2: UI (Streamlit)
# =====================================================
st.set_page_config(page_title="DocLens", page_icon="📄")
st.title("📄 DocLens")
st.caption("Upload a PDF, then ask questions. Answers come only from your document.")

embeddings, llm = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.file_id = None
    st.session_state.retriever = None
    st.session_state.vectorstore = None

# ---- Sidebar: upload ----
with st.sidebar:
    st.header("Your document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None:
        pdf_bytes = uploaded.getvalue()
        file_id = hashlib.md5(pdf_bytes).hexdigest()[:12]

        # only rebuild when a different file is uploaded
        if file_id != st.session_state.file_id:
            if st.session_state.vectorstore is not None:
                try:
                    st.session_state.vectorstore.delete_collection()   # free the old index
                except Exception:
                    pass
            with st.spinner("Reading and indexing your PDF..."):
                try:
                    result, n_pages, n_chunks = build_index(pdf_bytes, file_id, embeddings)
                except Exception as e:
                    st.error(f"Could not read this PDF: {e}")
                    st.stop()

            st.session_state.file_id = file_id
            st.session_state.messages = []          # new document means a new chat
            if result is None:
                st.session_state.retriever = None
                st.session_state.vectorstore = None
                st.session_state.info = (n_pages, 0)
            else:
                st.session_state.vectorstore, st.session_state.retriever = result
                st.session_state.info = (n_pages, n_chunks)

        n_pages, n_chunks = st.session_state.get("info", (0, 0))
        if n_chunks:
            st.success(f"{uploaded.name}: {n_pages} pages, {n_chunks} chunks")
        else:
            st.error("No readable text found. This may be a scanned PDF, which needs OCR.")

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Main area: chat ----
if st.session_state.retriever is None:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(f"Page {s['page']}")
                    st.write(s["text"])

query = st.chat_input("Ask about your document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                answer, docs = ask(query, st.session_state.retriever, llm)
            sources = [
                {"page": (d.metadata.get("page", 0) + 1), "text": d.page_content}
                for d in docs
            ]
            st.write(answer)
            with st.expander("Sources"):
                for s in sources:
                    st.caption(f"Page {s['page']}")
                    st.write(s["text"])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Error: {e}")