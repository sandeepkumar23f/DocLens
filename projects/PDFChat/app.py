import streamlit as st
from main import answer_question

# Page configuration
st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="centered"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_key" not in st.session_state:
    st.session_state.pdf_key = None


# Header
st.title("📄 PDF Chat")
st.caption("Upload your PDF and ask anything about it.")

st.divider()


# PDF Upload Section
st.subheader("Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    # Clear previous conversation when PDF changes
    current_pdf = (uploaded_file.name, uploaded_file.size)

    if st.session_state.pdf_key != current_pdf:
        st.session_state.messages = []
        st.session_state.pdf_key = current_pdf

    st.divider()

    st.subheader("Ask your question")

    # Question input and send button
    with st.form("question_form", clear_on_submit=True):

        user_question = st.text_input(
            "Your question",
            placeholder="What are the technical skills?"
        )

        send_button = st.form_submit_button(
            "Send 🚀",
            use_container_width=True
        )

    # Process question
    if send_button and user_question.strip():

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.spinner("Generating answer..."):

            try:
                answer = answer_question(
                    uploaded_file,
                    user_question
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {e}")


    # Display conversation
    st.divider()
    st.subheader("Conversation")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

else:
    st.info("Please upload a PDF to start chatting.")