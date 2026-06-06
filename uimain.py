
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat With Your PDF")

# Models
embedding_model = MistralAIEmbeddings()
llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:
'I could not find the answer in the document.'
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.success(
            f"PDF processed successfully! ({len(chunks)} chunks)"
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input(
            "Ask questions about your PDF..."
        )

        if query:

            st.session_state.messages.append(
                {"role": "user", "content": query}
            )

            with st.chat_message("user"):
                st.markdown(query)

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            response = llm.invoke(final_prompt)

            with st.chat_message("assistant"):
                st.markdown(response.content)

                with st.expander(
                    "Retrieved Chunks"
                ):
                    for i, doc in enumerate(docs, 1):
                        st.write(f"Chunk {i}")
                        st.write(doc.page_content)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.content
                }
            )

