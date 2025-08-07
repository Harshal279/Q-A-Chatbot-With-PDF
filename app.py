from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st
import os

# Define embedding model
embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("Conversational RAG Model with PDF Upload")
st.write("Upload a PDF and chat with its content!")

apikey = st.text_input("Enter Groq API Key", type="password")

if apikey:
    # Define LLM
    llm = ChatGroq(api_key=apikey, model="llama3-8b-8192")  # double check model name, typo fixed
    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file:
        # Save file temporarily
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load documents
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # History-aware retriever prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        # QA Prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Chat history management
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        # Wrap in RunnableWithMessageHistory
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User input
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("Assistant:", response["answer"])
            st.write("Chat History:")
            for msg in session_history.messages:
                st.write(f"{msg.type.capitalize()}: {msg.content}")
    else:
        st.info("Please upload a PDF to get started.")
else:
    st.warning("Please enter the Groq API Key.")
