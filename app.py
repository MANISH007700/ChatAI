import streamlit as st
import openai
from loguru import logger 
from dotenv import load_dotenv

from utils import get_pdf_text
from utils import get_text_chunks
from utils import get_vectorstore
from utils import get_conversation_chain
from utils import handle_userinput

from htmlTemplates import css


def main():

    load_dotenv()   # load ing env vars

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.text("Extracting Texts Done ✅")

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.text("Processing Chunks Done ✅")
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                st.text("Memory loaded Done ✅. You can ask questions now..")
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
