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

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs, API KEY here and click on 'Process'", accept_multiple_files=True)
        openai_api_key = st.text_input("OPENAI API KEY", key="file_qa_api_key", type='password')
        
        if st.button("Process"):
            with st.spinner("Processing"):
                
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.text("Extracting Texts Done ✅")

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.text("Processing Chunks Done ✅")
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks, openai_api_key)
                st.text("Memory loaded Done ✅")
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)


    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Start chatting with your docs.")

    if not user_question:
        st.info("Please upload docs and openai key, and then proceed to chat")

    if user_question and not openai_api_key:
        st.info("Please add openai key ")
    
    if user_question and openai_api_key: 
        handle_userinput(user_question)



if __name__ == '__main__':
    main()
