import streamlit as st
import time
import base64
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

import streamlit.components.v1 as components


from web_template import css, page_title, bot_template, user_template, hr_sidebar, notice_text, sidebar_about, sidebar_howto

def get_pdf_doc(pdf_file):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits


def get_webpage_url(webpage_url):
    print("Loading document from URL...")
    st.markdown(''' :green[Loading document from URL...] ''')
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    print("Splitting document into chunks...")
    st.markdown(''' :green[Splitting document into chunks...] ''')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits



def get_pdf_content(documents):
    st.markdown(''' :green[Loading PDF document ...] ''')
    raw_text = ""

    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text


def get_chunks(text):
    st.markdown(''' :green[Splitting document into chunks...] ''')
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_embeddings(chunks, embeddings_model):
    start_time = time.time()
    if embeddings_model == "nomic-embed-text":
       model_name = 'nomic-embed-text'
       print('embeddings_model = ', embeddings_model)
       embeddings = OllamaEmbeddings(model=model_name)
    elif embeddings_model == "sentence-transformers/all-mpnet-base-v2":
       print('embeddings_model = ', embeddings_model)
       model_name = "sentence-transformers/all-mpnet-base-v2"
       model_kwargs = {"device": "mps"}
       embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    else:
       embeddings = OllamaEmbeddings(model=embeddings_model)
     
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
    print(f"Embedding time: {time.time() - start_time:.2f} seconds")
    st.write(f"Embedding time: {time.time() - start_time:.2f} seconds")


    return vector_storage

def get_embeddings_docs(chunks, embeddings_model):
    if embeddings_model == "nomic-embed-text":
       print('embeddings_model = ', embeddings_model)
       embeddings = OllamaEmbeddings(model=embeddings_model)
    elif embeddings_model == "sentence-transformers/all-mpnet-base-v2":
       print('embeddings_model = ', embeddings_model)
       model_name = "sentence-transformers/all-mpnet-base-v2"
       model_kwargs = {"device": "mps"}
       embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    else:
       embeddings = OllamaEmbeddings(model=embeddings_model)
    vector_storage = FAISS.from_documents(chunks, embeddings)

    return vector_storage




def start_conversation(vector_embeddings, system_prompt):
    llm=Ollama(model="llama3", temperature=0.1)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{system_prompt}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:"
    )

    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return conversation


def get_image_url(image_url):
    file_ = open(image_url, "rb")
    contents = file_.read()
    icon_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return icon_url  


def process_query(query_text):
    response = st.session_state.conversation({'question': query_text})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content).
                replace("{{IMG1}}", get_image_url('copy-icon.png')).replace("{{IMG2}}", get_image_url('happy-icon.png')).
                replace("{{IMG3}}", get_image_url('sad-icon.png')), unsafe_allow_html=True)
            #st.download_button("Save to File", message.content, "BruinRag_Download.txt", key=i) 


def main():

    st.set_page_config(page_title="Chat with PDFs, URLs in Local RAG", page_icon=":books:", layout="wide")

    @st.dialog("Select a Data Source")
    def add_data_source():                    
        webpage_url = st.text_input("Enter Webpage URL", type="default", key="webpage_url")
        add_url_button = st.button("Add URL")
        if add_url_button:
            with st.spinner("Processing..."):
                text_chunks    = get_webpage_url(webpage_url)
                vector_embeddings = get_embeddings_docs(text_chunks, embeddings_model)
                st.session_state.conversation = start_conversation(vector_embeddings, st.session_state.system_prompt)
            st.rerun()
        st.subheader("PDF Documents")
        documents = st.file_uploader(
            "Upload your PDF files and and Click Run PDF", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Run PDF"):
            with st.spinner("Processing..."):
                extracted_text = get_pdf_content(documents)
                text_chunks = get_chunks(extracted_text)
                vector_embeddings = get_embeddings(text_chunks, embeddings_model)
                st.session_state.conversation = start_conversation(vector_embeddings, st.session_state.system_prompt)
            st.rerun()    


    add_source_button = st.button("Add a Data Source") 
    if add_source_button:
            add_data_source()  

    with open('styles.css') as f:
        CSS = f.read() 

    st.markdown(f'<style>{CSS}</style>', unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)

    st.sidebar.subheader('Models and Embeddings')
    llm_model = st.sidebar.selectbox("Select Model", options=["llama3", "llama2"])
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["sentence-transformers/all-mpnet-base-v2", "nomic-embed-text", "llama3"],
        help="When you change the embeddings model, the documents will need to be added again.",
    )
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    elif st.session_state["embeddings_model"] != embeddings_model:
        print("New embedding model selected...", embeddings_model)
        st.markdown(''' :green[New embedding model selected...] ''')
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True

    st.sidebar.write(hr_sidebar, unsafe_allow_html=True)
    st.sidebar.subheader('System Prompt')
    system_prompt = st.sidebar.text_area(
        "Customize the assistant's behavior:",
        value="You are a helpful assistant that answers questions based on the provided documents. Be concise and accurate in your responses.",
        height=350,
        help="This prompt guides how the assistant responds to your questions."
    )
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = system_prompt
    else:
        st.session_state["system_prompt"] = system_prompt

    def clear_cache():
        keys = list(st.session_state.keys())
        for key in keys:
            st.session_state.pop(key)

    def clear_chat_history():
        st.session_state["chat_history"] = 0

    query = st.text_input("")

    if query:
        process_query(query)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(notice_text, unsafe_allow_html=True)
     
    st.sidebar.write(hr_sidebar, unsafe_allow_html=True)    


if __name__ == "__main__":
    main()