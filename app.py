import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2
import os

# Import pysqlite3 if sqlite3 is not available
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    st.error("Failed to import pysqlite3. Please ensure it is installed.")
    st.stop()

def read_and_textify(files):
    """Extract text from PDF files and return lists of texts and source identifiers."""
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            if text:  # Ensure that text is not None
                text_list.append(text)
                sources_list.append(f"{file.name}_page_{i}")
    return text_list, sources_list

st.set_page_config(layout="centered", page_title="AI (I Think?)")
st.header("AI (I Think?)")
st.write("---")

# File uploader
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf"])
st.write("---")

if uploaded_files:
    st.write(f"{len(uploaded_files)} document(s) loaded..")

    textify_output = read_and_textify(uploaded_files)
    documents, sources = textify_output

    # Initialize embeddings
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_api_key"])
    except Exception as e:
        st.error(f"Failed to initialize OpenAI embeddings: {e}")
        st.stop()

    # Initialize Chroma vector store with error handling
    try:
        vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
    except AttributeError as e:
        st.error(f"An AttributeError occurred during Chroma initialization: {e}")
        st.error("This might be due to a change in the Chroma API or method. Please check the latest Chroma documentation for updates.")
        st.error("Documentation: [Chroma Documentation](https://github.com/trychroma/chromadb)")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Chroma initialization: {e}")
        st.stop()

    # Setup retriever and LLM model
    try:
        retriever = vStore.as_retriever(search_kwargs={'k': 2})
    except Exception as e:
        st.error(f"An error occurred while setting up the retriever: {e}")
        st.stop()

    model_name = "gpt-3.5-turbo"
    try:
        llm = OpenAI(model_name=model_name, openai_api_key=st.secrets["openai_api_key"], streaming=True)
        model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    except Exception as e:
        st.error(f"An error occurred while creating the RetrievalQAWithSourcesChain model: {e}")
        st.stop()

    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")

    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = model({"question": user_q}, return_only_outputs=True)
                st.subheader('Your response:')
                st.write(result.get('answer', 'No answer found.'))
                st.subheader('Source pages:')
                st.write(result.get('sources', 'No sources found.'))
        except Exception as e:
            st.error(f"An error occurred during model inference: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
else:
    st.info("Upload files to analyze")
