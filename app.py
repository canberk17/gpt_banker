import os

from langchain.llms import OpenAI

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from dotenv import load_dotenv

from langchain.agents.agent_toolkits import(
  create_vectorstore_agent,
  VectorStoreToolkit,
  VectorStoreInfo
)

# Load environment variables from .env file
load_dotenv()

os.environ['OPENAI_API_KEY']= os.getenv('OPENAI_API_KEY')


llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('annualreport.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

vectorstore_info=VectorStoreInfo(
  name='annualreport',
  description="a banking annual reports as a pdf",
  vectorstore=store

)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('💲💲 GPT Investment Banker')

prompt=st.text_input('Input your prompt here')


if prompt:

  # response=llm(prompt)

  response=agent_executor.run(prompt)

  st.write(response)

  with st.expander('Document Similarity Search'):
    search=store.similarity_search