import os
from langchain_community.document_loaders import TextLoader
# Use an underscore and add an 's' to the end of splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("documents/company_handbook.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectordb"
)

print("Documents successfully ingested!")