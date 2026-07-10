from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embeddings
)

def retrieve(query):
    results = vectorstore.similarity_search(
        query,
        k=3
    )

    return results