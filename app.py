from retriever import retrieve
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = retrieve(question)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer based only on the context.
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)