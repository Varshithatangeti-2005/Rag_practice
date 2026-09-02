# Import the Hugging Face embedding model
# This model converts text into numerical vectors (embeddings)
from langchain_huggingface import HuggingFaceEmbeddings

# Import Chroma vector database
# Chroma stores and searches embeddings
from langchain_community.vectorstores import Chroma


# ============================================================
# STEP 1: LOAD EMBEDDING MODEL
# ============================================================

# Load the embedding model
# all-MiniLM-L6-v2 generates embeddings with 384 dimensions
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ============================================================
# STEP 2: CONNECT TO EXISTING CHROMA DATABASE
# ============================================================

# Load the previously created vector database
# persist_directory points to the folder where embeddings are stored
# embedding_function is required to convert user queries into vectors
vectorstore = Chroma(
    persist_directory="./vectordb",
    embedding_function=embeddings
)

print("Connected to Chroma vector database!")


# ============================================================
# STEP 3: RETRIEVAL FUNCTION
# ============================================================

def retrieve(query):
    """
    Retrieves the most relevant document chunks
    based on the user's query.
    
    Parameters:
        query (str): User's question
        
    Returns:
        list: Top matching document chunks
    """

    # Convert query into an embedding
    # Search the vector database for similar chunks
    # k=3 means return the top 3 most relevant chunks
    results = vectorstore.similarity_search(
        query,
        k=3
    )

    # Return retrieved chunks
    return results


# ============================================================
# STEP 4: TEST RETRIEVAL
# ============================================================

# Example query
query = "What is the work from home policy?"

# Retrieve relevant chunks
documents = retrieve(query)

# Display results
for i, doc in enumerate(documents, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content)
