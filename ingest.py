# Import required libraries

# Loads text files as LangChain documents
from langchain_community.document_loaders import TextLoader

# Splits large documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Free Hugging Face embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Vector database for storing embeddings
from langchain_chroma import Chroma

# Loads environment variables from .env file
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# ============================================================
# STEP 1: LOAD DOCUMENT
# ============================================================

# Load the text file from the documents folder
loader = TextLoader("documents/company_handbook.txt")

# Convert file content into LangChain Document objects
documents = loader.load()


# ============================================================
# STEP 2: SPLIT DOCUMENT INTO CHUNKS
# ============================================================

# Create a text splitter
# chunk_size = maximum characters per chunk
# chunk_overlap = shared characters between consecutive chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split document into smaller chunks
chunks = splitter.split_documents(documents)

# Print number of chunks created
print(f"Total chunks created: {len(chunks)}")


# ============================================================
# STEP 3: LOAD EMBEDDING MODEL
# ============================================================

# Load a free Hugging Face embedding model
# This model converts text into vectors of dimension 384
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ============================================================
# STEP 4: CREATE VECTOR DATABASE
# ============================================================

# Convert chunks into embeddings
# Store embeddings inside ChromaDB
# persist_directory specifies where the database will be saved
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectordb"
)

print("Vector database created successfully!")


# ============================================================
# STEP 5: COMPLETED
# ============================================================

print("Documents successfully ingested!")
