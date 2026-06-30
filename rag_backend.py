#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Environment Packages installation


# In[3]:


#All required core imports
import os, re, json, time, getpass
from pathlib import Path

#Langchain building blocks
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore

#Groq LLM integration
from langchain_groq import ChatGroq

#LCEL and Chain primitives
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# In[6]:


print(".env exists here:", os.path.exists(".env"))


# In[7]:


#API Keys loading and global variables
from dotenv import load_dotenv

#load keys seamlessly from the local .env file
load_dotenv()

#verify they loaded correctly without printing the secret text
if "GROQ_API_KEY" in os.environ:
    print("Groq API key loaded successfully from environment configuration!")
else:
    print("Error: GROQ_API_KEY not found in .env file")

#Target storage locations matching your hard drive setup
BACKUP_DIR = Path(r"D:\Multimodal-Temporal-RAG\my_notebook_json_backups")
PERSIST_DB_DIR = "./my_langchain_chroma_db"
PARENT_STORE_DIR = "./my_parent_documents_store"
COLLECTION_NAME = "notebooks_parent_child_index"


# In[8]:


#Initialize Embedding Engine and Persistent Storage
print("Loading Local HuggingFace Embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Hook back into the exact vector database instance saved on disk
vectorstore = Chroma(
    collection_name = COLLECTION_NAME,
    embedding_function = embeddings,
    persist_directory = PERSIST_DB_DIR
)

#connect back to the parent files store on the hard drive
local_file_store = LocalFileStore(PARENT_STORE_DIR)
parent_docstore_local = create_kv_docstore(local_file_store)

#Reinitialize the child chunk splitter configurations
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)

#Reassemble the core base retriever structure seamlessly
base_parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      
    docstore=parent_docstore_local, 
    child_splitter=child_splitter, 
)

print("Successfully connected to hard drive vector database and parent document stores!")


# In[12]:


#Hybrid Filtering Search Engine(The core Regex Strategy)
#The Correct Streamlit-Ready Hybrid Retriever ---
def hybrid_regex_retriever(query, filename_filter=None, topic_filter=None, date_filter=None):
    """
    Applies optional metadata dropdown filters directly to your working 
    base_parent_retriever instance before invoking the search.
    """
    chroma_filter = {}
    conditions = []
    
    # 1. Gather all active filters
    if filename_filter:
        # Match your exact casing strategy from your notes (e.g. "Running_notes_part_1.pdf")
        if not filename_filter.lower().endswith(('.pdf', '.json')):
            filename_filter += ".pdf"
        # Make sure the first letter is capitalized to match your database files
        filename_filter = filename_filter[0].upper() + filename_filter[1:]
        conditions.append({"filename": filename_filter})
        
    if topic_filter:
        conditions.append({"topic": topic_filter})
        
    if date_filter:
        conditions.append({"date": date_filter})
        
    # 2. Package filters into ChromaDB operational syntax
    if len(conditions) > 1:
        chroma_filter = {"$and": conditions}
    elif len(conditions) == 1:
        chroma_filter = conditions[0]
    else:
        chroma_filter = None

    # 3. Inject the filter into your base parent retriever search parameters
    if chroma_filter:
        base_parent_retriever.search_kwargs = {"filter": chroma_filter, "k": 4}
    else:
        # Clear filters if none are chosen from your dropdowns
        base_parent_retriever.search_kwargs = {"k": 4}
        
    print(f"--- HYBRID RETRIEVER ACTIVE ---")
    print(f"Active Filters Injected: {chroma_filter}")
    print(f"-------------------------------")
    
    # 4. Invoke the parent retriever (this correctly reads the full pages from your hard drive!)
    return base_parent_retriever.invoke(query)


# In[10]:


#LCEL Chain and Prompt Setup
# 1. Define your exact prompt layout using ChatPromptTemplate
prompt_template = """
You are an expert software engineering assistant. Below is an extracted section from the user's handwritten notebook pages.
Use only this context to answer the question. If you do not know the answer based on the context, state that isn't covered in the notes.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
custom_prompt = ChatPromptTemplate.from_template(prompt_template)

# 2. Configure Cloud LLM via Groq
cloud_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 3. Helper to format document contents for the chain context block
def format_docs(docs):
    """Takes a list of retrieved LangChain Document objects and joins their text content."""
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Assemble the final production LCEL RAG Chain
# We use a lambda variable to seamlessly trigger your custom hybrid regex retriever function
final_rag_chain = (
    {
        "context": lambda x: format_docs(hybrid_regex_retriever(x["question"], x.get("filename"), x.get("topic"), x.get("date"))),
        "question": lambda x: x["question"]
    }
    | custom_prompt
    | cloud_llm
    | StrOutputParser()
)

print("LCEL RAG chain pipeline assembled successfully!")


# In[13]:


#Running a Query through the LCEL chain
# Define target criteria
input_payload = {
    "question": "Explain List written in my notes inside running_notes_part_1",
    "filename": "Running_notes_part_1" # Your metadata constraint parameter
}

# Execute using standard LCEL invocation syntax
response = final_rag_chain.invoke(input_payload)

print("\n=== GROQ LCEL PIPELINE RESPONSE ===\n")
print(response)


# In[ ]:




