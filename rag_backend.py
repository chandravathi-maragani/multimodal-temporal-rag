#!/usr/bin/env python
# coding: utf-8

import asyncio
import os
import re
import json
import time
import difflib
from pathlib import Path
from dotenv import load_dotenv

# Langchain core structures
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore

# Groq LLM integration
from langchain_groq import ChatGroq

# LCEL and Chain primitives
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment keys
load_dotenv()

# Target hard drive directories 
PERSIST_DB_DIR = "./my_langchain_chroma_db"
PARENT_STORE_DIR = "./my_parent_documents_store"
COLLECTION_NAME = "notebooks_parent_child_index"

# Initialize Engines
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DB_DIR
)
local_file_store = LocalFileStore(PARENT_STORE_DIR)
parent_docstore_local = create_kv_docstore(local_file_store)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)

base_parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      
    docstore=parent_docstore_local, 
    child_splitter=child_splitter, 
)

# -----------------------------------------------------------------------------
# 1. METADATA SOURCE OF TRUTH INVENTORY GENERATOR
# -----------------------------------------------------------------------------
def generate_metadata_source_of_truth(json_dir_path: str) -> dict:
    unique_filenames = set()
    unique_topics = set()
    unique_dates = set()
    json_path = Path(json_dir_path)
    
    if not json_path.exists():
        return {"filenames": [], "topics": [], "dates": []}
        
    months_2025 = {"sep", "oct", "nov", "dec", "september", "october", "november", "december"}
    
    for file_path in json_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fallback_filename = file_path.stem + ".pdf"
            filename_val = data.get("filename", fallback_filename)
            if filename_val:
                filename_val = filename_val.strip()
                if filename_val.lower().endswith('.json'):
                    filename_val = filename_val[:-5]
                if not filename_val.lower().endswith('.pdf'):
                    filename_val += ".pdf"
                filename_val = filename_val[0].upper() + filename_val[1:]
                unique_filenames.add(str(filename_val))
                
            pages = data.get("pages", [])
            for page in pages:
                topic_val = page.get("topic")
                if topic_val:
                    unique_topics.add(str(topic_val).lower().strip())
                raw_date = page.get("date")
                if raw_date and str(raw_date).strip().lower() not in ["unknown", "n/a", "undated"]:
                    clean_date = str(raw_date).strip()
                    lower_date = clean_date.lower()
                    if re.search(r'\b(2025|2026)\b', clean_date):
                        unique_dates.add(clean_date)
                    else:
                        if any(m in lower_date for m in months_2025):
                            unique_dates.add(f"2025 {clean_date}")
                        else:
                            unique_dates.add(f"2026 {clean_date}")
        except Exception:
            continue
            
    return {
        "filenames": sorted(list(unique_filenames)),
        "topics": sorted(list(unique_topics)),
        "dates": sorted(list(unique_dates))
    }

# Safe initialization for Streamlit imports
def get_cached_metadata():
    try:
        return generate_metadata_source_of_truth(r"D:\Multimodal-Temporal-RAG\my_notebook_json_backups")
    except Exception:
        return {"filenames": [], "topics": [], "dates": []}

# -----------------------------------------------------------------------------
# 2. FUZZY MATCH EXTRACTION ENGINE
# -----------------------------------------------------------------------------
def extract_metadata_via_fuzzy_match(user_query: str, metadata_directory: dict) -> dict:
    query_clean = user_query.strip().lower()
    extracted_filters = {"filename": None, "topic": None, "date": None}

    best_file_match = None
    highest_file_ratio = 0.0
    for filename in metadata_directory.get("filenames", []):
        base_name = filename.lower().replace(".pdf", "").replace("_", " ")
        if base_name in query_clean or filename.lower() in query_clean:
            extracted_filters["filename"] = filename
            break
        ratio = difflib.SequenceMatcher(None, base_name, query_clean).ratio()
        if ratio > highest_file_ratio and ratio > 0.4:
            highest_file_ratio = ratio
            best_file_match = filename
    if not extracted_filters["filename"] and highest_file_ratio > 0.6:
        extracted_filters["filename"] = best_file_match

    best_topic_match = None
    highest_topic_ratio = 0.0
    for topic in metadata_directory.get("topics", []):
        if topic in query_clean:
            extracted_filters["topic"] = topic
            break
        ratio = difflib.SequenceMatcher(None, topic, query_clean).ratio()
        if ratio > highest_topic_ratio and ratio > 0.5:
            highest_topic_ratio = ratio
            best_topic_match = topic
    if not extracted_filters["topic"] and highest_topic_ratio > 0.7:
        extracted_filters["topic"] = best_topic_match

    for date_str in metadata_directory.get("dates", []):
        raw_date_part = date_str.replace("2025 ", "").replace("2026 ", "").lower()
        source_tokens = set(re.findall(r'[a-z0-9]+', raw_date_part))
        clean_query_processed = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', query_clean)
        query_tokens = set(re.findall(r'[a-z0-9]+', clean_query_processed))

        if raw_date_part in query_clean or date_str.lower() in query_clean:
            extracted_filters["date"] = date_str
            break
        if source_tokens.issubset(query_tokens):
            extracted_filters["date"] = date_str
            break
        day_token = next((t for t in source_tokens if t.isdigit()), None)
        month_token = next((t for t in source_tokens if not t.isdigit()), None)
        if day_token in query_tokens and month_token:
            if any(q_tok.startswith(month_token[:3]) or month_token.startswith(q_tok[:3]) for q_tok in query_tokens if not q_tok.isdigit()):
                extracted_filters["date"] = date_str
                break

    return extracted_filters
     
# -----------------------------------------------------------------------------
# 3. HYBRID RETRIEVER ENGINE
# -----------------------------------------------------------------------------
async def hybrid_regex_retriever_async(query, filename_filter=None, topic_filter=None, date_filter=None):
    # 1. Broaden the initial vector search pool if the user is asking a vague temporal question
    if date_filter and any(keyword in query.lower() for keyword in ["what notes", "what did i write", "what happened", "query", "lists"]):
        base_parent_retriever.search_kwargs = {"k": 100}
    else:
        base_parent_retriever.search_kwargs = {"k": 20} 
    
    # Extract the base documents from your store
    all_docs = await base_parent_retriever.ainvoke(query)
    
    if not filename_filter and not topic_filter and not date_filter:
        return all_docs[:4]
        
    filtered_docs = []
    
    # 2. RUN INDEPENDENT VALIDATION LOOPS FOR EACH FILTER KEY
    for doc in all_docs:
        meta = doc.metadata or {}
        page_text = doc.page_content.lower()
        
        # --- A. DATE FILTER KEYWORD CHECK ---
        if date_filter:
            target_date = str(date_filter).lower().strip()
            clean_target_str = target_date.replace("2025", "").replace("2026", "").replace("-", " ").replace("/", " ").replace(":", " ").strip()
            raw_tokens = set(re.findall(r'[a-z0-9]+', clean_target_str))
            clean_target_tokens = {t[:3] if (not t.isdigit() and len(t) > 3) else t for t in raw_tokens}

            clean_pool_text = page_text.replace("-", " ").replace("/", " ").replace(":", " ")
            pool_tokens = set(re.findall(r'[a-z0-9]+', clean_pool_text))
            normalized_pool_tokens = {t[:3] if (not t.isdigit() and len(t) > 3) else t for t in pool_tokens}

            is_date_in_chunk = clean_target_tokens.issubset(normalized_pool_tokens) or clean_target_str in clean_pool_text
            if not is_date_in_chunk:
                continue # Skip this document chunk if the date isn't explicitly mentioned

        # --- B. FILENAME FILTER METADATA CHECK (UNTOUCHED) ---
        if filename_filter:
            doc_file = str(meta.get("filename", "")).lower()
            target_file = str(filename_filter).lower()
            if target_file not in doc_file and doc_file not in target_file:
                continue # Skip if filenames don't align
                
        # --- C. TOPIC FILTER METADATA CHECK (UNTOUCHED) ---
        if topic_filter:
            doc_topic = str(meta.get("topic", "")).lower().strip()
            target_topic = str(topic_filter).lower().strip()
            if target_topic not in doc_topic and doc_topic not in target_topic:
                continue # Skip if topics don't align

        # If a document survives all active filter blocks, add it to the final pool
        filtered_docs.append(doc)
        
    # If the user's filter intersections leave you with an empty list, flag it cleanly
    if not filtered_docs:
        return "FILTER_MISMATCH"
        
    return filtered_docs[:4]


    
           
# -----------------------------------------------------------------------------
# 4. PRODUCTION LCEL PIPELINE ASSEMBLE
# -----------------------------------------------------------------------------
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
cloud_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Also update execute_rag_pipeline right below it to handle the flag:
async def execute_rag_pipeline_async(input_data: dict) -> dict | str:
    raw_docs = await hybrid_regex_retriever_async(
        query=input_data["question"],
        filename_filter=input_data.get("filename"),
        topic_filter=input_data.get("topic"),
        date_filter=input_data.get("date")
    )
    
    # Catch the mismatch flag and pass it back cleanly
    if raw_docs == "FILTER_MISMATCH":
        return "FILTER_MISMATCH"
        
    context_str = format_docs(raw_docs)
    chain_runner = custom_prompt | cloud_llm | StrOutputParser()
    answer = await chain_runner.ainvoke({"context": context_str, "question": input_data["question"]})   

    return {
        "answer":answer,
        "context":context_str
    }

# -----------------------------------------------------------------------------
# 5. CORE SYSTEM WRAPPERS FOR APPS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. CORE SYSTEM WRAPPERS FOR APPS
# -----------------------------------------------------------------------------
async def query_rag_system_async(user_query: str, filename_filter=None, topic_filter=None, date_filter=None) -> dict | str:
    """
    Accepts a raw text query string along with manual dropdown sidebar overrides.
    """
    final_filename = filename_filter
    final_topic = topic_filter
    final_date = date_filter

    current_inventory = get_cached_metadata()
    detected = extract_metadata_via_fuzzy_match(user_query, current_inventory)
    
    if not final_filename:
        final_filename = detected.get("filename")
    if not final_topic:
        final_topic = detected.get("topic")
    if not final_date:
        final_date = detected.get("date")
        
    # OPTIMIZATION: Isolate Bare Month and Day roots to assist ChromaDB semantic scoring
    optimized_query = user_query
    if final_date:
        clean_date_text = str(final_date).replace("2025", "").replace("2026", "")
        clean_date_text = clean_date_text.replace("-", " ").replace("/", " ").replace(":", " ").strip()
        
        tokens = re.findall(r'[a-z0-9]+', clean_date_text.lower())
        day_token = next((t for t in tokens if t.isdigit()), "")
        month_token = next((t for t in tokens if not t.isdigit()), "")
        
        # Slice month down to 3 characters (e.g. "january" -> "jan") to guarantee match hits
        if len(month_token) > 3:
            month_token = month_token[:3]
            
        optimized_query = f"{user_query} {month_token} {day_token}".strip()
        
    payload = {
        "question": optimized_query,
        "filename": final_filename,
        "topic": final_topic,
        "date": final_date
    }
    
    # Call the sequential pipeline execution engine directly
    return await execute_rag_pipeline_async(payload)
