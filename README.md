# Multimodal Temporal RAG Engine

A Retrieval-Augmented Generation (RAG) system designed to parse, index, filter, and query complex scanned handwritten notebooks and technical study materials. 

By leveraging a Parent-Child chunking architecture, this application ensures the Large Language Model always reviews full, coherent pages instead of fragmented text snippets.

---

## Key Architectural Features

* Multimodal Extraction: Handles unstructured layout structures from scanned handwritten PDF volumes mapped with metadata footprints.
* Parent Document Retrieval: Small text fragments are embedded via a local model and indexed inside ChromaDB, while full-page layouts are mapped onto local hard drive storage utilizing a key-value file store.
* Deterministic Metadata Filtering: Features a custom hybrid routing function that maps user-selected boundaries directly into strict database operational conditions.
* Cloud Synthesis: Connects seamlessly to llama-3.3-70b-versatile via Groq using LangChain Expression Language for precise, zero-temperature factual answers.
* Interactive UI: A lightweight user dashboard equipped with sidebar filter selectors for customized, multi-tier notebook querying.

---

## Tech Stack and Dependencies

* Orchestration: LangChain, LCEL
* Vector Database: ChromaDB
* Embeddings: HuggingFace (Local Execution)
* Large Language Model: Groq Cloud API (llama-3.3-70b-versatile)
* User Interface: Streamlit Framework
* Environment Management: Python, Conda, python-dotenv

---

## Project Structure

* .env - Secure API credentials
* .gitignore - Excludes database layers and keys
* app.py - Streamlit frontend interface
* rag_backend.py - Core LCEL pipeline and data models
* my_final_handwritten_notes_rag.ipynb - Dev notebook and testing environment
* README.md - Portfolio overview
