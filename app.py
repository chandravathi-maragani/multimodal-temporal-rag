import streamlit as st
from dotenv import load_dotenv
from rag_backend import final_rag_chain  # <-- Imports your exact notebook chain!

# 1. Page Config & Environment Initialization
st.set_page_config(page_title="Handwritten Notes Search Engine", page_icon="📝")
load_dotenv()

st.title("📝 Handwritten Notes Smart RAG Search Engine")
st.markdown("Query your scanned study materials with deterministic metadata filters and semantic hybrid extraction.")

# 2. Sidebar Dropdowns Matching Your Specific Database Parameters
st.sidebar.header("🔍 Search Filters")

filename = st.sidebar.selectbox(
    "Source Document File:", 
    ["All Files", "Running_notes_part_1", "Running_notes_part_2"]
)

topic = st.sidebar.selectbox(
    "Notebook Topic Area:", 
    ["All Topics", "List Continuation", "Agentic_AI", "Object_Oriented_Programming"]
)

# 3. Main Query Bar Console
query = st.text_input("💬 What would you like to search across your notes?")

# 4. Trigger Your Notebook's LCEL Payload
if st.button("Search Notes", type="primary") and query:
    with st.spinner("Searching and synthesizing..."):
        # Map UI dropdown options directly to your backend lambda expectations
        payload = {
            "question": query,
            "filename": None if filename == "All Files" else filename,
            "topic": None if topic == "All Topics" else topic
        }
        
        try:
            # Fire your notebook's chain exactly as you tested it!
            response = final_rag_chain.invoke(payload)
            st.subheader("💡 Synthesized Intelligence Answer:")
            st.write(response)
            
        except Exception as e:
            st.error(f"An execution issue popped up: {e}")