import streamlit as st
from dotenv import load_dotenv
import rag_backend  # Import your cleaned backend script

# 1. Page Config & Environment Initialization
st.set_page_config(page_title="Handwritten Notes Search Engine", page_icon="📝", layout="wide")
load_dotenv()

st.title("📝 Handwritten Notes Smart RAG Search Engine")
st.markdown("Query your scanned study materials with automatic fuzzy extraction or explicit sidebar criteria.")

# 2. Sidebar Dropdowns - Dynamically Loaded from live data
st.sidebar.header("🔍 Manual Search Filters")
st.sidebar.markdown("Use these dropdowns to filter down explicitly if you prefer.")

# Fetch live inventory metadata lists cleanly from the backend file load
db_files = ["All Files"] + rag_backend.metadata_directory.get("filenames", [])
db_topics = ["All Topics"] + [t.title() for t in rag_backend.metadata_directory.get("topics", [])]
db_dates = ["All Dates"] + rag_backend.metadata_directory.get("dates", [])

selected_file = st.sidebar.selectbox("Source Document File:", db_files)
selected_topic = st.sidebar.selectbox("Notebook Topic Area:", db_topics)
selected_date = st.sidebar.selectbox("Specific Session Date:", db_dates)

# Normalize layout selections for the filter matrix strings
ui_file = None if selected_file == "All Files" else selected_file
ui_topic = None if selected_topic == "All Topics" else selected_topic.lower()
ui_date = None if selected_date == "All Dates" else selected_date

# 3. Main Query Bar Console
query = st.text_input("💬 What would you like to search across your notes?", 
                      placeholder="e.g., Explain activation functions from running_notes_part_1")

# 4. Trigger Your Automated Backend Pipeline
if st.button("Search Notes", type="primary") and query:
    with st.spinner("Analyzing criteria and executing vector search..."):
        try:
            # 1. Peek inside the fuzzy matcher for the UI diagnostic block
            detected = rag_backend.extract_metadata_via_fuzzy_match(query, rag_backend.metadata_directory)
            
            # 2. Display the visual feedback box showing what the text query caught
            if any(detected.values()):
                with st.expander("🔍 Natural Language Filter Matrix Detected", expanded=True):
                    if ui_file or detected.get("filename"):
                        st.markdown(f"📁 **File Target:** `{ui_file if ui_file else detected['filename']}` "
                                    f"{'(Sidebar Override)' if ui_file else '(Auto-Extracted)'}")
                    if ui_topic or detected.get("topic"):
                        st.markdown(f"🏷️ **Topic Target:** `{ui_topic.title() if ui_topic else detected['topic'].title()}` "
                                    f"{'(Sidebar Override)' if ui_topic else '(Auto-Extracted)'}")
                    if ui_date or detected.get("date"):
                        st.markdown(f"📅 **Date Target:** `{ui_date if ui_date else detected['date']}` "
                                    f"{'(Sidebar Override)' if ui_date else '(Auto-Extracted)'}")

            # 3. Execute backend search via single system route wrapper
            response = rag_backend.query_rag_system(
                user_query=query,
                filename_filter=ui_file,
                topic_filter=ui_topic,
                date_filter=ui_date
            )
            
            # 4. Render output cleanly based on the return flag
            st.markdown("---")
            if response == "FILTER_MISMATCH":
                st.warning(
                    f"⚠️ **Search Scope Mismatch:** The requested topic or query context is not available in the "
                    f"currently selected file filter (`{selected_file}`). "
                    f"Please switch the sidebar dropdown back to **'All Files'** to search across your whole notebook collection!"
                )
            else:
                st.subheader("💡 Synthesized Intelligence Answer:")
                st.write(response)
            
        except Exception as e:
            st.error(f"An execution issue popped up: {e}")
