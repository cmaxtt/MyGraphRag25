import streamlit as st
import os
import tempfile
from ingest import Ingestor
from search import SearchEngine

st.set_page_config(page_title="Local Hybrid GraphRAG", layout="wide")

st.title("🌐 Local Hybrid GraphRAG System")
st.markdown("Build relationships from your documents and query them with Hybrid (Vector + Graph) Retrieval.")

@st.cache_resource
def get_ingestor():
    return Ingestor()

@st.cache_resource
def get_search_engine():
    return SearchEngine()

# Initialize components
ingestor = get_ingestor()
search_engine = get_search_engine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for information and file upload
with st.sidebar:
    st.title("⚙️ System Status")
    st.info(f"**LLM Model:** `{search_engine.llm_model}`\n\n**Embed Model:** `{search_engine.embed_model}`")
    st.divider()
    
    st.header("📁 Document Ingestion")
    uploaded_files = st.file_uploader("Upload Documents (PDF, DOCX, XLSX, CSV)", accept_multiple_files=True)
    
    if st.button("🚀 Process Documents") and uploaded_files:
        with st.status("Processing documents...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                st.write(f"Processing `{uploaded_file.name}`...")
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    ingestor.process_file(tmp_path)
                    st.success(f"Finished `{uploaded_file.name}`")
                except Exception as e:
                    st.error(f"Error processing `{uploaded_file.name}`: {e}")
                finally:
                    os.unlink(tmp_path)
            status.update(label="Ingestion Complete!", state="complete", expanded=False)

    if st.button("🗑️ Reset Databases"):
        # This is a dangerous but useful feature for dev
        st.warning("Feature not fully implemented via UI for safety. See README.md for manual reset.")

# Chat Interface
st.header("💬 Chat with your Knowledge Base")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                answer = search_engine.hybrid_search(prompt)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error during search: {e}")

# Cleanup on app close (if possible) or session end
# Note: Database handles are managed in session state, 
# but permanent close would happen when the process dies.
