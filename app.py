import os
import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA

from document_processors import load_multimodal_data, load_data_from_directory
from utils import set_environment_variables

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize settings
def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

# Create index from documents
def create_index(documents):
    vector_store = MilvusVectorStore(
            host = "127.0.0.1",
            port = 19530,
            dim = 1024
    )
    # vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True) #For CPU only vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


# Main function to run the Streamlit app

def main():
    set_environment_variables()
    initialize_settings()

    # Add custom CSS to make left column fixed
st.markdown("""
<style>

    /* Make LEFT column fixed */
    [data-testid="stHorizontalBlock"] > div:first-child {
        position: sticky !important;
        top: 0 !important;
        height: 100vh !important;
        overflow-y: auto !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        z-index: 999 !important;
        padding-right: 10px;
        border-right: 1px solid #e2e2e2;
    }

      /* RIGHT column scrolls normally */
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {
        height: 100vh !important;
        overflow-y: auto !important;
        padding-bottom: 120px; /* Space so chat messages don't cover input */
    }

    /* FIX chat input at bottom */
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 35% !important;
        width: 60% !important;
        z-index: 9999 !important;
        background: white !important;
        padding: 15px 10px;
        border-top: 1px solid #ddd;
    }

    /* Optional: Reduce bottom padding of main area */
    .main .block-container {
        padding-bottom: 160px !important;
    }

</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 2])

    # ---------- LEFT SIDE (now sticky/fixed) ----------
with col1:
        st.image("logo.jpeg", width=150)
        st.title("Multimodal RAG")

        input_method = st.radio("Choose input method:", ("Upload Files", "Enter Directory Path"))

        if input_method == "Upload Files":
            uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True)

            if uploaded_files and st.button("Process Files"):
                with st.spinner("Processing files..."):
                    documents = load_multimodal_data(uploaded_files)
                    st.session_state['index'] = create_index(documents)
                    st.session_state['history'] = []
                st.success("Files processed and index created!")

        else:
            directory_path = st.text_input("Enter directory path:")

            if directory_path and st.button("Process Directory"):
                if os.path.isdir(directory_path):
                    with st.spinner("Processing directory..."):
                        documents = load_data_from_directory(directory_path)
                        st.session_state['index'] = create_index(documents)
                        st.session_state['history'] = []
                    st.success("Directory processed and index created!")
                else:
                    st.error("Invalid directory path. Please enter a valid path.")

    # ---------- RIGHT SIDE (scrollable chat area) ----------
with col2:
        if 'index' in st.session_state:

            # Center-align title

            st.markdown("""
                <h1 style='text-align: center;'>Chat</h1>

<style>
    /* Make the full right column stretch like ChatGPT */
    .block-container {
        padding-bottom: 120px;  /* leaves room for input bar */
    }

    /* Fixed input box */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        right: 150px;
        width: 55%;
        z-index: 999;
    
    }
</style>

            """, unsafe_allow_html=True)
           # st.markdown(
               # "<h1 style='text-align: center; padding-bottom:400px;'>Chat</h1>",
               # unsafe_allow_html=True
          #  )

            if 'history' not in st.session_state:
                st.session_state['history'] = []

            query_engine = st.session_state['index'].as_query_engine(
                similarity_top_k=20,
                streaming=True
            )

            # CHAT HISTORY
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
# AUTO SCROLL TO BOTTOM
            scroll_js = """
            <script>
              var chatBox = window.parent.document.querySelector('.block-container');
              chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
            </script>
            """
            st.markdown(scroll_js, unsafe_allow_html=True)

            user_input = st.chat_input("Enter your query:")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)

                st.session_state['history'].append(
                    {"role": "user", "content": user_input}
                )

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    response = query_engine.query(user_input)

                    for token in response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                st.session_state['history'].append(
                    {"role": "assistant", "content": full_response}
                )

            if st.button("Clear Chat"):
                st.session_state['history'] = []
                st.rerun()

if __name__ == "__main__":
    main()
