"""
Streamlit UI for Agentic RAG Chatbot with Groq Integration
Multi-format document Q&A with MCP
"""

import streamlit as st
import sys
import os
import time
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coordinator_agent import CoordinatorAgent
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from mcp.message_protocol import message_bus

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Chatbot with Groq",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize agents (singleton pattern)
@st.cache_resource
def initialize_agents():
    """Initialize all agents with Groq configuration"""
    coordinator = CoordinatorAgent()
    ingestion = IngestionAgent()
    retrieval = RetrievalAgent()
    llm_response = LLMResponseAgent(model_name="llama3-8b-8192")
    
    return coordinator, ingestion, retrieval, llm_response

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

def display_api_status(llm_agent):
    """Display API configuration status"""
    model_info = llm_agent.get_model_info()
    
    if model_info['provider'] == 'Groq':
        st.success(f"✅ Connected to Groq ({model_info['model_name']})")
    else:
        st.warning("⚠️ Using fallback responses - Groq API not configured")
    
    return model_info

def main():
    # Initialize agents
    coordinator, ingestion, retrieval, llm_response = initialize_agents()
    
    # Header
    st.title("🚀 Agentic RAG Chatbot")
    st.markdown("**Multi-Format Document Q&A using Groq's Llama3-8B via LangChain**")
    
    # Display API status at the top
    model_info = display_api_status(llm_response)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("🔑 Groq API Setup")
        
        # Check if API key is already set
        # Always assume API key is in .env
        current_api_key = os.getenv('GROQ_API_KEY', '')

        if current_api_key:
            st.success("✅ Groq API key loaded from environment")
            masked_key = current_api_key[:8] + "..." + current_api_key[-4:]
            st.code(f"Key: {masked_key}")
        else:
            st.error("❌ GROQ_API_KEY is missing from environment. Please check your .env file.")
            st.stop()

        
        # Model Configuration
        st.subheader("🤖 Model Settings")
        st.info(f"**Model:** {model_info['model_name']}")
        st.info(f"**Provider:** {model_info['provider']}")
        st.info(f"**Status:** {model_info['status']}")
        
        st.divider()
        
        # Document Upload Section
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'pptx', 'csv', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, PPTX, CSV, TXT, Markdown"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Read file content
                        file_content = uploaded_file.read()
                        
                        # Process document
                        trace_id = coordinator.process_document_upload(
                            uploaded_file.name, 
                            file_content
                        )
                        
                        # Wait for processing to complete
                        result = coordinator.wait_for_result(trace_id, timeout=20)
                        
                        if result and result['status'] == 'completed':
                            st.session_state.uploaded_files.append({
                                'name': uploaded_file.name,
                                'size': len(file_content),
                                'type': uploaded_file.type,
                                'trace_id': trace_id,
                                'status': 'success'
                            })
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            error_msg = result['result']['error'] if result else "Processing timeout"
                            st.error(f"❌ Failed to process {uploaded_file.name}: {error_msg}")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Documents")
            for file_info in st.session_state.uploaded_files:
                with st.container():
                    st.write(f"📄 **{file_info['name']}**")
                    st.caption(f"Size: {file_info['size']:,} bytes")
        
        st.divider()
        
        # System Statistics
        st.subheader("📊 System Statistics")
        stats = coordinator.get_system_stats()
        retrieval_stats = retrieval.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Requests", stats['total_requests'])
            st.metric("Documents", retrieval_stats['total_documents'])
        with col2:
            st.metric("Total Chunks", retrieval_stats['total_chunks'])
            st.metric("Vectors", retrieval_stats['total_vectors'])
        
        # Clear session button
        if st.button("🧹 Clear Session", type="secondary"):
            coordinator.clear_session()
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            st.success("Session cleared!")
            st.experimental_rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Show setup instructions if no API key
    if not os.getenv('GROQ_API_KEY'):
        st.info("""
        🔧 **Setup Instructions:**
        1. Get a free API key from [Groq Console](https://console.groq.com/)
        2. Enter it in the sidebar, or add `GROQ_API_KEY=your_key_here` to a `.env` file
        3. Upload some documents
        4. Start asking questions!
        """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "sources" in message:
                st.markdown(message["content"])
                
                # Display sources in a more compact format
                if message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])} found)"):
                        for i, source in enumerate(message["sources"]):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{source['filename']}**")
                            with col2:
                                st.write(f"Chunk: {source['chunk_index']}")
                            with col3:
                                st.write(f"Score: {source['similarity_score']}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are uploaded
        if not st.session_state.uploaded_files:
            st.warning("⚠️ Please upload some documents first!")
            return
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing documents and generating response..."):
                # Send query to coordinator
                trace_id = coordinator.process_user_query(prompt, top_k=5)
                
                # Wait for response with longer timeout for Groq processing
                result = coordinator.wait_for_result(trace_id, timeout=30)
                
                if result and result['status'] == 'completed' and result['result']['success']:
                    response_data = result['result']
                    response_text = response_data['answer']
                    sources = response_data.get('sources', [])
                    
                    st.markdown(response_text)
                    
                    # Display sources in a compact format
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)} found)"):
                            for i, source in enumerate(sources):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.write(f"**{source['filename']}**")
                                with col2:
                                    st.write(f"Chunk: {source['chunk_index']}")
                                with col3:
                                    st.write(f"Score: {source['similarity_score']}")
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                    
                else:
                    error_msg = "Sorry, I encountered an error processing your question."
                    if result and 'result' in result and 'error' in result['result']:
                        error_msg = f"Error: {result['result']['error']}"
                    
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Footer with additional information
    with st.expander("🔧 Advanced Settings & Debug"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Variables")
            st.info("Create a `.env` file with:")
            st.code("GROQ_API_KEY=your_groq_api_key_here")
            
        with col2:
            st.subheader("Model Information")
            st.json(model_info)
        
        st.subheader("MCP Message History (Recent)")
        messages = message_bus.get_message_history()
        
        if messages:
            # Show recent messages
            recent_messages = messages[-5:]  # Last 5 messages
            for msg in recent_messages:
                st.json({
                    "timestamp": msg.timestamp,
                    "sender": msg.sender,
                    "receiver": msg.receiver,
                    "type": msg.type,
                    "trace_id": msg.trace_id,
                })
        else:
            st.write("No messages yet")

if __name__ == "__main__":
    main()