"""
Streamlit Web Application for Sustainable Agent Consultant
Provides an interactive interface for analyzing sustainability reports using RAG
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import tempfile

from config import Config
from rag_agent import SustainabilityRAGAgent
from vector_store import VectorStore
from pdf_processor import PDFProcessor

# Page configuration
st.set_page_config(
    page_title="🌱 Sustainable Agent Consultant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = SustainabilityRAGAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🌱 Sustainable Agent Consultant</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Sustainability Report Analysis using RAG Framework</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Control Panel")
        
        # Navigation
        page = st.selectbox(
            "Choose a section:",
            ["🏠 Dashboard", "📄 Upload Reports", "💬 Chat Assistant", "📊 Analytics", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Knowledge base stats
        stats = st.session_state.agent.get_knowledge_base_stats()
        st.markdown("### 📚 Knowledge Base")
        st.metric("Total Documents", stats.get("total_documents", 0))
        if "unique_files" in stats:
            st.metric("Unique Files", stats["unique_files"])
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.agent.clear_conversation_history()
            st.success("Chat history cleared!")
        
        if st.button("🔄 Refresh Stats"):
            st.rerun()
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📄 Upload Reports":
        show_upload_page()
    elif page == "💬 Chat Assistant":
        show_chat_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_dashboard():
    """Display the main dashboard"""
    st.markdown('<div class="sub-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Get knowledge base stats
    stats = st.session_state.agent.get_knowledge_base_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📚 Total Documents",
            value=stats.get("total_documents", 0),
            help="Total number of document chunks in the knowledge base"
        )
    
    with col2:
        st.metric(
            label="📁 Unique Files",
            value=stats.get("unique_files", 0),
            help="Number of unique sustainability reports uploaded"
        )
    
    with col3:
        st.metric(
            label="💬 Conversations",
            value=len(st.session_state.agent.get_conversation_history()),
            help="Total number of conversations with the assistant"
        )
    
    with col4:
        st.metric(
            label="🔍 Embedding Model",
            value="SentenceTransformer",
            help="AI model used for document embeddings"
        )
    
    # Recent activity
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Recent Activity")
        conversation_history = st.session_state.agent.get_conversation_history()
        
        if conversation_history:
            recent_conversations = conversation_history[-5:]  # Last 5 conversations
            for i, conv in enumerate(reversed(recent_conversations)):
                with st.expander(f"💬 {conv.get('analysis_type', 'General')} - {conv.get('timestamp', 'Unknown')[:19]}"):
                    st.write("**Question:**", conv.get('question', 'N/A'))
                    st.write("**Response:**", conv.get('response', 'N/A')[:300] + "..." if len(conv.get('response', '')) > 300 else conv.get('response', 'N/A'))
        else:
            st.info("No recent conversations. Start by uploading a sustainability report and asking questions!")
    
    with col2:
        st.markdown("### 🌱 Sustainability Categories")
        config = Config()
        categories = config.SUSTAINABILITY_CATEGORIES[:8]  # Show first 8 categories
        
        for category in categories:
            st.markdown(f"• {category}")
        
        if len(config.SUSTAINABILITY_CATEGORIES) > 8:
            st.markdown(f"... and {len(config.SUSTAINABILITY_CATEGORIES) - 8} more")

def show_upload_page():
    """Display the file upload interface"""
    st.markdown('<div class="sub-header">📄 Upload Sustainability Reports</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload your sustainability reports (PDF, DOCX, or TXT format) to build your knowledge base.
    The system will automatically extract text, create embeddings, and make the content searchable.
    """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose sustainability report files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files containing sustainability reports"
    )
    
    if uploaded_files:
        st.markdown("### 📋 Upload Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("Company Name (optional)", help="Name of the company for this report")
            report_year = st.number_input("Report Year (optional)", min_value=2000, max_value=2030, value=2023, help="Year of the sustainability report")
        
        with col2:
            industry = st.selectbox(
                "Industry (optional)",
                ["", "Technology", "Manufacturing", "Energy", "Finance", "Healthcare", "Retail", "Transportation", "Other"],
                help="Industry sector for benchmarking"
            )
            report_type = st.selectbox(
                "Report Type (optional)",
                ["", "Annual Sustainability Report", "ESG Report", "Carbon Disclosure", "Integrated Report", "Other"],
                help="Type of sustainability report"
            )
        
        if st.button("🚀 Process Files", type="primary"):
            process_uploaded_files(uploaded_files, {
                "company_name": company_name,
                "report_year": report_year,
                "industry": industry,
                "report_type": report_type,
                "upload_timestamp": datetime.now().isoformat()
            })

def process_uploaded_files(uploaded_files, metadata):
    """Process uploaded files and add them to the knowledge base"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / total_files)
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Add file-specific metadata
            file_metadata = metadata.copy()
            file_metadata.update({
                "original_filename": uploaded_file.name,
                "file_size": uploaded_file.size,
                "file_type": uploaded_file.type
            })
            
            # Process the file
            result = st.session_state.agent.add_report_to_knowledge_base(tmp_file_path, file_metadata)
            results.append((uploaded_file.name, result))
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            results.append((uploaded_file.name, {"success": False, "error": str(e)}))
    
    # Display results
    status_text.text("Processing complete!")
    progress_bar.progress(1.0)
    
    st.markdown("### 📊 Processing Results")
    
    success_count = 0
    for filename, result in results:
        if result.get("success"):
            success_count += 1
            st.markdown(f'<div class="success-message">✅ <strong>{filename}</strong> - Successfully processed {result.get("chunks_added", 0)} chunks</div>', unsafe_allow_html=True)
            
            # Show extracted metrics if available
            if result.get("metrics_extracted"):
                with st.expander(f"📈 Extracted Metrics from {filename}"):
                    metrics = result["metrics_extracted"]
                    for metric_type, values in metrics.items():
                        if values:
                            st.write(f"**{metric_type.replace('_', ' ').title()}:**")
                            for value in values[:3]:  # Show first 3 values
                                st.write(f"• {value[:100]}...")
        else:
            st.markdown(f'<div class="error-message">❌ <strong>{filename}</strong> - {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
    
    if success_count > 0:
        st.balloons()
        st.success(f"Successfully processed {success_count} out of {total_files} files!")
        
        # Update uploaded files list
        st.session_state.uploaded_files.extend([f[0] for f in results if f[1].get("success")])

def show_chat_page():
    """Display the chat interface"""
    st.markdown('<div class="sub-header">💬 Chat with Your Sustainability Assistant</div>', unsafe_allow_html=True)
    
    # Analysis type selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_type = st.selectbox(
            "Choose analysis type:",
            [
                ("general_analysis", "🔍 General Analysis"),
                ("report_summary", "📄 Report Summary"),
                ("carbon_analysis", "🌡️ Carbon Footprint Analysis"),
                ("compliance_check", "✅ Compliance Check"),
                ("benchmarking", "📊 Benchmarking")
            ],
            format_func=lambda x: x[1]
        )
    
    with col2:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.agent.clear_conversation_history()
            st.rerun()
    
    # Additional options for specific analysis types
    additional_context = {}
    
    if analysis_type[0] == "compliance_check":
        framework = st.selectbox(
            "Select reporting framework:",
            ["GRI Standards", "SASB", "TCFD", "CDP", "UN Global Compact", "ISO 14001"]
        )
        additional_context["framework"] = framework
    
    elif analysis_type[0] == "benchmarking":
        industry = st.selectbox(
            "Select industry for benchmarking:",
            ["Technology", "Manufacturing", "Energy", "Finance", "Healthcare", "Retail", "Transportation", "Other"]
        )
        additional_context["industry"] = industry
    
    # Chat interface
    st.markdown("### 💭 Conversation")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask about sustainability reports...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from agent
        with st.spinner("Analyzing sustainability data..."):
            if analysis_type[0] == "report_summary":
                result = st.session_state.agent.generate_report_summary()
            elif analysis_type[0] == "carbon_analysis":
                result = st.session_state.agent.analyze_carbon_footprint()
            elif analysis_type[0] == "compliance_check":
                result = st.session_state.agent.check_compliance(additional_context.get("framework", "GRI Standards"))
            elif analysis_type[0] == "benchmarking":
                result = st.session_state.agent.benchmark_performance(additional_context.get("industry", "General"))
            else:
                result = st.session_state.agent.query_knowledge_base(user_input, analysis_type[0], additional_context)
        
        if result.get("success"):
            response = result["response"]
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            error_msg = f"❌ Error: {result.get('error', 'Unknown error occurred')}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

def show_analytics_page():
    """Display analytics and insights"""
    st.markdown('<div class="sub-header">📊 Analytics & Insights</div>', unsafe_allow_html=True)
    
    # Check if there's data in the knowledge base
    stats = st.session_state.agent.get_knowledge_base_stats()
    
    if stats.get("total_documents", 0) == 0:
        st.warning("No data available. Please upload some sustainability reports first.")
        return
    
    # Analytics options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Quick Analysis")
        
        if st.button("📄 Generate Complete Report Summary", use_container_width=True):
            with st.spinner("Generating comprehensive report summary..."):
                result = st.session_state.agent.generate_report_summary()
                if result.get("success"):
                    st.markdown("#### 📋 Comprehensive Report Summary")
                    st.markdown(result["response"])
                else:
                    st.error(f"Error: {result.get('error')}")
        
        if st.button("🌡️ Carbon Footprint Analysis", use_container_width=True):
            with st.spinner("Analyzing carbon footprint data..."):
                result = st.session_state.agent.analyze_carbon_footprint()
                if result.get("success"):
                    st.markdown("#### 🌡️ Carbon Footprint Analysis")
                    st.markdown(result["response"])
                else:
                    st.error(f"Error: {result.get('error')}")
    
    with col2:
        st.markdown("### ⚖️ Compliance & Benchmarking")
        
        framework = st.selectbox("Select framework for compliance check:", 
                                ["GRI Standards", "SASB", "TCFD", "CDP"])
        
        if st.button(f"✅ Check {framework} Compliance", use_container_width=True):
            with st.spinner(f"Checking compliance with {framework}..."):
                result = st.session_state.agent.check_compliance(framework)
                if result.get("success"):
                    st.markdown(f"#### ✅ {framework} Compliance Check")
                    st.markdown(result["response"])
                else:
                    st.error(f"Error: {result.get('error')}")
        
        industry = st.selectbox("Select industry for benchmarking:", 
                               ["Technology", "Manufacturing", "Energy", "Finance"])
        
        if st.button(f"📊 Benchmark Against {industry}", use_container_width=True):
            with st.spinner(f"Benchmarking against {industry} industry..."):
                result = st.session_state.agent.benchmark_performance(industry)
                if result.get("success"):
                    st.markdown(f"#### 📊 {industry} Industry Benchmarking")
                    st.markdown(result["response"])
                else:
                    st.error(f"Error: {result.get('error')}")
    
    # Sustainability categories insights
    st.markdown("### 🌱 Sustainability Categories Analysis")
    
    config = Config()
    selected_categories = st.multiselect(
        "Select categories to analyze:",
        config.SUSTAINABILITY_CATEGORIES,
        default=config.SUSTAINABILITY_CATEGORIES[:3]
    )
    
    if selected_categories:
        for category in selected_categories:
            with st.expander(f"📈 {category} Analysis"):
                if st.button(f"Analyze {category}", key=f"analyze_{category}"):
                    with st.spinner(f"Analyzing {category}..."):
                        result = st.session_state.agent.get_sustainability_insights(category)
                        if result.get("success"):
                            st.markdown(result["response"])
                        else:
                            st.error(f"Error: {result.get('error')}")

def show_settings_page():
    """Display settings and configuration"""
    st.markdown('<div class="sub-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Enter your OpenAI API key for LLM functionality"
        )
        
        if st.button("💾 Save API Key"):
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("API key saved for this session!")
    
    with col2:
        st.markdown("**Current Configuration:**")
        config = Config()
        st.write(f"• LLM Model: {config.LLM_MODEL}")
        st.write(f"• Embedding Model: {config.EMBEDDING_MODEL}")
        st.write(f"• Temperature: {config.TEMPERATURE}")
        st.write(f"• Max Tokens: {config.MAX_TOKENS}")
    
    # Vector Database Management
    st.markdown("### 🗄️ Vector Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Database Statistics:**")
        stats = st.session_state.agent.get_knowledge_base_stats()
        st.json(stats)
    
    with col2:
        st.markdown("**Database Actions:**")
        
        if st.button("🔄 Refresh Database Stats"):
            st.rerun()
        
        st.markdown("---")
        
        st.warning("⚠️ Dangerous Actions")
        if st.button("🗑️ Reset Vector Database", type="secondary"):
            if st.checkbox("I understand this will delete all uploaded documents"):
                vector_store = VectorStore()
                if vector_store.reset_collection():
                    st.success("Vector database reset successfully!")
                    st.session_state.uploaded_files = []
                    st.rerun()
                else:
                    st.error("Failed to reset vector database")
    
    # Application Info
    st.markdown("### ℹ️ Application Information")
    
    st.markdown("""
    **Sustainable Agent Consultant** is an AI-powered tool for analyzing sustainability reports using:
    
    - **RAG Framework**: Retrieval-Augmented Generation for accurate, context-aware responses
    - **Vector Embeddings**: Semantic search using sentence transformers
    - **Multiple PDF Processing**: Support for various PDF extraction methods
    - **Comprehensive Analysis**: Report summaries, carbon analysis, compliance checks, and benchmarking
    - **Interactive Chat**: Natural language interface for sustainability insights
    
    **Supported File Formats**: PDF, DOCX, TXT
    **Maximum File Size**: 50MB per file
    **Supported Frameworks**: GRI Standards, SASB, TCFD, CDP, UN Global Compact, ISO 14001
    """)

if __name__ == "__main__":
    main()