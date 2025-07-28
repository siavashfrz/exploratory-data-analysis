# 🌱 Sustainable Agent Consultant

An AI-powered sustainability consultant that analyzes sustainability reports (PDF format) using the RAG (Retrieval-Augmented Generation) framework. This comprehensive solution provides intelligent insights, compliance checks, carbon footprint analysis, and benchmarking for sustainability reports.

## 🚀 Features

### 📄 Document Processing
- **Multi-format Support**: PDF, DOCX, and TXT files
- **Advanced PDF Extraction**: Multiple extraction methods (pdfplumber, PyPDF2, pypdf) for maximum text recovery
- **Intelligent Text Chunking**: Smart text segmentation for optimal embedding performance
- **Metadata Extraction**: Automatic extraction of sustainability metrics and KPIs

### 🤖 AI-Powered Analysis
- **RAG Framework**: Retrieval-Augmented Generation for context-aware responses
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Multiple Analysis Types**:
  - General sustainability analysis
  - Comprehensive report summaries
  - Carbon footprint analysis
  - Compliance checking (GRI, SASB, TCFD, CDP, etc.)
  - Industry benchmarking

### 💬 Interactive Interface
- **Streamlit Web App**: User-friendly interface for document upload and analysis
- **Chat Interface**: Natural language conversations about sustainability data
- **Dashboard**: Real-time statistics and insights
- **Analytics Page**: Quick analysis tools and category-specific insights

### 🔍 Advanced Capabilities
- **Multi-Framework Compliance**: Support for GRI Standards, SASB, TCFD, CDP, UN Global Compact, ISO 14001
- **Industry Benchmarking**: Compare performance against industry standards
- **Category Analysis**: Focused analysis on specific sustainability categories
- **Conversation History**: Track and review previous analyses

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for LLM functionality)

### Quick Setup
1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd sustainable-agent-consultant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file and add your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

### Alternative Installation with Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## 🎯 Usage

### 1. Upload Reports
- Navigate to the "📄 Upload Reports" section
- Upload your sustainability reports (PDF, DOCX, or TXT)
- Add metadata (company name, year, industry, report type)
- Click "🚀 Process Files" to add them to the knowledge base

### 2. Chat with the Assistant
- Go to "💬 Chat Assistant"
- Select your analysis type:
  - **General Analysis**: Ask any sustainability-related questions
  - **Report Summary**: Get comprehensive report overviews
  - **Carbon Analysis**: Focus on carbon emissions and climate data
  - **Compliance Check**: Verify against sustainability frameworks
  - **Benchmarking**: Compare against industry standards
- Start chatting with natural language queries

### 3. Analytics & Insights
- Visit "📊 Analytics" for quick analysis tools
- Generate complete report summaries
- Perform carbon footprint analyses
- Check compliance against frameworks
- Benchmark performance against industries
- Analyze specific sustainability categories

### 4. Dashboard Monitoring
- Monitor knowledge base statistics
- View recent conversation history
- Track uploaded files and processing status

## 📊 Supported Analysis Types

### 🔍 General Analysis
Ask any questions about your sustainability reports:
- "What are the main environmental initiatives?"
- "How has water consumption changed over time?"
- "What are the key social responsibility programs?"

### 📄 Report Summary
Get comprehensive overviews including:
- Executive summary
- Key environmental metrics
- Social responsibility initiatives
- Governance and strategy
- Achievements and challenges
- Actionable recommendations

### 🌡️ Carbon Footprint Analysis
Detailed carbon emissions analysis:
- Scope 1, 2, and 3 emissions breakdown
- Historical trends and comparisons
- Reduction strategies and initiatives
- Science-based targets progress
- Net-zero commitments

### ✅ Compliance Checking
Verify reports against frameworks:
- **GRI Standards**: Global Reporting Initiative
- **SASB**: Sustainability Accounting Standards Board
- **TCFD**: Task Force on Climate-related Financial Disclosures
- **CDP**: Carbon Disclosure Project
- **UN Global Compact**: United Nations principles
- **ISO 14001**: Environmental management systems

### 📊 Industry Benchmarking
Compare performance against sectors:
- Technology
- Manufacturing
- Energy
- Finance
- Healthcare
- Retail
- Transportation

## 🏗️ Architecture

### Core Components

1. **PDF Processor** (`pdf_processor.py`)
   - Multi-method text extraction
   - Text preprocessing and cleaning
   - Intelligent chunking
   - Metadata extraction

2. **Vector Store** (`vector_store.py`)
   - ChromaDB integration
   - Sentence transformer embeddings
   - Semantic search capabilities
   - Document management

3. **RAG Agent** (`rag_agent.py`)
   - LangChain integration
   - OpenAI GPT models
   - Prompt engineering
   - Context-aware responses

4. **Streamlit App** (`streamlit_app.py`)
   - Web interface
   - File upload handling
   - Interactive chat
   - Analytics dashboard

### Data Flow
```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store
                                                           ↓
User Query → Embedding → Similarity Search → Context Retrieval → LLM → Response
```

## ⚙️ Configuration

### Environment Variables
Configure the application using the `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (uses defaults if not set)
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TEMPERATURE=0.7
MAX_TOKENS=2048
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.7
```

### Customization
- **Models**: Change LLM or embedding models in `config.py`
- **Categories**: Modify sustainability categories in `config.py`
- **Prompts**: Customize analysis prompts in `rag_agent.py`
- **UI**: Modify the Streamlit interface in `streamlit_app.py`

## 🎨 Sustainability Categories

The system analyzes reports across 15+ sustainability categories:

- Environmental Impact
- Carbon Emissions
- Energy Consumption
- Water Usage
- Waste Management
- Social Responsibility
- Corporate Governance
- Supply Chain Sustainability
- Biodiversity
- Climate Change Adaptation
- Renewable Energy
- Circular Economy
- ESG Metrics
- Stakeholder Engagement
- Sustainability Goals

## 📈 Performance Features

### Optimization
- **Parallel Processing**: Concurrent file processing
- **Caching**: Vector embeddings cached for quick retrieval
- **Batch Processing**: Efficient bulk operations
- **Smart Chunking**: Context-aware text segmentation

### Scalability
- **Vector Database**: Persistent ChromaDB storage
- **Memory Management**: Efficient embedding handling
- **File Size Limits**: Configurable processing limits
- **Progress Tracking**: Real-time processing feedback

## 🔧 API Usage

### Programmatic Access
```python
from rag_agent import SustainabilityRAGAgent

# Initialize agent
agent = SustainabilityRAGAgent()

# Add report to knowledge base
result = agent.add_report_to_knowledge_base("report.pdf")

# Query the knowledge base
response = agent.query_knowledge_base("What are the carbon reduction targets?")

# Generate report summary
summary = agent.generate_report_summary()

# Analyze carbon footprint
carbon_analysis = agent.analyze_carbon_footprint()

# Check compliance
compliance = agent.check_compliance("GRI Standards")

# Benchmark performance
benchmark = agent.benchmark_performance("Technology")
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Add tests if applicable
6. Submit a pull request

### Code Structure
```
├── config.py              # Configuration settings
├── pdf_processor.py       # PDF text extraction
├── vector_store.py        # Vector database operations
├── rag_agent.py          # Main RAG implementation
├── streamlit_app.py      # Web interface
├── requirements.txt      # Dependencies
├── .env.example         # Environment template
└── README.md           # Documentation
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

1. **API Key Errors**
   - Ensure OpenAI API key is set in `.env` file
   - Check API key validity and credits

2. **PDF Processing Issues**
   - Try different PDF extraction methods
   - Ensure PDFs contain extractable text (not just images)

3. **Memory Issues**
   - Reduce chunk size for large documents
   - Process files individually for very large reports

4. **Embedding Errors**
   - Check internet connection for model downloads
   - Ensure sufficient disk space for model storage

### Getting Help
- Create an issue on GitHub
- Check the documentation
- Review example usage in the code

## 🌟 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web interface
- [LangChain](https://langchain.com/) - LLM framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [OpenAI](https://openai.com/) - Language models

---

**🌱 Making sustainability reporting smarter with AI**