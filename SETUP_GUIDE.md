# 🌱 Sustainable Agent Consultant - Setup Guide

This guide will help you set up and run the Sustainable Agent Consultant application.

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (required for AI functionality)
- At least 4GB of RAM (for embedding models)
- 2GB of free disk space

## 🚀 Quick Start

### 1. Clone or Download the Project

If you haven't already, ensure all the project files are in your working directory.

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file and add your OpenAI API key
nano .env  # or use your preferred text editor
```

Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 5. Test the Installation

```bash
python test_agent.py
```

If all tests pass, you're ready to go!

### 6. Run the Application

#### Option A: Using the Startup Script
```bash
chmod +x run_app.sh
./run_app.sh
```

#### Option B: Direct Streamlit Run
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔧 Configuration Options

### Environment Variables

You can customize the application behavior by modifying the `.env` file:

```bash
# AI Model Configuration
LLM_MODEL=gpt-3.5-turbo              # or gpt-4 for better results
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TEMPERATURE=0.7                       # AI response creativity (0.0-1.0)
MAX_TOKENS=2048                       # Maximum response length

# Document Processing
CHUNK_SIZE=1000                       # Text chunk size for embedding
CHUNK_OVERLAP=200                     # Overlap between chunks
MAX_FILE_SIZE_MB=50                   # Maximum upload file size

# Vector Database
VECTOR_DB_PATH=./vector_db           # Local database path
COLLECTION_NAME=sustainability_reports
RETRIEVAL_TOP_K=5                    # Number of relevant chunks to retrieve
SIMILARITY_THRESHOLD=0.7             # Minimum similarity score
```

## 📱 Using the Application

### 1. Upload Documents

- Use the file uploader to upload PDF sustainability reports
- Supported formats: PDF, DOCX, TXT
- Maximum file size: 50MB (configurable)

### 2. Ask Questions

Once documents are uploaded, you can ask questions like:
- "What are the carbon emission targets?"
- "Summarize the waste management initiatives"
- "How does this compare to industry benchmarks?"
- "What certifications does the company have?"

### 3. View Analytics

The application provides:
- Document processing statistics
- Sustainability metrics extraction
- Visual charts and graphs
- Compliance analysis

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. OpenAI API Errors
- Check your API key is correct
- Ensure you have sufficient OpenAI credits
- Verify your API key has the required permissions

#### 3. Memory Issues
If you encounter memory issues:
- Close other applications
- Use a smaller embedding model
- Reduce `CHUNK_SIZE` in configuration

#### 4. PDF Processing Issues
- Ensure PDFs are not password-protected
- Try converting to text format first
- Check file size is within limits

### Debug Mode

Run with debug logging:
```bash
export LOG_LEVEL=DEBUG
python test_agent.py
```

## 🚀 Advanced Usage

### Programmatic API

You can use the components programmatically:

```python
from rag_agent import SustainabilityRAGAgent
from pdf_processor import PDFProcessor

# Initialize agent
agent = SustainabilityRAGAgent()

# Process a document
processor = PDFProcessor()
result = processor.process_file("sustainability_report.pdf")

# Add to knowledge base
agent.add_document(result['chunks'], "company_report.pdf")

# Ask questions
response = agent.query("What are the carbon reduction targets?")
print(response)
```

### Batch Processing

Process multiple files:
```python
import os
from pathlib import Path

pdf_directory = Path("./pdf_reports")
for pdf_file in pdf_directory.glob("*.pdf"):
    result = processor.process_file(str(pdf_file))
    agent.add_document(result['chunks'], pdf_file.name)
```

## 📊 Performance Optimization

### For Better Performance:
1. Use GPU acceleration (if available)
2. Increase `CHUNK_SIZE` for longer documents
3. Use `gpt-4` for more accurate analysis
4. Pre-process documents in batch

### For Lower Resource Usage:
1. Use smaller embedding models
2. Reduce `MAX_TOKENS`
3. Lower `RETRIEVAL_TOP_K`
4. Process documents one at a time

## 🔒 Security Considerations

- Keep your OpenAI API key secure
- Don't commit `.env` file to version control
- Be aware that documents are processed locally but queries go to OpenAI
- Consider using local LLMs for sensitive documents

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Run `python test_agent.py` to verify setup
3. Check the logs for error messages
4. Ensure all dependencies are correctly installed

## 🎯 Next Steps

Once you have the basic setup working:
1. Try uploading real sustainability reports
2. Experiment with different types of questions
3. Explore the analytics features
4. Customize the configuration for your needs

Happy analyzing! 🌱