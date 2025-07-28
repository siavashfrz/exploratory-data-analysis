"""
Configuration settings for Sustainable Agent Consultant
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Sustainable Agent Consultant"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Model configurations
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048
    
    # Vector database settings
    VECTOR_DB_PATH = "./vector_db"
    COLLECTION_NAME = "sustainability_reports"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # PDF processing settings
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FORMATS = ['.pdf', '.txt', '.docx']
    
    # Retrieval settings
    RETRIEVAL_TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Streamlit settings
    PAGE_TITLE = "Sustainable Agent Consultant"
    PAGE_ICON = "🌱"
    LAYOUT = "wide"
    
    # Sustainability categories for analysis
    SUSTAINABILITY_CATEGORIES = [
        "Environmental Impact",
        "Carbon Emissions",
        "Energy Consumption",
        "Water Usage",
        "Waste Management",
        "Social Responsibility",
        "Corporate Governance",
        "Supply Chain Sustainability",
        "Biodiversity",
        "Climate Change Adaptation",
        "Renewable Energy",
        "Circular Economy",
        "ESG Metrics",
        "Stakeholder Engagement",
        "Sustainability Goals"
    ]
    
    # Report analysis prompts
    ANALYSIS_PROMPTS = {
        "summary": "Provide a comprehensive summary of the sustainability report focusing on key environmental and social initiatives.",
        "carbon_footprint": "Analyze the carbon footprint and greenhouse gas emissions data from the report.",
        "targets": "Identify and evaluate the sustainability targets and goals mentioned in the report.",
        "performance": "Assess the organization's sustainability performance against industry benchmarks.",
        "recommendations": "Provide actionable recommendations for improving sustainability practices based on the report."
    }