#!/usr/bin/env python3
"""
Test script for Sustainable Agent Consultant
Validates the functionality without requiring an OpenAI API key initially
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from pdf_processor import PDFProcessor
from vector_store import VectorStore

def create_sample_sustainability_text():
    """Create sample sustainability report text for testing"""
    return """
    SAMPLE SUSTAINABILITY REPORT 2024
    
    EXECUTIVE SUMMARY
    This sustainability report outlines our commitment to environmental responsibility and social impact.
    
    CARBON FOOTPRINT
    Our organization achieved a 25% reduction in carbon emissions compared to 2023.
    Total CO2 emissions: 1,250 metric tons
    Renewable energy usage: 75% of total energy consumption
    
    WASTE MANAGEMENT
    We implemented comprehensive waste reduction programs:
    - 80% waste diversion from landfills
    - 95% recycling rate for office materials
    - Zero plastic waste in our cafeterias
    
    SOCIAL RESPONSIBILITY
    Employee satisfaction rate: 92%
    Diversity and inclusion initiatives reached 1,000+ employees
    Community investment: $500,000 in local environmental projects
    
    WATER CONSERVATION
    Water usage reduced by 30% through efficiency programs
    Rainwater harvesting system installed
    Water recycling system covers 60% of facility needs
    
    SUSTAINABLE SUPPLY CHAIN
    85% of suppliers meet our sustainability criteria
    Local sourcing increased by 40%
    Sustainable packaging adoption: 90%
    
    FUTURE COMMITMENTS
    Target net-zero emissions by 2030
    100% renewable energy by 2028
    Zero waste to landfill by 2027
    """

def test_pdf_processor():
    """Test PDF processing functionality"""
    print("🧪 Testing PDF Processor...")
    
    try:
        processor = PDFProcessor()
        
        # Create a temporary text file to simulate processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(create_sample_sustainability_text())
            temp_file = f.name
        
        # Test text processing
        result = processor.process_file(temp_file)
        if result.get('status') != 'success':
            raise Exception(f"Processing failed: {result.get('error', 'Unknown error')}")
        
        chunks = result['chunks']
        print(f"✅ Successfully processed file into {len(chunks)} chunks")
        
        # Test sustainability metrics extraction
        metrics = processor.extract_sustainability_metrics(result['processed_text'])
        print(f"✅ Extracted {sum(len(v) for v in metrics.values())} sustainability metrics")
        
        # Cleanup
        os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ PDF Processor test failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\n🧪 Testing Vector Store...")
    
    try:
        vector_store = VectorStore()
        
        # Test initialization
        print("✅ Vector store initialized successfully")
        
        # Test embedding generation
        sample_text = "Carbon emissions reduced by 25% through renewable energy initiatives"
        embeddings = vector_store.generate_embeddings([sample_text])
        print(f"✅ Generated embedding with dimension: {len(embeddings[0])}")
        
        # Test adding documents
        sample_content = create_sample_sustainability_text().strip()
        sample_docs = [sample_content]
        sample_metadata = [{
            'filename': 'sample_report.txt',
            'chunk_id': 0,
            'company': 'Test Company'
        }]
        
        success = vector_store.add_documents(sample_docs, sample_metadata)
        if success:
            print("✅ Successfully added documents to vector store")
        
        # Test similarity search
        results = vector_store.search_similar(
            "carbon emissions reduction", 
            n_results=3
        )
        print(f"✅ Similarity search returned {len(results['documents'])} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector Store test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration...")
    
    try:
        config = Config()
        print(f"✅ Embedding model: {config.EMBEDDING_MODEL}")
        print(f"✅ Chunk size: {config.CHUNK_SIZE}")
        print(f"✅ Vector DB path: {config.VECTOR_DB_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌱 SUSTAINABLE AGENT CONSULTANT - TEST SUITE")
    print("=" * 55)
    
    # Test configuration
    config_success = test_config()
    
    # Test PDF processor
    pdf_success = test_pdf_processor()
    
    # Test vector store
    vector_success = test_vector_store()
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 TEST SUMMARY")
    print("=" * 55)
    print(f"Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"PDF Processor: {'✅ PASS' if pdf_success else '❌ FAIL'}")
    print(f"Vector Store:  {'✅ PASS' if vector_success else '❌ FAIL'}")
    
    if all([config_success, pdf_success, vector_success]):
        print("\n🎉 All tests passed! The Sustainable Agent Consultant is ready to use.")
        print("\n📋 NEXT STEPS:")
        print("1. Add your OpenAI API key to the .env file")
        print("2. Run the Streamlit app: streamlit run streamlit_app.py")
        print("3. Upload PDF sustainability reports for analysis")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return all([config_success, pdf_success, vector_success])

if __name__ == "__main__":
    main()