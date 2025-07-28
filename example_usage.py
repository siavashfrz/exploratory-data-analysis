"""
Example Usage Script for Sustainable Agent Consultant
Demonstrates how to use the RAG agent programmatically for sustainability report analysis
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

# Import the main components
from config import Config
from rag_agent import SustainabilityRAGAgent
from pdf_processor import PDFProcessor
from vector_store import VectorStore


def main():
    """Main function demonstrating various use cases"""
    
    print("🌱 Sustainable Agent Consultant - Example Usage")
    print("=" * 50)
    
    # Initialize the RAG agent
    print("\n1. Initializing RAG Agent...")
    agent = SustainabilityRAGAgent()
    
    # Check if OpenAI API key is set
    config = Config()
    if not config.OPENAI_API_KEY:
        print("⚠️  Warning: OpenAI API key not set. Some features may not work.")
        print("Please set OPENAI_API_KEY in your .env file")
    else:
        print("✅ OpenAI API key configured")
    
    # Get initial knowledge base stats
    print("\n2. Knowledge Base Statistics:")
    stats = agent.get_knowledge_base_stats()
    print(f"   Total documents: {stats.get('total_documents', 0)}")
    print(f"   Collection name: {stats.get('collection_name', 'N/A')}")
    print(f"   Embedding model: {stats.get('embedding_model', 'N/A')}")
    
    # Example 1: Adding a sustainability report (if you have one)
    print("\n3. Adding Sustainability Reports:")
    print("   Note: To test this, place a PDF sustainability report in the current directory")
    
    # Check if there are any PDF files in the current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if pdf_files:
        print(f"   Found PDF files: {pdf_files}")
        for pdf_file in pdf_files[:2]:  # Process first 2 PDFs
            print(f"   Processing: {pdf_file}")
            
            metadata = {
                "company_name": "Example Corp",
                "report_year": 2023,
                "industry": "Technology",
                "report_type": "Annual Sustainability Report",
                "example_upload": True
            }
            
            result = agent.add_report_to_knowledge_base(pdf_file, metadata)
            
            if result.get("success"):
                print(f"   ✅ Successfully processed {pdf_file}")
                print(f"      Chunks added: {result.get('chunks_added', 0)}")
                print(f"      Text length: {result.get('text_length', 0)}")
            else:
                print(f"   ❌ Failed to process {pdf_file}: {result.get('error')}")
    else:
        print("   No PDF files found in current directory")
        print("   To test file upload, add some PDF sustainability reports to this directory")
    
    # Example 2: Querying the knowledge base
    print("\n4. Querying the Knowledge Base:")
    
    sample_questions = [
        "What are the main environmental initiatives mentioned in the reports?",
        "How much has carbon emissions been reduced?",
        "What sustainability targets have been set?",
        "What are the key ESG metrics reported?"
    ]
    
    for question in sample_questions:
        print(f"\n   Question: {question}")
        
        result = agent.query_knowledge_base(question, "general_analysis")
        
        if result.get("success"):
            response = result["response"]
            print(f"   Response: {response[:200]}...")
            print(f"   Analysis type: {result.get('analysis_type')}")
            print(f"   Context used: {result.get('context_used')}")
        else:
            print(f"   ❌ Error: {result.get('error')}")
        
        print("   " + "-" * 40)
    
    # Example 3: Generate report summary
    print("\n5. Generating Report Summary:")
    summary_result = agent.generate_report_summary()
    
    if summary_result.get("success"):
        print("   ✅ Report summary generated successfully")
        print(f"   Summary (first 300 chars): {summary_result['response'][:300]}...")
    else:
        print(f"   ❌ Failed to generate summary: {summary_result.get('error')}")
    
    # Example 4: Carbon footprint analysis
    print("\n6. Carbon Footprint Analysis:")
    carbon_result = agent.analyze_carbon_footprint()
    
    if carbon_result.get("success"):
        print("   ✅ Carbon analysis completed successfully")
        print(f"   Analysis (first 300 chars): {carbon_result['response'][:300]}...")
    else:
        print(f"   ❌ Failed to analyze carbon footprint: {carbon_result.get('error')}")
    
    # Example 5: Compliance check
    print("\n7. Compliance Check (GRI Standards):")
    compliance_result = agent.check_compliance("GRI Standards")
    
    if compliance_result.get("success"):
        print("   ✅ Compliance check completed successfully")
        print(f"   Assessment (first 300 chars): {compliance_result['response'][:300]}...")
    else:
        print(f"   ❌ Failed to check compliance: {compliance_result.get('error')}")
    
    # Example 6: Industry benchmarking
    print("\n8. Industry Benchmarking (Technology):")
    benchmark_result = agent.benchmark_performance("Technology")
    
    if benchmark_result.get("success"):
        print("   ✅ Benchmarking completed successfully")
        print(f"   Benchmark (first 300 chars): {benchmark_result['response'][:300]}...")
    else:
        print(f"   ❌ Failed to perform benchmarking: {benchmark_result.get('error')}")
    
    # Example 7: Category-specific insights
    print("\n9. Sustainability Category Insights:")
    categories_to_test = ["Carbon Emissions", "Energy Consumption", "Water Usage"]
    
    for category in categories_to_test:
        print(f"\n   Analyzing category: {category}")
        insights_result = agent.get_sustainability_insights(category)
        
        if insights_result.get("success"):
            print(f"   ✅ Insights for {category} generated")
            print(f"   Insights (first 200 chars): {insights_result['response'][:200]}...")
        else:
            print(f"   ❌ Failed to get insights for {category}: {insights_result.get('error')}")
    
    # Example 8: Conversation history
    print("\n10. Conversation History:")
    history = agent.get_conversation_history()
    print(f"    Total conversations: {len(history)}")
    
    if history:
        print("    Recent conversations:")
        for i, conv in enumerate(history[-3:]):  # Show last 3 conversations
            print(f"    {i+1}. {conv.get('timestamp', 'Unknown time')[:19]}")
            print(f"       Type: {conv.get('analysis_type', 'Unknown')}")
            print(f"       Question: {conv.get('question', 'N/A')[:50]}...")
    
    # Example 9: Vector store operations
    print("\n11. Vector Store Operations:")
    vector_store = VectorStore()
    
    # Add some sample documents for demonstration
    sample_docs = [
        "Our company reduced carbon emissions by 25% through renewable energy adoption and efficiency improvements.",
        "Water consumption decreased by 15% due to new conservation technologies and employee awareness programs.",
        "Waste recycling programs helped us achieve 80% waste diversion from landfills this year.",
        "Employee diversity initiatives increased representation of underrepresented groups by 20%.",
        "Supply chain sustainability assessments were conducted for 90% of our key suppliers."
    ]
    
    print("    Adding sample sustainability documents...")
    success = vector_store.add_documents(sample_docs)
    print(f"    Sample documents added: {success}")
    
    # Search for similar documents
    print("    Testing similarity search...")
    search_results = vector_store.search_similar("carbon emissions reduction", n_results=3)
    print(f"    Found {len(search_results['documents'])} similar documents")
    
    for i, (doc, score) in enumerate(zip(search_results['documents'], search_results['scores'])):
        print(f"    {i+1}. Score: {score:.3f} - {doc[:60]}...")
    
    # Example 10: Performance metrics
    print("\n12. Performance Metrics:")
    final_stats = agent.get_knowledge_base_stats()
    print(f"    Final document count: {final_stats.get('total_documents', 0)}")
    print(f"    Unique files: {final_stats.get('unique_files', 0)}")
    print(f"    Total conversations: {len(agent.get_conversation_history())}")
    
    # Example 11: Configuration display
    print("\n13. Configuration Summary:")
    config = Config()
    print(f"    LLM Model: {config.LLM_MODEL}")
    print(f"    Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"    Temperature: {config.TEMPERATURE}")
    print(f"    Max Tokens: {config.MAX_TOKENS}")
    print(f"    Chunk Size: {config.CHUNK_SIZE}")
    print(f"    Retrieval Top K: {config.RETRIEVAL_TOP_K}")
    
    print("\n" + "=" * 50)
    print("🎉 Example usage completed!")
    print("\nNext steps:")
    print("1. Add your OpenAI API key to .env file if not already done")
    print("2. Place some PDF sustainability reports in this directory")
    print("3. Run this script again to see full functionality")
    print("4. Use streamlit run streamlit_app.py for the web interface")


def demo_pdf_processing():
    """Demonstrate PDF processing capabilities"""
    print("\n📄 PDF Processing Demo")
    print("=" * 30)
    
    processor = PDFProcessor()
    
    # Check for PDF files
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if pdf_files:
        pdf_file = pdf_files[0]
        print(f"Processing: {pdf_file}")
        
        # Validate file
        is_valid = processor.validate_file(pdf_file)
        print(f"File validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        if is_valid:
            # Process the file
            result = processor.process_file(pdf_file)
            
            if result.get("status") == "success":
                print(f"✅ Processing successful")
                print(f"   Text length: {result.get('text_length', 0)} characters")
                print(f"   Number of chunks: {result.get('chunk_count', 0)}")
                print(f"   Filename: {result.get('filename')}")
                
                # Show extracted metrics
                metrics = result.get('metrics', {})
                print(f"   Extracted metrics:")
                for metric_type, values in metrics.items():
                    if values:
                        print(f"     {metric_type}: {len(values)} items found")
            else:
                print(f"❌ Processing failed: {result.get('error')}")
    else:
        print("No PDF files found for processing demo")


def demo_advanced_queries():
    """Demonstrate advanced query capabilities"""
    print("\n🔍 Advanced Query Demo")
    print("=" * 30)
    
    agent = SustainabilityRAGAgent()
    
    # Complex multi-part queries
    advanced_queries = [
        {
            "question": "Compare the carbon reduction strategies with industry best practices",
            "type": "benchmarking",
            "context": {"industry": "Technology"}
        },
        {
            "question": "Assess compliance with TCFD recommendations",
            "type": "compliance_check",
            "context": {"framework": "TCFD"}
        },
        {
            "question": "Identify gaps in water management reporting",
            "type": "general_analysis",
            "context": {}
        }
    ]
    
    for query in advanced_queries:
        print(f"\nQuery: {query['question']}")
        print(f"Type: {query['type']}")
        
        if query['type'] == 'benchmarking':
            result = agent.benchmark_performance(
                query['context']['industry']
            )
        elif query['type'] == 'compliance_check':
            result = agent.check_compliance(
                query['context']['framework']
            )
        else:
            result = agent.query_knowledge_base(
                query['question'], 
                query['type'], 
                query['context']
            )
        
        if result.get("success"):
            print(f"✅ Response generated ({len(result['response'])} characters)")
            print(f"Preview: {result['response'][:150]}...")
        else:
            print(f"❌ Error: {result.get('error')}")


if __name__ == "__main__":
    # Run the main demonstration
    main()
    
    # Optional: Run additional demos
    print("\n" + "="*60)
    
    demo_choice = input("\nRun additional demos? (y/n): ").lower().strip()
    
    if demo_choice == 'y':
        demo_pdf_processing()
        demo_advanced_queries()
    
    print("\n🌱 Thank you for trying Sustainable Agent Consultant!")
    print("For web interface, run: streamlit run streamlit_app.py")