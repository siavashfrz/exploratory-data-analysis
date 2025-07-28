"""
RAG Agent module for Sustainable Agent Consultant
Combines document retrieval with language generation for intelligent sustainability consulting
"""

import os
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import Config
from vector_store import VectorStore
from pdf_processor import PDFProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SustainabilityRAGAgent:
    """RAG Agent for sustainability consulting and report analysis"""
    
    def __init__(self):
        self.config = Config()
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.llm = None
        self.conversation_history = []
        self._initialize_llm()
        self._setup_prompts()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if not self.config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not found. Some features may not work.")
                return
            
            self.llm = ChatOpenAI(
                model_name=self.config.LLM_MODEL,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
                openai_api_key=self.config.OPENAI_API_KEY
            )
            
            logger.info(f"Initialized LLM: {self.config.LLM_MODEL}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self.llm = None
    
    def _setup_prompts(self):
        """Setup prompt templates for different types of analysis"""
        self.prompts = {
            "general_analysis": PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a sustainability consultant expert. Based on the following context from sustainability reports, answer the question comprehensively and provide actionable insights.

Context:
{context}

Question: {question}

Please provide a detailed, professional response that:
1. Directly addresses the question
2. References specific data from the context when available
3. Provides actionable recommendations
4. Highlights key sustainability metrics and trends
5. Considers industry best practices

Response:"""
            ),
            
            "report_summary": PromptTemplate(
                input_variables=["context"],
                template="""You are a sustainability expert. Analyze the following sustainability report content and provide a comprehensive summary.

Report Content:
{context}

Please provide a structured summary that includes:

## Executive Summary
Brief overview of the organization's sustainability performance and key highlights.

## Key Environmental Metrics
- Carbon emissions and reduction targets
- Energy consumption and renewable energy adoption
- Water usage and conservation efforts
- Waste management and circular economy initiatives

## Social Responsibility
- Employee welfare and diversity initiatives
- Community engagement and social impact
- Supply chain sustainability practices

## Governance and Strategy
- Sustainability governance structure
- Long-term sustainability goals and commitments
- Stakeholder engagement approaches

## Key Achievements and Challenges
- Notable sustainability achievements
- Areas requiring improvement
- Future sustainability plans

## Recommendations
Actionable recommendations for improving sustainability performance.

Summary:"""
            ),
            
            "carbon_analysis": PromptTemplate(
                input_variables=["context"],
                template="""You are a carbon footprint specialist. Analyze the following content from sustainability reports focusing specifically on carbon emissions and climate-related data.

Content:
{context}

Please provide a detailed carbon footprint analysis that includes:

## Carbon Emissions Overview
- Current carbon footprint (Scope 1, 2, and 3 emissions)
- Historical trends and year-over-year changes
- Carbon intensity metrics

## Reduction Strategies
- Current carbon reduction initiatives
- Renewable energy adoption
- Energy efficiency measures
- Carbon offset programs

## Targets and Goals
- Science-based targets (SBTs) or other carbon goals
- Timeline for achieving net-zero or carbon neutrality
- Progress against stated targets

## Benchmarking
- Comparison with industry standards
- Alignment with Paris Agreement goals
- Compliance with carbon reporting frameworks

## Recommendations
- Priority areas for carbon reduction
- Potential improvement strategies
- Investment recommendations for low-carbon technologies

Analysis:"""
            ),
            
            "compliance_check": PromptTemplate(
                input_variables=["context", "framework"],
                template="""You are a sustainability compliance expert. Analyze the following sustainability report content against the {framework} framework.

Report Content:
{context}

Framework: {framework}

Please provide a compliance assessment that includes:

## Framework Alignment
- How well the report aligns with {framework} requirements
- Required disclosures that are present
- Missing or incomplete disclosures

## Data Quality Assessment
- Completeness of sustainability data
- Data verification and assurance
- Methodology transparency

## Reporting Quality
- Clarity and comprehensiveness of reporting
- Stakeholder relevance
- Materiality assessment quality

## Compliance Score
Provide an overall compliance score (1-10) with justification.

## Improvement Recommendations
- Specific areas for improvement
- Missing disclosures to address
- Best practices to implement

Assessment:"""
            ),
            
            "benchmarking": PromptTemplate(
                input_variables=["context", "industry"],
                template="""You are a sustainability benchmarking expert. Analyze the following sustainability data and compare it against {industry} industry benchmarks.

Sustainability Data:
{context}

Industry: {industry}

Please provide a benchmarking analysis that includes:

## Performance Comparison
- How the organization performs relative to industry averages
- Percentile ranking where possible
- Best-in-class comparisons

## Key Performance Indicators
- Carbon intensity vs. industry average
- Energy efficiency metrics
- Water usage efficiency
- Waste reduction performance

## Competitive Positioning
- Sustainability leaders in the industry
- Areas where the organization excels
- Areas requiring improvement

## Industry Trends
- Emerging sustainability trends in {industry}
- Regulatory developments affecting the sector
- Innovation opportunities

## Strategic Recommendations
- Areas for competitive advantage
- Investment priorities
- Partnership opportunities

Benchmarking Report:"""
            )
        }
    
    def add_report_to_knowledge_base(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process and add a sustainability report to the knowledge base"""
        try:
            logger.info(f"Processing sustainability report: {file_path}")
            
            # Process the PDF file
            processing_result = self.pdf_processor.process_file(file_path)
            
            if "error" in processing_result:
                return {"success": False, "error": processing_result["error"]}
            
            # Add to vector store
            success = self.vector_store.add_sustainability_report(
                chunks=processing_result["chunks"],
                filename=processing_result["filename"],
                metadata=metadata
            )
            
            if success:
                result = {
                    "success": True,
                    "filename": processing_result["filename"],
                    "chunks_added": processing_result["chunk_count"],
                    "text_length": processing_result["text_length"],
                    "metrics_extracted": processing_result["metrics"]
                }
                logger.info(f"Successfully added report to knowledge base: {result}")
                return result
            else:
                return {"success": False, "error": "Failed to add to vector store"}
                
        except Exception as e:
            logger.error(f"Error adding report to knowledge base: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def query_knowledge_base(self, 
                           question: str, 
                           analysis_type: str = "general_analysis",
                           additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the knowledge base and generate a response using RAG"""
        try:
            if not self.llm:
                return {
                    "success": False,
                    "error": "Language model not initialized. Please check OpenAI API key."
                }
            
            # Get relevant context from vector store
            context = self.vector_store.get_relevant_context(question)
            
            if not context:
                return {
                    "success": False,
                    "error": "No relevant information found in the knowledge base."
                }
            
            # Select appropriate prompt template
            if analysis_type not in self.prompts:
                analysis_type = "general_analysis"
            
            prompt_template = self.prompts[analysis_type]
            
            # Prepare prompt variables
            if analysis_type == "general_analysis":
                prompt_vars = {"context": context, "question": question}
            elif analysis_type == "report_summary":
                prompt_vars = {"context": context}
            elif analysis_type == "carbon_analysis":
                prompt_vars = {"context": context}
            elif analysis_type == "compliance_check":
                framework = additional_context.get("framework", "GRI Standards") if additional_context else "GRI Standards"
                prompt_vars = {"context": context, "framework": framework}
            elif analysis_type == "benchmarking":
                industry = additional_context.get("industry", "General") if additional_context else "General"
                prompt_vars = {"context": context, "industry": industry}
            else:
                prompt_vars = {"context": context, "question": question}
            
            # Generate response using LLM
            formatted_prompt = prompt_template.format(**prompt_vars)
            
            messages = [
                SystemMessage(content="You are an expert sustainability consultant with deep knowledge of environmental, social, and governance (ESG) practices."),
                HumanMessage(content=formatted_prompt)
            ]
            
            response = self.llm(messages)
            
            # Store conversation in history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "analysis_type": analysis_type,
                "response": response.content,
                "context_length": len(context)
            })
            
            return {
                "success": True,
                "response": response.content,
                "analysis_type": analysis_type,
                "context_used": len(context) > 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_report_summary(self, filename: str = None) -> Dict[str, Any]:
        """Generate a comprehensive summary of a sustainability report"""
        try:
            # If filename specified, get context from that specific file
            if filename:
                # Search for content from specific file
                search_query = "sustainability report summary environmental social governance"
                results = self.vector_store.search_similar(
                    search_query, 
                    n_results=20,
                    filter_metadata={"filename": filename}
                )
                
                if not results["documents"]:
                    return {
                        "success": False,
                        "error": f"No content found for file: {filename}"
                    }
                
                # Combine all chunks from the file
                context = "\n\n---\n\n".join(results["documents"])
            else:
                # Get general context
                context = self.vector_store.get_relevant_context(
                    "sustainability report overview environmental social governance", 
                    max_chars=6000
                )
            
            if not context:
                return {
                    "success": False,
                    "error": "No sustainability report content found."
                }
            
            return self.query_knowledge_base(
                question="", 
                analysis_type="report_summary"
            )
            
        except Exception as e:
            logger.error(f"Error generating report summary: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_carbon_footprint(self, filename: str = None) -> Dict[str, Any]:
        """Analyze carbon footprint and climate-related information"""
        try:
            # Search for carbon and climate-related content
            search_query = "carbon emissions greenhouse gas climate change net zero carbon footprint"
            
            filter_metadata = {"filename": filename} if filename else None
            
            results = self.vector_store.search_similar(
                search_query,
                n_results=15,
                filter_metadata=filter_metadata
            )
            
            if not results["documents"]:
                return {
                    "success": False,
                    "error": "No carbon footprint data found."
                }
            
            context = "\n\n---\n\n".join(results["documents"])
            
            return self.query_knowledge_base(
                question="",
                analysis_type="carbon_analysis"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing carbon footprint: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def check_compliance(self, framework: str = "GRI Standards", filename: str = None) -> Dict[str, Any]:
        """Check compliance against sustainability reporting frameworks"""
        try:
            search_query = f"sustainability reporting {framework} compliance disclosure"
            
            filter_metadata = {"filename": filename} if filename else None
            
            results = self.vector_store.search_similar(
                search_query,
                n_results=15,
                filter_metadata=filter_metadata
            )
            
            if not results["documents"]:
                return {
                    "success": False,
                    "error": f"No content found for compliance check against {framework}."
                }
            
            context = "\n\n---\n\n".join(results["documents"])
            
            return self.query_knowledge_base(
                question="",
                analysis_type="compliance_check",
                additional_context={"framework": framework}
            )
            
        except Exception as e:
            logger.error(f"Error checking compliance: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def benchmark_performance(self, industry: str, filename: str = None) -> Dict[str, Any]:
        """Benchmark sustainability performance against industry standards"""
        try:
            search_query = f"sustainability performance metrics {industry} benchmarking industry comparison"
            
            filter_metadata = {"filename": filename} if filename else None
            
            results = self.vector_store.search_similar(
                search_query,
                n_results=15,
                filter_metadata=filter_metadata
            )
            
            if not results["documents"]:
                return {
                    "success": False,
                    "error": f"No content found for benchmarking against {industry} industry."
                }
            
            context = "\n\n---\n\n".join(results["documents"])
            
            return self.query_knowledge_base(
                question="",
                analysis_type="benchmarking",
                additional_context={"industry": industry}
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking performance: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_sustainability_insights(self, category: str = None) -> Dict[str, Any]:
        """Get insights on specific sustainability categories"""
        try:
            if category and category in self.config.SUSTAINABILITY_CATEGORIES:
                search_query = f"{category} sustainability metrics performance targets"
            else:
                search_query = "sustainability environmental social governance ESG metrics"
            
            results = self.vector_store.search_similar(search_query, n_results=10)
            
            if not results["documents"]:
                return {
                    "success": False,
                    "error": "No sustainability insights found."
                }
            
            # Create a focused question based on category
            if category:
                question = f"What insights and trends can you provide about {category} in the sustainability reports?"
            else:
                question = "What are the key sustainability insights and trends from the reports?"
            
            return self.query_knowledge_base(question, "general_analysis")
            
        except Exception as e:
            logger.error(f"Error getting sustainability insights: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return self.vector_store.get_collection_stats()
    
    def remove_report(self, filename: str) -> bool:
        """Remove a specific report from the knowledge base"""
        return self.vector_store.delete_documents_by_filename(filename)


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG agent
    agent = SustainabilityRAGAgent()
    
    # Test query (assuming some documents are already in the knowledge base)
    result = agent.query_knowledge_base("What are the main carbon reduction strategies?")
    
    if result["success"]:
        print("Query successful!")
        print(f"Response: {result['response'][:200]}...")
    else:
        print(f"Query failed: {result['error']}")
    
    # Get knowledge base stats
    stats = agent.get_knowledge_base_stats()
    print(f"Knowledge base stats: {stats}")
    
    print("RAG Agent module loaded successfully!")