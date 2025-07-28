"""
Vector Store module for Sustainable Agent Consultant
Handles document embedding, storage, and retrieval using ChromaDB and sentence transformers
"""

import os
import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles document embedding and retrieval using ChromaDB"""
    
    def __init__(self):
        self.config = Config()
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the embedding model and ChromaDB client"""
        try:
            # Initialize sentence transformer model
            logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            
            # Initialize ChromaDB client
            persist_directory = Path(self.config.VECTOR_DB_PATH)
            persist_directory.mkdir(exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.config.COLLECTION_NAME
                )
                logger.info(f"Loaded existing collection: {self.config.COLLECTION_NAME}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.config.COLLECTION_NAME,
                    metadata={"description": "Sustainability reports collection"}
                )
                logger.info(f"Created new collection: {self.config.COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store components: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            if not texts:
                return []
            
            # Generate embeddings using sentence transformer
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            
            # Convert to list format for ChromaDB
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def generate_document_id(self, filename: str, chunk_index: int) -> str:
        """Generate a unique ID for a document chunk"""
        content = f"{filename}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]] = None,
                     ids: List[str] = None) -> bool:
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.generate_embeddings(documents)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Generate IDs if not provided
            if not ids:
                ids = [self.generate_document_id("doc", i) for i in range(len(documents))]
            
            # Generate default metadata if not provided
            if not metadatas:
                metadatas = [{"text_length": len(doc)} for doc in documents]
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def add_sustainability_report(self, 
                                 chunks: List[str], 
                                 filename: str,
                                 metadata: Dict[str, Any] = None) -> bool:
        """Add a sustainability report to the vector store"""
        try:
            if not chunks:
                logger.warning("No chunks provided for the report")
                return False
            
            # Generate IDs for chunks
            ids = [self.generate_document_id(filename, i) for i in range(len(chunks))]
            
            # Create metadata for each chunk
            base_metadata = metadata or {}
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "filename": filename,
                    "chunk_index": i,
                    "text_length": len(chunk),
                    "total_chunks": len(chunks),
                    "document_type": "sustainability_report"
                }
                chunk_metadata.update(base_metadata)
                metadatas.append(chunk_metadata)
            
            # Add to vector store
            success = self.add_documents(chunks, metadatas, ids)
            
            if success:
                logger.info(f"Successfully added sustainability report '{filename}' with {len(chunks)} chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding sustainability report: {str(e)}")
            return False
    
    def search_similar(self, 
                      query: str, 
                      n_results: int = None,
                      filter_metadata: Dict[str, Any] = None) -> Dict[str, List]:
        """Search for similar documents using semantic similarity"""
        try:
            n_results = n_results or self.config.RETRIEVAL_TOP_K
            
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "scores": []
            }
            
            # Convert distances to similarity scores (1 - distance)
            if results["distances"] and results["distances"][0]:
                processed_results["scores"] = [
                    1 - dist for dist in results["distances"][0]
                ]
            
            logger.info(f"Found {len(processed_results['documents'])} similar documents")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": [], "scores": []}
    
    def search_by_category(self, 
                          query: str, 
                          category: str = None,
                          n_results: int = None) -> Dict[str, List]:
        """Search documents by sustainability category"""
        try:
            filter_metadata = None
            if category and category in self.config.SUSTAINABILITY_CATEGORIES:
                # You could add category classification logic here
                # For now, we'll just do a general search
                pass
            
            return self.search_similar(query, n_results, filter_metadata)
            
        except Exception as e:
            logger.error(f"Error searching by category: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": [], "scores": []}
    
    def get_relevant_context(self, 
                           query: str, 
                           max_chars: int = 4000,
                           min_similarity: float = None) -> str:
        """Get relevant context for a query by combining similar documents"""
        try:
            min_similarity = min_similarity or self.config.SIMILARITY_THRESHOLD
            
            # Search for similar documents
            results = self.search_similar(query, n_results=10)
            
            if not results["documents"]:
                return ""
            
            # Filter by similarity threshold
            relevant_docs = []
            for i, (doc, score) in enumerate(zip(results["documents"], results["scores"])):
                if score >= min_similarity:
                    metadata = results["metadatas"][i] if i < len(results["metadatas"]) else {}
                    relevant_docs.append({
                        "text": doc,
                        "score": score,
                        "metadata": metadata
                    })
            
            if not relevant_docs:
                # If no docs meet threshold, take the top result
                if results["documents"]:
                    relevant_docs.append({
                        "text": results["documents"][0],
                        "score": results["scores"][0] if results["scores"] else 0.0,
                        "metadata": results["metadatas"][0] if results["metadatas"] else {}
                    })
            
            # Combine documents up to max_chars
            context = ""
            char_count = 0
            
            for doc_info in relevant_docs:
                doc_text = doc_info["text"]
                if char_count + len(doc_text) <= max_chars:
                    if context:
                        context += "\n\n---\n\n"
                    context += doc_text
                    char_count += len(doc_text)
                else:
                    # Add partial text if it fits
                    remaining_chars = max_chars - char_count
                    if remaining_chars > 100:  # Only add if significant text can fit
                        if context:
                            context += "\n\n---\n\n"
                        context += doc_text[:remaining_chars] + "..."
                    break
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""
    
    def delete_documents_by_filename(self, filename: str) -> bool:
        """Delete all documents from a specific file"""
        try:
            # Get all documents with the filename
            results = self.collection.get(
                where={"filename": filename},
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} documents from file: {filename}")
                return True
            else:
                logger.info(f"No documents found for file: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            # Get collection count
            count_result = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.get(
                limit=100,
                include=["metadatas"]
            )
            
            stats = {
                "total_documents": count_result,
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_model": self.config.EMBEDDING_MODEL
            }
            
            # Analyze metadata if available
            if sample_results["metadatas"]:
                filenames = set()
                document_types = set()
                
                for metadata in sample_results["metadatas"]:
                    if "filename" in metadata:
                        filenames.add(metadata["filename"])
                    if "document_type" in metadata:
                        document_types.add(metadata["document_type"])
                
                stats.update({
                    "unique_files": len(filenames),
                    "document_types": list(document_types),
                    "sample_files": list(filenames)[:10]  # First 10 files
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)"""
        try:
            self.chroma_client.delete_collection(name=self.config.COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"description": "Sustainability reports collection"}
            )
            logger.info("Collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore()
    
    # Example documents
    sample_docs = [
        "Our company reduced carbon emissions by 25% this year through renewable energy adoption.",
        "Water consumption decreased by 15% due to improved efficiency measures.",
        "We achieved 100% renewable energy for all our manufacturing facilities.",
        "Waste recycling programs helped us divert 80% of waste from landfills."
    ]
    
    # Add sample documents
    success = vector_store.add_documents(sample_docs)
    print(f"Added documents: {success}")
    
    # Search for similar content
    results = vector_store.search_similar("carbon emissions reduction")
    print(f"Search results: {len(results['documents'])} documents found")
    
    # Get collection stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    print("Vector Store module loaded successfully!")