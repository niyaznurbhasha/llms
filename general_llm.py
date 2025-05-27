from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Tuple, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer

class TopicLLM:
    def __init__(self, 
                 model_name: str = "deepseek-ai/deepseek-llm-7b-base",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the TopicLLM with a base model and embedding model
        
        Args:
            model_name: The name of the base LLM model to use
            embedding_model: The name of the sentence transformer model for embeddings
        """
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.Client()
        self.collections = {}  # Store multiple collections for different topics
        
        # Setup logging
        logging.basicConfig(
            filename='topic_llm.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize metrics
        self.metrics = {
            "total_queries": 0,
            "topics": {},
            "average_response_time": 0,
            "query_history": []
        }
    
    def create_topic_collection(self, topic: str) -> str:
        """Create a new collection for a specific topic"""
        collection_name = f"topic_{topic.lower().replace(' ', '_')}"
        if collection_name not in self.collections:
            self.collections[collection_name] = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model
                )
            )
            self.metrics["topics"][topic] = {
                "queries": 0,
                "documents": 0,
                "last_updated": datetime.now().isoformat()
            }
        return collection_name
    
    def add_documents(self, topic: str, documents: List[Dict[str, Any]]):
        """
        Add documents to a topic collection
        
        Args:
            topic: The topic to add documents to
            documents: List of documents with metadata
        """
        collection_name = self.create_topic_collection(topic)
        collection = self.collections[collection_name]
        
        for idx, doc in enumerate(tqdm(documents, desc=f"Adding documents to {topic}")):
            try:
                # Extract text content and metadata
                content = doc.get("content", "")
                metadata = {
                    k: v for k, v in doc.items() if k != "content"
                }
                
                # Add document to collection
                collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[f"doc_{topic}_{idx}"]
                )
                
                self.metrics["topics"][topic]["documents"] += 1
                
            except Exception as e:
                logging.error(f"Error adding document to {topic}: {str(e)}")
    
    def query_topic(self, 
                   topic: str,
                   query: str,
                   num_results: int = 3,
                   filters: Optional[Dict] = None) -> Tuple[str, List[Dict]]:
        """
        Query a specific topic collection
        
        Args:
            topic: The topic to query
            query: The query text
            num_results: Number of results to return
            filters: Optional filters to apply to the query
            
        Returns:
            Tuple of (answer, retrieved_context)
        """
        start_time = datetime.now()
        
        # Get or create collection
        collection_name = self.create_topic_collection(topic)
        collection = self.collections[collection_name]
        
        # Update metrics
        self.metrics["total_queries"] += 1
        self.metrics["topics"][topic]["queries"] += 1
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=num_results,
            where=filters
        )
        
        # Construct context from retrieved documents
        context = "\n\n".join([
            f"Document: {doc}\nMetadata: {json.dumps(meta, indent=2)}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])
        
        # Construct prompt
        prompt = self._construct_prompt(topic, query, context)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * (self.metrics["total_queries"] - 1) + response_time)
            / self.metrics["total_queries"]
        )
        
        # Update query history
        self.metrics["query_history"].append({
            "topic": topic,
            "query": query,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Return both the answer and the retrieved context
        retrieved_context = [
            {
                "content": doc,
                **meta
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
        
        return answer, retrieved_context
    
    def _construct_prompt(self, topic: str, query: str, context: str) -> str:
        """Construct a prompt for the LLM based on topic and query"""
        return f"""You are an expert on {topic}. Use the following context to answer the query.
        
        Context:
        {context}
        
        Query: {query}
        
        Provide a comprehensive answer that:
        1. Directly addresses the query
        2. Uses relevant information from the context
        3. Maintains accuracy and objectivity
        4. Includes specific details and examples when available
        5. Acknowledges any limitations or uncertainties
        """
    
    def get_topic_statistics(self, topic: str) -> Dict:
        """Get statistics for a specific topic"""
        if topic not in self.metrics["topics"]:
            return {"error": f"No data available for topic: {topic}"}
        
        return {
            "topic": topic,
            "total_queries": self.metrics["topics"][topic]["queries"],
            "total_documents": self.metrics["topics"][topic]["documents"],
            "last_updated": self.metrics["topics"][topic]["last_updated"],
            "average_response_time": self.metrics["average_response_time"]
        }
    
    def get_all_statistics(self) -> Dict:
        """Get statistics for all topics"""
        return {
            "total_queries": self.metrics["total_queries"],
            "topics": self.metrics["topics"],
            "average_response_time": self.metrics["average_response_time"],
            "total_topics": len(self.metrics["topics"])
        }
    
    def export_topic_data(self, topic: str, output_dir: str = "data"):
        """Export all data for a specific topic"""
        if topic not in self.metrics["topics"]:
            return {"error": f"No data available for topic: {topic}"}
        
        output_path = Path(output_dir) / f"{topic.lower().replace(' ', '_')}_data"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export documents
        collection_name = self.create_topic_collection(topic)
        collection = self.collections[collection_name]
        results = collection.get()
        
        data = {
            "metadata": {
                "topic": topic,
                "export_date": datetime.now().isoformat(),
                "total_documents": len(results["documents"]),
                "statistics": self.get_topic_statistics(topic)
            },
            "documents": [
                {
                    "content": doc,
                    **meta
                }
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]
        }
        
        # Save to JSON
        with open(output_path / "documents.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save metrics
        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.get_topic_statistics(topic), f, indent=2, ensure_ascii=False)
        
        return {"success": f"Data exported to {output_path}"}

if __name__ == "__main__":
    # Example usage
    llm = TopicLLM()
    
    # Add some example documents
    documents = [
        {
            "content": "Muscle injuries are common in sports and can be classified as strains or tears.",
            "source": "medical_journal",
            "type": "factual",
            "confidence": 0.95
        },
        {
            "content": "Proper warm-up and stretching can help prevent muscle injuries.",
            "source": "fitness_guide",
            "type": "advice",
            "confidence": 0.9
        }
    ]
    
    llm.add_documents("fitness muscle injuries", documents)
    
    # Query the topic
    answer, context = llm.query_topic(
        "fitness muscle injuries",
        "What are the best ways to prevent muscle injuries?",
        num_results=2
    )
    
    print("Answer:", answer)
    print("\nContext:", json.dumps(context, indent=2))
    
    # Get statistics
    print("\nStatistics:", json.dumps(llm.get_topic_statistics("fitness muscle injuries"), indent=2)) 