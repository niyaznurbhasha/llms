import os
from pathlib import Path
import json
import logging
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGDatabase:
    def __init__(self):
        # Setup paths
        self.base_path = Path("fitness_app/data")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.vector_db_path = self.base_path / "vector_db"
        self.vector_db_path.mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 768  # Dimension for 'all-MiniLM-L6-v2' model
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index if available"""
        index_path = self.vector_db_path / "injury_recovery.index"
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            index_path = self.vector_db_path / "injury_recovery.index"
            faiss.write_index(self.index, str(index_path))
            logger.info("Saved FAISS index")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
    
    def process_and_store_data(self, data: list, source: str):
        """Process and store data in the RAG system"""
        try:
            # Process in smaller batches to manage GPU memory
            batch_size = 32
            total_processed = 0
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Generate embeddings for batch
                texts = []
                for item in batch:
                    if isinstance(item, dict) and "content" in item:
                        texts.append(item["content"])
                    else:
                        logger.warning(f"Skipping invalid item in batch: {item}")
                        continue
                
                if not texts:
                    logger.warning("No valid texts to process in batch")
                    continue
                
                try:
                    # Generate embeddings with smaller batch size for GPU memory
                    embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=4)
                    
                    # Add to FAISS index
                    self.index.add(embeddings.astype('float32'))
                    
                    # Save metadata
                    metadata = []
                    for item, embedding in zip(batch, embeddings):
                        if isinstance(item, dict) and "content" in item:
                            metadata.append({
                                "title": item.get("title", ""),
                                "source": source,
                                "category": item.get("category", ""),
                                "type": item.get("type", "general"),
                                "metadata": item.get("metadata", {}),
                                "embedding": embedding.tolist()
                            })
                    
                    # Save metadata
                    metadata_path = self.vector_db_path / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                existing_metadata = json.load(f)
                            existing_metadata.extend(metadata)
                        except json.JSONDecodeError:
                            logger.warning("Error reading existing metadata, starting fresh")
                            existing_metadata = metadata
                    else:
                        existing_metadata = metadata
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(existing_metadata, f, indent=2)
                    
                    total_processed += len(texts)
                    logger.info(f"Processed batch of {len(texts)} items. Total processed: {total_processed}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                    continue
            
            # Save index
            self._save_index()
            
            logger.info(f"Successfully processed and stored {total_processed} items from {source}")
            
        except Exception as e:
            logger.error(f"Error processing and storing data: {str(e)}", exc_info=True)
            raise
    
    def search(self, query: str, k: int = 5):
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0].astype('float32')
            
            # Search the index
            distances, indices = self.index.search(np.array([query_embedding]), k)
            
            # Load metadata
            metadata_path = self.vector_db_path / "metadata.json"
            if not metadata_path.exists():
                return []
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(metadata):  # Valid index
                    result = {
                        'metadata': metadata[idx],
                        'relevance_score': float(1 / (1 + distance)),  # Convert distance to similarity score
                        'index': int(idx)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return [] 