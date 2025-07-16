import os
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBBuilder:
    def __init__(self):
        self.data_dir = Path("fitness_app/data")
        self.vector_db_path = Path("fitness_app/data/vector_db")
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index
        self.dimension = 384  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store metadata for each vector
        self.metadata = []
    
    def load_data(self):
        """Load data from JSON files"""
        data = []
        
        # Load injury data
        injury_path = self.data_dir / "injuries"
        if injury_path.exists():
            for file in injury_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data.extend(json.load(f)["data"])
        
        # Load recovery data
        recovery_path = self.data_dir / "recovery"
        if recovery_path.exists():
            for file in recovery_path.glob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data.extend(json.load(f)["data"])
        
        return data
    
    def create_embeddings(self, data):
        """Create embeddings for the data"""
        texts = []
        for item in data:
            # Combine title and content for better context
            text = f"{item.get('title', '')} {item.get('content', '')}"
            texts.append(text)
            
            # Store metadata
            self.metadata.append({
                'title': item.get('title', ''),
                'source': item.get('source', ''),
                'category': item.get('category', ''),
                'type': item.get('type', '')
            })
        
        # Create embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return np.array(all_embeddings).astype('float32')
    
    def build_index(self):
        """Build and save the FAISS index"""
        # Load data
        logger.info("Loading data...")
        data = self.load_data()
        logger.info(f"Loaded {len(data)} items")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        embeddings = self.create_embeddings(data)
        
        # Add vectors to index
        logger.info("Building FAISS index...")
        self.index.add(embeddings)
        
        # Save index
        index_path = self.vector_db_path / "injury_recovery.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.vector_db_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Vector database created with {len(data)} items")
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    builder = VectorDBBuilder()
    builder.build_index() 