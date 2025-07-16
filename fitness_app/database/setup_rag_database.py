import sqlite3
import json
import requests
from datetime import datetime
import logging
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import time
from ratelimit import limits, sleep_and_retry

class RAGDatabase:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Setup paths
        self.base_path = Path("fitness_app/data")
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.base_path / "injury_recovery.db"
        self.faiss_path = self.base_path / "injury_embeddings.faiss"
        self.metadata_path = self.base_path / "injury_metadata.json"
        
        # Initialize databases
        self.conn = None
        self.cursor = None
        self.vector_dim = 768  # Dimension for 'all-MiniLM-L6-v2' model
        
        # Setup GPU resources
        self.res = faiss.StandardGpuResources()  # Use GPU
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup logging
        logging.basicConfig(
            filename=self.base_path / 'database_setup.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def setup_sqlite(self):
        """Set up SQLite database with the same schema as before"""
        try:
            self.conn = sqlite3.connect(self.sqlite_path)
            self.cursor = self.conn.cursor()
            
            # Create tables (same as before)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS injuries (
                    injury_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    common_symptoms TEXT,
                    severity_levels TEXT,
                    risk_factors TEXT,
                    typical_recovery_time TEXT,
                    warning_signs TEXT,
                    body_part TEXT,
                    category TEXT,
                    metadata TEXT
                )
            ''')
            # ... (other table creation statements)
            
            self.conn.commit()
            logging.info("SQLite database setup completed")
            
        except Exception as e:
            logging.error(f"Error setting up SQLite: {str(e)}")
            raise
    
    def setup_faiss(self):
        """Set up FAISS index for vector search using GPU"""
        try:
            # Create a CPU index first
            cpu_index = faiss.IndexFlatL2(self.vector_dim)
            
            # Move it to GPU
            self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
            
            logging.info("FAISS GPU index created successfully")
            
        except Exception as e:
            logging.error(f"Error setting up FAISS GPU: {str(e)}")
            # Fallback to CPU if GPU fails
            logging.info("Falling back to CPU index")
            self.index = faiss.IndexFlatL2(self.vector_dim)
    
    # Rate limiting decorator for OpenFDA API
    @sleep_and_retry
    @limits(calls=240, period=60)  # 240 calls per minute
    def make_openfda_request(self, url, params):
        """Make a rate-limited request to OpenFDA API"""
        return requests.get(url, params=params)
    
    def fetch_openfda_data(self):
        """Fetch injury data from OpenFDA API with rate limiting"""
        try:
            openfda_key = os.getenv('OPENFDA_API_KEY')
            if not openfda_key:
                raise ValueError("OPENFDA_API_KEY not found in environment variables")
            
            url = "https://api.fda.gov/drug/event.json"
            params = {
                "api_key": openfda_key,
                "search": "patient.reaction.reactionmeddrapt:(\"SPORTS INJURY\" OR \"MUSCULOSKELETAL\" OR \"JOINT PAIN\" OR \"BACK PAIN\" OR \"TENDONITIS\" OR \"FRACTURE\" OR \"SPRAIN\" OR \"DISLOCATION\")",
                "limit": 100
            }
            
            response = self.make_openfda_request(url, params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
        except Exception as e:
            logging.error(f"Error fetching OpenFDA data: {str(e)}")
            return []
    
    def fetch_who_data(self):
        """Fetch injury data from WHO OData API"""
        try:
            # WHO OData API endpoint for injury data
            url = "https://ghoapi.azureedge.net/api/Indicator"
            
            # First, get the list of injury-related indicators
            params = {
                "$filter": "contains(IndicatorName, 'injury') or contains(IndicatorName, 'accident') or contains(IndicatorName, 'trauma') or contains(IndicatorName, 'fracture') or contains(IndicatorName, 'sprain') or contains(IndicatorName, 'strain')"
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            indicators = response.json().get("value", [])
            
            # Fetch data for each indicator
            who_data = []
            for indicator in indicators:
                indicator_code = indicator.get("IndicatorCode")
                if indicator_code:
                    # Fetch data for this indicator
                    data_url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
                    data_response = requests.get(data_url)
                    if data_response.status_code == 200:
                        who_data.extend(data_response.json().get("value", []))
                
                # Be nice to the API
                time.sleep(1)
            
            return who_data
            
        except Exception as e:
            logging.error(f"Error fetching WHO data: {str(e)}")
            return []
    
    def process_who_data(self, data):
        """Process WHO data into our schema"""
        processed_data = []
        for item in data:
            processed_item = {
                "name": item.get("IndicatorName", ""),
                "description": item.get("IndicatorName", ""),
                "common_symptoms": json.dumps([]),  # WHO data doesn't include symptoms
                "severity_levels": json.dumps(["Mild", "Moderate", "Severe"]),
                "risk_factors": json.dumps([]),
                "typical_recovery_time": "Varies",
                "warning_signs": json.dumps([]),
                "body_part": "To be determined",
                "category": "WHO Injury Data",
                "metadata": json.dumps(item)
            }
            processed_data.append(processed_item)
        return processed_data
    
    def create_embeddings(self, text):
        """Create embeddings for text using the sentence transformer"""
        try:
            # Create embedding
            embedding = self.model.encode([text])[0]
            return embedding
            
        except Exception as e:
            logging.error(f"Error creating embedding: {str(e)}")
            return None
    
    def process_and_store_data(self, data, source="OpenFDA"):
        """Process data and store in both SQLite and FAISS"""
        try:
            metadata = []
            embeddings_batch = []
            batch_size = 100  # Process embeddings in batches for better GPU utilization
            
            for item in data:
                # Extract text for embedding
                if source == "OpenFDA":
                    text_for_embedding = f"""
                    Injury: {item.get('reaction', {}).get('reactionmeddrapt', '')}
                    Description: {item.get('reaction', {}).get('reactionoutcome', '')}
                    Symptoms: {', '.join(item.get('reaction', {}).get('reactionmeddrapt', []))}
                    """
                    injury_data = {
                        "name": item.get("reaction", {}).get("reactionmeddrapt", ""),
                        "description": item.get("reaction", {}).get("reactionoutcome", ""),
                        "common_symptoms": json.dumps(item.get("reaction", {}).get("reactionmeddrapt", [])),
                        "severity_levels": json.dumps(["Mild", "Moderate", "Severe"]),
                        "risk_factors": json.dumps(item.get("patient", {}).get("drug", [])),
                        "typical_recovery_time": "Varies",
                        "warning_signs": json.dumps(["Increased pain", "Swelling", "Limited mobility"]),
                        "body_part": "To be determined",
                        "category": "Sports Injury",
                        "metadata": json.dumps(item)
                    }
                else:  # WHO data
                    text_for_embedding = f"""
                    Injury: {item.get('IndicatorName', '')}
                    Description: {item.get('IndicatorName', '')}
                    Value: {item.get('Value', '')}
                    """
                    injury_data = {
                        "name": item.get("IndicatorName", ""),
                        "description": item.get("IndicatorName", ""),
                        "common_symptoms": json.dumps([]),
                        "severity_levels": json.dumps(["Mild", "Moderate", "Severe"]),
                        "risk_factors": json.dumps([]),
                        "typical_recovery_time": "Varies",
                        "warning_signs": json.dumps([]),
                        "body_part": "To be determined",
                        "category": "WHO Injury Data",
                        "metadata": json.dumps(item)
                    }
                
                # Create embedding
                embedding = self.create_embeddings(text_for_embedding)
                if embedding is not None:
                    embeddings_batch.append(embedding)
                    
                    # Store metadata
                    metadata.append({
                        "injury_id": len(metadata),
                        "text": text_for_embedding,
                        "source": source,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Store in SQLite
                self.cursor.execute('''
                    INSERT INTO injuries (
                        name, description, common_symptoms, severity_levels,
                        risk_factors, typical_recovery_time, warning_signs,
                        body_part, category, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    injury_data["name"],
                    injury_data["description"],
                    injury_data["common_symptoms"],
                    injury_data["severity_levels"],
                    injury_data["risk_factors"],
                    injury_data["typical_recovery_time"],
                    injury_data["warning_signs"],
                    injury_data["body_part"],
                    injury_data["category"],
                    injury_data["metadata"]
                ))
                
                # Process embeddings in batches
                if len(embeddings_batch) >= batch_size:
                    self.index.add(np.array(embeddings_batch))
                    embeddings_batch = []
            
            # Process any remaining embeddings
            if embeddings_batch:
                self.index.add(np.array(embeddings_batch))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Save FAISS index
            # Convert GPU index to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(self.faiss_path))
            
            self.conn.commit()
            logging.info(f"Successfully processed and stored {len(data)} records from {source}")
            
        except Exception as e:
            logging.error(f"Error processing and storing data: {str(e)}")
            raise
    
    def setup(self):
        """Set up both databases and populate with data"""
        try:
            # Setup databases
            self.setup_sqlite()
            self.setup_faiss()
            
            # Fetch and process data from both sources
            openfda_data = self.fetch_openfda_data()
            who_data = self.fetch_who_data()
            
            # Process and store data
            self.process_and_store_data(openfda_data, source="OpenFDA")
            self.process_and_store_data(who_data, source="WHO")
            
            logging.info("Database setup completed successfully")
            
        except Exception as e:
            logging.error(f"Error in setup: {str(e)}")
            raise
        finally:
            if self.conn:
                self.conn.close()

    def __del__(self):
        """Cleanup GPU resources"""
        if hasattr(self, 'res'):
            del self.res
        if hasattr(self, 'index'):
            del self.index

if __name__ == "__main__":
    db = RAGDatabase()
    db.setup() 