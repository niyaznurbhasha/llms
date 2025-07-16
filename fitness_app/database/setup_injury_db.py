import sqlite3
import json
import requests
from datetime import datetime
import logging
from pathlib import Path

class InjuryDatabase:
    def __init__(self, db_path: str = "fitness_app/data/injury_recovery.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        
        # Setup logging
        logging.basicConfig(
            filename=self.db_path.parent / 'database_setup.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logging.info("Successfully connected to database")
        except Exception as e:
            logging.error(f"Error connecting to database: {str(e)}")
            raise
    
    def create_tables(self):
        """Create the database tables"""
        try:
            # Injuries table
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
            
            # Recovery phases table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS recovery_phases (
                    phase_id INTEGER PRIMARY KEY,
                    injury_id INTEGER,
                    phase_number INTEGER,
                    phase_name TEXT,
                    description TEXT,
                    duration TEXT,
                    goals TEXT,
                    restrictions TEXT,
                    recommended_activities TEXT,
                    pain_management TEXT,
                    metadata TEXT,
                    FOREIGN KEY (injury_id) REFERENCES injuries (injury_id)
                )
            ''')
            
            # Treatments table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS treatments (
                    treatment_id INTEGER PRIMARY KEY,
                    injury_id INTEGER,
                    name TEXT,
                    description TEXT,
                    effectiveness_rating REAL,
                    duration TEXT,
                    frequency TEXT,
                    instructions TEXT,
                    precautions TEXT,
                    metadata TEXT,
                    FOREIGN KEY (injury_id) REFERENCES injuries (injury_id)
                )
            ''')
            
            # Progress metrics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_metrics (
                    metric_id INTEGER PRIMARY KEY,
                    injury_id INTEGER,
                    metric_name TEXT,
                    description TEXT,
                    measurement_type TEXT,
                    normal_range TEXT,
                    unit TEXT,
                    metadata TEXT,
                    FOREIGN KEY (injury_id) REFERENCES injuries (injury_id)
                )
            ''')
            
            # User logs table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_logs (
                    log_id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    injury_id INTEGER,
                    log_date TIMESTAMP,
                    pain_level INTEGER,
                    mobility_score INTEGER,
                    activity_level TEXT,
                    treatment_adherence TEXT,
                    notes TEXT,
                    audio_transcript TEXT,
                    metadata TEXT,
                    FOREIGN KEY (injury_id) REFERENCES injuries (injury_id)
                )
            ''')
            
            self.conn.commit()
            logging.info("Successfully created database tables")
            
        except Exception as e:
            logging.error(f"Error creating tables: {str(e)}")
            raise
    
    def fetch_openfda_data(self):
        """Fetch injury data from OpenFDA API"""
        try:
            # Example query for sports-related injuries
            url = "https://api.fda.gov/drug/event.json"
            params = {
                "search": "patient.reaction.reactionmeddrapt:(\"SPORTS INJURY\" OR \"MUSCULOSKELETAL\" OR \"JOINT PAIN\" OR \"BACK PAIN\" OR \"TENDONITIS\" OR \"FRACTURE\" OR \"SPRAIN\" OR \"DISLOCATION\")",
                "limit": 100
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
        except Exception as e:
            logging.error(f"Error fetching OpenFDA data: {str(e)}")
            return []
    
    def fetch_who_data(self):
        """Fetch injury data from WHO API"""
        try:
            # Example query for injury statistics
            url = "https://www.who.int/data/gho/data/themes/topics/injuries"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse WHO data (this is a simplified example)
            # In reality, you'd need to parse the HTML or use their API
            return []
            
        except Exception as e:
            logging.error(f"Error fetching WHO data: {str(e)}")
            return []
    
    def process_injury_data(self, data):
        """Process and insert injury data into the database"""
        try:
            for item in data:
                # Extract relevant information
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
                
                # Insert into database
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
            
            self.conn.commit()
            logging.info(f"Successfully processed and inserted {len(data)} injury records")
            
        except Exception as e:
            logging.error(f"Error processing injury data: {str(e)}")
            raise
    
    def setup(self):
        """Set up the complete database"""
        try:
            self.connect()
            self.create_tables()
            
            # Fetch and process data
            openfda_data = self.fetch_openfda_data()
            who_data = self.fetch_who_data()
            
            # Process and insert data
            self.process_injury_data(openfda_data)
            # self.process_injury_data(who_data)  # Uncomment when WHO data fetching is implemented
            
            logging.info("Database setup completed successfully")
            
        except Exception as e:
            logging.error(f"Error in database setup: {str(e)}")
            raise
        finally:
            if self.conn:
                self.conn.close()

if __name__ == "__main__":
    db = InjuryDatabase()
    db.setup() 