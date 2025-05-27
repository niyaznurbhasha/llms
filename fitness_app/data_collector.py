from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import time
import random

class FitnessDataCollector:
    def __init__(self, output_dir: str = "data"):
        self.data_dir = Path(output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.categories = {
            "exercises": [],
            "nutrition": [],
            "injuries": [],
            "recovery": [],
            "fitness_knowledge": []
        }
        
        # Initialize sources
        self.sources = self._parse_sources("")
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Quality thresholds
        self.quality_thresholds = {
            "min_text_length": 50,  # Reduced from 100 to allow shorter but valuable content
            "max_text_length": 1000000,  # Keep max length
            "min_confidence_score": 0.5,  # Reduced from 0.7 to allow more good quality content
            "max_similarity_threshold": 0.85,  # Increased from 0.75 to allow more similar but still unique content
            "required_fields": ["title", "content", "source", "category", "reliability"],  # Removed cross_references and practical_examples as required
            "target_items": {
                "nutrition": 25000,  # Keep target numbers
                "exercises": 15000,
                "injuries": 10000,
                "recovery": 10000,
                "fitness_knowledge": 5000
            }
        }
        
        # Create category-specific directories
        for category in self.categories.keys():
            (self.data_dir / category).mkdir(exist_ok=True)
        
        # Setup logging with rotation
        log_path = self.data_dir / 'data_collection.log'
        # Clear existing log file
        if log_path.exists():
            log_path.unlink()
            
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True  # Force reconfiguration of the root logger
        )
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers (except for measurements in exercise descriptions)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
    
    def calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate a quality score for a data item"""
        try:
            score = 0.0
            max_score = 5.0  # Reduced from 7.0 to make scoring more lenient
            
            # Check required fields
            if all(field in item for field in self.quality_thresholds["required_fields"]):
                score += 1.0
            
            # Check text length
            content = item.get("content", "")
            if isinstance(content, str):
                content_length = len(content)
                if self.quality_thresholds["min_text_length"] <= content_length <= self.quality_thresholds["max_text_length"]:
                    score += 1.0
                    # Bonus for longer content
                    if content_length > 500:  # Reduced from 1000
                        score += 0.5
                    if content_length > 2000:  # Reduced from 5000
                        score += 0.5
            
            # Check source reliability
            if item.get("reliability") == "high":
                score += 1.0
            
            # Check for structured data
            if "metadata" in item and isinstance(item["metadata"], dict):
                score += 1.0
            
            # Check for references/citations (optional bonus)
            if "references" in item and len(item["references"]) > 0:
                score += 0.5
            
            # Check for cross-references (optional bonus)
            if "cross_references" in item and len(item["cross_references"]) > 0:
                score += 0.5
            
            # Check for practical examples (optional bonus)
            if "practical_examples" in item and item["practical_examples"]:
                score += 0.5
            
            return min(score / max_score, 1.0)  # Normalize to 0-1 range
        except Exception as e:
            logging.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def is_duplicate(self, new_item: Dict[str, Any], existing_items: List[Dict[str, Any]]) -> bool:
        """Check if an item is too similar to existing items"""
        if not existing_items:
            return False
        
        # Preprocess the new item's content
        new_content = self.preprocess_text(new_item.get("content", ""))
        
        # Preprocess existing items' content
        existing_contents = [self.preprocess_text(item.get("content", "")) for item in existing_items]
        
        # Calculate TF-IDF vectors
        all_contents = [new_content] + existing_contents
        tfidf_matrix = self.vectorizer.fit_transform(all_contents)
        
        # Calculate similarity scores
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Check if any similarity exceeds threshold
        return np.any(similarities > self.quality_thresholds["max_similarity_threshold"])
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        """Validate a data item against quality criteria"""
        try:
            # Check required fields
            if not all(field in item for field in self.quality_thresholds["required_fields"]):
                logging.warning(f"Missing required fields in item from {item.get('source', 'unknown')}")
                return False
            
            # Check content length
            content = item.get("content", "")
            if not isinstance(content, str):
                logging.warning(f"Content is not a string in item from {item.get('source', 'unknown')}")
                return False
                
            if not (self.quality_thresholds["min_text_length"] <= len(content) <= self.quality_thresholds["max_text_length"]):
                logging.warning(f"Content length {len(content)} outside acceptable range for {item.get('source', 'unknown')}")
                return False
            
            # Calculate quality score
            quality_score = self.calculate_quality_score(item)
            if quality_score < self.quality_thresholds["min_confidence_score"]:
                logging.warning(f"Quality score {quality_score} below threshold for {item.get('source', 'unknown')}")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating item from {item.get('source', 'unknown')}: {str(e)}")
            return False
    
    def _parse_sources(self, llm_response: str) -> List[Dict]:
        """Parse LLM response into structured source information"""
        return [
            # Exercise Sources (Public APIs)
            {
                "name": "ExerciseDB API",
                "url": "https://api.exercisedb.io/v1/exercises",
                "type": "api",
                "access_method": "api",
                "data_format": "json",
                "category": "exercises",
                "reliability": "high",
                "description": "Public exercise database API",
                "verification": "public_api",
                "headers": {
                    "X-RapidAPI-Key": "YOUR_API_KEY",
                    "X-RapidAPI-Host": "exercisedb.p.rapidapi.com"
                }
            },
            {
                "name": "WGER Exercise API",
                "url": "https://wger.de/api/v2/exercise/",
                "type": "api",
                "access_method": "api",
                "data_format": "json",
                "category": "exercises",
                "reliability": "high",
                "description": "Open source exercise database API",
                "verification": "public_api"
            },
            # Nutrition Sources (Public APIs)
            {
                "name": "USDA Food Database API",
                "url": "https://api.nal.usda.gov/fdc/v1/foods/search",
                "type": "api",
                "access_method": "api",
                "data_format": "json",
                "category": "nutrition",
                "reliability": "high",
                "description": "Official USDA food database API",
                "verification": "government_api",
                "params": {
                    "api_key": "YOUR_API_KEY",
                    "pageSize": 100
                }
            },
            {
                "name": "Nutritionix API",
                "url": "https://trackapi.nutritionix.com/v2/natural/nutrients",
                "type": "api",
                "access_method": "api",
                "data_format": "json",
                "category": "nutrition",
                "reliability": "high",
                "description": "Nutrition database API",
                "verification": "public_api",
                "headers": {
                    "x-app-id": "YOUR_APP_ID",
                    "x-app-key": "YOUR_API_KEY"
                }
            },
            # Injury Sources (Public APIs)
            {
                "name": "OpenFDA API",
                "url": "https://api.fda.gov/drug/event.json",
                "type": "api",
                "access_method": "api",
                "data_format": "json",
                "category": "injuries",
                "reliability": "high",
                "description": "FDA adverse event reports API",
                "verification": "government_api",
                "params": {
                    "api_key": "YOUR_API_KEY",
                    "limit": 100
                }
            },
            # Academic Sources (Public APIs)
            {
                "name": "PubMed API",
                "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                "type": "api",
                "access_method": "api",
                "data_format": "json",
                "category": "fitness_knowledge",
                "reliability": "high",
                "description": "PubMed research database API",
                "verification": "academic_api",
                "params": {
                    "api_key": "YOUR_API_KEY",
                    "db": "pubmed",
                    "retmode": "json"
                }
            },
            # Open Data Sources
            {
                "name": "Kaggle Fitness Dataset",
                "url": "https://www.kaggle.com/datasets/fmendes/fitness-exercises-with-animations",
                "type": "dataset",
                "access_method": "direct_download",
                "data_format": "csv",
                "category": "exercises",
                "reliability": "high",
                "description": "Fitness exercises dataset with animations",
                "verification": "curated_dataset"
            },
            {
                "name": "UCI Fitness Dataset",
                "url": "https://archive.ics.uci.edu/ml/datasets/Fitness+Exercises",
                "type": "dataset",
                "access_method": "direct_download",
                "data_format": "csv",
                "category": "exercises",
                "reliability": "high",
                "description": "UCI fitness exercises dataset",
                "verification": "academic_dataset"
            }
        ]
    
    def verify_source_reliability(self, source: Dict[str, Any]) -> bool:
        """Verify the reliability of a data source"""
        verification_methods = {
            "government_official": True,  # Government sources are always reliable
            "academic_institution": True,  # Academic institutions are reliable
            "professional_organization": True,  # Professional organizations are reliable
            "peer_reviewed": True,  # Peer-reviewed content is reliable
            "medical_professional": True,  # Medical professionals are reliable
            "professional_curated": True,  # Professionally curated content is reliable
            "international_organization": True,  # International organizations are reliable
            "public_api": True,  # Public APIs are reliable
            "government_api": True,  # Government APIs are reliable
            "academic_api": True,  # Academic APIs are reliable
            "curated_dataset": True,  # Curated datasets are reliable
            "academic_dataset": True  # Academic datasets are reliable
        }
        
        return source.get("verification") in verification_methods
    
    def collect_category_data(self, category: str):
        """Collect data for a specific category with reliability checks"""
        sources = [s for s in self.sources if s["category"] == category]
        for source in sources:
            try:
                # Verify source reliability
                if not self.verify_source_reliability(source):
                    logging.warning(f"Skipping unverified source: {source['name']}")
                    continue
                
                print(f"\nCollecting from {source['name']}...")
                raw_data = []
                
                # Collect data based on access method
                if source["access_method"] == "api":
                    self._collect_from_api(source)
                elif source["access_method"] == "web_scraping":
                    self._collect_from_web(source)
                elif source["access_method"] == "direct_download":
                    self._collect_from_download(source)
                
                # Get collected data for this source
                source_data = [item for item in self.categories[category] 
                             if item.get("source") == source["name"]]
                
                if source_data:
                    # Process and validate data
                    processed_data = []
                    for item in source_data:
                        # Add metadata if not present
                        if "metadata" not in item:
                            item["metadata"] = {
                                "content_length": len(item.get("content", "")),
                                "extraction_method": source["access_method"],
                                "collection_date": datetime.now().isoformat()
                            }
                        
                        # Preprocess content
                        if "content" in item:
                            item["content"] = self.preprocess_text(item["content"])
                        
                        # Validate and check for duplicates
                        if self.validate_item(item) and not self.is_duplicate(item, processed_data):
                            processed_data.append(item)
                    
                    # Update category with processed data
                    self.categories[category] = [item for item in self.categories[category] 
                                              if item.get("source") != source["name"]]
                    self.categories[category].extend(processed_data)
                    
                    # Save intermediate results
                    self.save_category_data(category)
                    
                    print(f"Collected {len(processed_data)} valid items from {source['name']}")
                else:
                    logging.warning(f"No data collected from {source['name']}")
                
            except Exception as e:
                logging.error(f"Error collecting from {source['name']}: {str(e)}")
                # Remove any partial data for this source
                self.categories[category] = [item for item in self.categories[category] 
                                          if item.get("source") != source["name"]]
    
    def save_category_data(self, category: str):
        """Save data for a specific category with quality metrics"""
        if self.categories[category]:
            # Calculate quality metrics
            quality_scores = [self.calculate_quality_score(item) for item in self.categories[category]]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # Save main data file
            filepath = self.data_dir / category / f"{category}_data.json"
            data = {
                "metadata": {
                    "category": category,
                    "collection_date": datetime.now().isoformat(),
                    "total_items": len(self.categories[category]),
                    "average_quality_score": avg_quality,
                    "quality_threshold": self.quality_thresholds["min_confidence_score"],
                    "sources": [s["name"] for s in self.sources if s["category"] == category]
                },
                "data": self.categories[category]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save individual source files with quality metrics
            for source in self.sources:
                if source["category"] == category:
                    source_data = [item for item in self.categories[category] 
                                 if source["name"] in item.get("source", "")]
                    if source_data:
                        source_quality_scores = [self.calculate_quality_score(item) for item in source_data]
                        source_avg_quality = sum(source_quality_scores) / len(source_quality_scores)
                        
                        source_file = self.data_dir / category / f"{source['name'].lower().replace(' ', '_')}.json"
                        with open(source_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                "metadata": {
                                    "source": source["name"],
                                    "collection_date": datetime.now().isoformat(),
                                    "total_items": len(source_data),
                                    "average_quality_score": source_avg_quality,
                                    "quality_threshold": self.quality_thresholds["min_confidence_score"]
                                },
                                "data": source_data
                            }, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(self.categories[category])} items to {category}/ (Avg quality: {avg_quality:.2f})")

    def log_collection_summary(self):
        """Log a summary of data collection progress for each category"""
        logging.info("\n=== Data Collection Summary ===")
        for category in self.categories:
            collected = len(self.categories[category])
            target = self.quality_thresholds["target_items"][category]
            percentage = (collected / target) * 100 if target > 0 else 0
            logging.info(f"{category}: {collected}/{target} items ({percentage:.1f}%)")
        logging.info("=============================\n")

    def collect_from_source(self, source: Dict):
        """Collect data from a specific source based on its type"""
        try:
            if source["access_method"] == "api":
                self._collect_from_api(source)
            elif source["access_method"] == "web_scraping":
                self._collect_from_web(source)
            elif source["access_method"] == "direct_download":
                self._collect_from_download(source)
            
            # Only log success if we actually collected data
            if any(item["source"] == source["name"] for category in self.categories.values() for item in category):
                logging.info(f"Successfully collected data from {source['name']}")
            else:
                logging.warning(f"No data collected from {source['name']}")
        except Exception as e:
            logging.error(f"Error collecting from {source['name']}: {str(e)}")

    def _collect_from_api(self, source: Dict):
        """Collect data from an API source with improved error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            # Add source-specific headers
            if "headers" in source:
                headers.update(source["headers"])
            
            # Get source-specific parameters
            params = source.get("params", {})
            
            # Add pagination parameters if not present
            if "page" not in params and "limit" not in params:
                params["limit"] = 100
            
            # Try to collect multiple pages of data
            page = 1
            total_items = 0
            max_pages = 10  # Limit to prevent excessive requests
            
            while page <= max_pages:
                try:
                    # Add page parameter if needed
                    if "page" not in params:
                        params["page"] = page
                    
                    # Add delay to avoid rate limiting
                    time.sleep(random.uniform(2, 5))
                    
                    response = requests.get(
                        source["url"],
                        headers=headers,
                        params=params,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if data is None:
                                logging.warning(f"Empty response from {source['name']}")
                                break
                            
                            # Process the data
                            processed_data = self._process_api_data(data, source)
                            if processed_data:
                                self.categories[source["category"]].extend(processed_data)
                                total_items += len(processed_data)
                                logging.info(f"Added {len(processed_data)} items from {source['name']} (Page {page})")
                                
                                # Check if we've reached the end of the data
                                if len(processed_data) < params.get("limit", 100):
                                    break
                            else:
                                logging.warning(f"No valid data processed from {source['name']} (Page {page})")
                                break
                            
                        except json.JSONDecodeError as e:
                            logging.warning(f"Response from {source['name']} is not valid JSON: {str(e)}")
                            break
                    
                    page += 1
                    
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Request error on page {page} for {source['name']}: {str(e)}")
                    break
                except Exception as e:
                    logging.error(f"Error processing page {page} for {source['name']}: {str(e)}")
                    break
            
            if total_items > 0:
                logging.info(f"Successfully collected {total_items} items from {source['name']}")
            else:
                logging.warning(f"No data collected from {source['name']}")
            
        except Exception as e:
            logging.error(f"Error in API collection from {source['name']}: {str(e)}")

    def _collect_from_web(self, source: Dict):
        """Collect data through web scraping with improved error handling"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.google.com/',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Try main URL first
            urls_to_try = [source["url"]] + source.get("alternative_urls", [])
            
            for url in urls_to_try:
                try:
                    # Add random delay to avoid rate limiting
                    time.sleep(random.uniform(2, 5))
                    
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract relevant content based on source type
                        content = self._extract_web_content(soup, source)
                        
                        if content and len(content.strip()) > 0:
                            # Clean and structure the content
                            cleaned_content = self._clean_web_content(content)
                            
                            # Get cross references from source definition
                            cross_refs = source.get("cross_references", [])
                            
                            item = {
                                "title": source["name"],
                                "content": cleaned_content,
                                "source": source["name"],
                                "category": source["category"],
                                "reliability": source["reliability"],
                                "url": url,  # Use the successful URL
                                "collection_date": datetime.now().isoformat(),
                                "cross_references": cross_refs,
                                "practical_examples": source.get("practical_examples", False),
                                "metadata": {
                                    "content_length": len(cleaned_content),
                                    "extraction_method": "web_scraping",
                                    "source_type": "web",
                                    "successful_url": url
                                }
                            }
                            
                            if self.validate_item(item):
                                self.categories[source["category"]].append(item)
                                logging.info(f"Successfully collected and processed data from {source['name']} using URL: {url}")
                                return  # Success, exit the function
                            else:
                                logging.warning(f"Collected data from {source['name']} did not pass validation")
                    
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Failed to access {url} for {source['name']}: {str(e)}")
                    continue  # Try next URL
                except Exception as e:
                    logging.error(f"Error processing {url} for {source['name']}: {str(e)}")
                    continue  # Try next URL
            
            # If we get here, all URLs failed
            logging.error(f"All URLs failed for {source['name']}")
            
        except Exception as e:
            logging.error(f"Error in web scraping from {source['name']}: {str(e)}")

    def _clean_web_content(self, content: str) -> str:
        """Clean and structure web content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove common web elements
        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
        content = re.sub(r'&[^;]+;', '', content)  # Remove HTML entities
        
        # Remove navigation elements
        content = re.sub(r'Home|About|Contact|Privacy Policy|Terms of Use', '', content, flags=re.IGNORECASE)
        
        # Remove social media links
        content = re.sub(r'Follow us on|Share this|Tweet|Like|Share', '', content, flags=re.IGNORECASE)
        
        # Remove common web elements
        content = re.sub(r'Cookie Policy|Accept Cookies|Cookie Settings', '', content, flags=re.IGNORECASE)
        
        # Remove multiple newlines and spaces
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r' +', ' ', content)
        
        return content.strip()

    def _extract_web_content(self, soup: BeautifulSoup, source: Dict) -> str:
        """Extract relevant content from web pages based on source type"""
        content = ""
        try:
            # Try different content containers based on source type
            if source["category"] == "exercises":
                # Look for exercise-specific content
                content = self._extract_exercise_content(soup)
            elif source["category"] == "nutrition":
                # Look for nutrition-specific content
                content = self._extract_nutrition_content(soup)
            elif source["category"] == "injuries":
                # Look for injury-specific content
                content = self._extract_injury_content(soup)
            else:
                # Generic content extraction
                content = self._extract_generic_content(soup)
            
            # Special handling for specific sources
            if source["name"] == "British Journal of Sports Medicine":
                content = self._extract_bjsm_content(soup)
            elif source["name"] == "International Journal of Sports Science":
                content = self._extract_ijss_content(soup)
            elif source["name"] == "Journal of Sports Science and Medicine":
                content = self._extract_jssm_content(soup)
            
            if not content:
                # Fallback to generic extraction if specific extraction fails
                content = self._extract_generic_content(soup)
                
        except Exception as e:
            logging.error(f"Error extracting content from {source['name']}: {str(e)}")
        return content

    def _extract_generic_content(self, soup: BeautifulSoup) -> str:
        """Extract generic content from web pages"""
        content = ""
        
        # Try different content containers
        content_containers = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=lambda x: x and ('content' in x.lower() or 'main' in x.lower())),
            soup.find('div', id=lambda x: x and ('content' in x.lower() or 'main' in x.lower()))
        ]
        
        for container in content_containers:
            if container:
                # Remove unwanted elements
                for unwanted in container.find_all(['script', 'style', 'nav', 'footer', 'header']):
                    unwanted.decompose()
                
                content = container.get_text(separator=' ', strip=True)
                if len(content) > 100:  # Only use if we got substantial content
                    break
        
        return content

    def _extract_exercise_content(self, soup: BeautifulSoup) -> str:
        """Extract exercise-specific content"""
        content = ""
        
        # Look for exercise-specific elements
        exercise_elements = soup.find_all(['div', 'section'], class_=lambda x: x and any(
            term in x.lower() for term in ['exercise', 'workout', 'training', 'technique']
        ))
        
        for element in exercise_elements:
            content += element.get_text(separator=' ', strip=True) + "\n"
        
        return content

    def _extract_nutrition_content(self, soup: BeautifulSoup) -> str:
        """Extract nutrition-specific content"""
        content = ""
        
        # Look for nutrition-specific elements
        nutrition_elements = soup.find_all(['div', 'section'], class_=lambda x: x and any(
            term in x.lower() for term in ['nutrition', 'diet', 'food', 'meal']
        ))
        
        for element in nutrition_elements:
            content += element.get_text(separator=' ', strip=True) + "\n"
        
        return content

    def _extract_injury_content(self, soup: BeautifulSoup) -> str:
        """Extract injury-specific content"""
        content = ""
        
        # Look for injury-specific elements
        injury_elements = soup.find_all(['div', 'section'], class_=lambda x: x and any(
            term in x.lower() for term in ['injury', 'recovery', 'rehabilitation', 'treatment']
        ))
        
        for element in injury_elements:
            content += element.get_text(separator=' ', strip=True) + "\n"
        
        return content

    def _extract_bjsm_content(self, soup: BeautifulSoup) -> str:
        """Extract content from British Journal of Sports Medicine"""
        content = ""
        try:
            # Try to find article content
            article = soup.find('article') or soup.find('div', class_='article-content')
            if article:
                # Remove unwanted elements
                for unwanted in article.find_all(['script', 'style', 'nav', 'footer', 'header', 'table']):
                    unwanted.decompose()
                
                # Extract text from all text elements
                for element in article.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']):
                    text = element.get_text(strip=True)
                    if text and len(text) > 10:
                        content += text + "\n"
        except Exception as e:
            logging.error(f"Error extracting BJSM content: {str(e)}")
        return content

    def _extract_ijss_content(self, soup: BeautifulSoup) -> str:
        """Extract content from International Journal of Sports Science"""
        content = ""
        try:
            # Try to find article content
            article = soup.find('div', class_='article-content') or soup.find('div', class_='article')
            if article:
                # Remove unwanted elements
                for unwanted in article.find_all(['script', 'style', 'nav', 'footer', 'header', 'table']):
                    unwanted.decompose()
                
                # Extract text from all text elements
                for element in article.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']):
                    text = element.get_text(strip=True)
                    if text and len(text) > 10:
                        content += text + "\n"
        except Exception as e:
            logging.error(f"Error extracting IJSS content: {str(e)}")
        return content

    def _extract_jssm_content(self, soup: BeautifulSoup) -> str:
        """Extract content from Journal of Sports Science and Medicine"""
        content = ""
        try:
            # Try to find article content
            article = soup.find('div', class_='article-content') or soup.find('div', class_='article')
            if article:
                # Remove unwanted elements
                for unwanted in article.find_all(['script', 'style', 'nav', 'footer', 'header', 'table']):
                    unwanted.decompose()
                
                # Extract text from all text elements
                for element in article.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']):
                    text = element.get_text(strip=True)
                    if text and len(text) > 10:
                        content += text + "\n"
        except Exception as e:
            logging.error(f"Error extracting JSSM content: {str(e)}")
        return content

    def _process_api_data(self, data: Dict, source: Dict) -> List[Dict]:
        """Process data from API responses into smaller chunks"""
        processed_items = []
        try:
            if data is None:
                logging.warning(f"No data received from API {source['name']}")
                return processed_items

            # Handle different API response formats
            items_to_process = []
            if isinstance(data, list):
                items_to_process = data
            elif isinstance(data, dict):
                if "results" in data:
                    items_to_process = data["results"]
                elif "data" in data:
                    items_to_process = data["data"]
                elif "items" in data:
                    items_to_process = data["items"]
                else:
                    items_to_process = [data]
            else:
                logging.warning(f"Unexpected API response format from {source['name']}")
                return processed_items

            for item in items_to_process:
                try:
                    if not isinstance(item, (dict, str)):
                        continue

                    # Extract and chunk content based on category
                    chunks = self._chunk_content(item, source["category"])
                    
                    for chunk in chunks:
                        processed_item = {
                            "title": chunk.get("title", source["name"]),
                            "content": chunk["content"],
                            "source": source["name"],
                            "category": source["category"],
                            "reliability": source["reliability"],
                            "url": source["url"],
                            "collection_date": datetime.now().isoformat(),
                            "metadata": {
                                "content_length": len(chunk["content"]),
                                "extraction_method": "api",
                                "chunk_type": chunk.get("type", "general"),
                                "original_format": "json" if isinstance(item, dict) else "text"
                            }
                        }
                        
                        if self.validate_item(processed_item):
                            processed_items.append(processed_item)
                            logging.info(f"Successfully processed chunk from {source['name']}")

                except Exception as item_e:
                    logging.error(f"Error processing individual item from {source['name']}: {str(item_e)}")
                    continue

        except Exception as e:
            logging.error(f"Error processing API data from {source['name']}: {str(e)}")
        return processed_items

    def _chunk_content(self, content: Union[Dict, str], category: str) -> List[Dict]:
        """Break down content into meaningful chunks based on category"""
        chunks = []
        
        if isinstance(content, str):
            # Split text into sentences
            sentences = nltk.sent_tokenize(content)
            current_chunk = []
            current_length = 0
            
            # Different max lengths for different categories
            max_lengths = {
                "exercises": 500,      # Shorter for exercise descriptions
                "nutrition": 300,      # Shorter for nutrition facts
                "injuries": 800,       # Longer for detailed injury descriptions
                "recovery": 600,       # Medium for recovery methods
                "fitness_knowledge": 1000  # Longest for general knowledge
            }
            max_length = max_lengths.get(category, 1000)
            
            for sentence in sentences:
                if current_length + len(sentence) > max_length:
                    if current_chunk:
                        chunks.append({
                            "content": " ".join(current_chunk),
                            "type": f"{category}_text_chunk"
                        })
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            
            if current_chunk:
                chunks.append({
                    "content": " ".join(current_chunk),
                    "type": f"{category}_text_chunk"
                })
        else:
            # Process structured data based on category
            if category == "exercises":
                chunks.extend(self._chunk_exercise_data(content))
            elif category == "nutrition":
                chunks.extend(self._chunk_nutrition_data(content))
            elif category == "injuries":
                chunks.extend(self._chunk_injury_data(content))
            elif category == "recovery":
                chunks.extend(self._chunk_recovery_data(content))
            else:
                chunks.extend(self._chunk_general_data(content))
        
        return chunks

    def _chunk_exercise_data(self, data: Dict) -> List[Dict]:
        """Break down exercise data into individual exercises with specialized chunks"""
        chunks = []
        
        # Handle different exercise data structures
        if "exercises" in data:
            exercises = data["exercises"]
        elif "items" in data:
            exercises = data["items"]
        else:
            exercises = [data]
        
        for exercise in exercises:
            if isinstance(exercise, dict):
                # Extract exercise details
                name = exercise.get("name", "")
                description = exercise.get("description", "")
                instructions = exercise.get("instructions", [])
                muscles = exercise.get("muscles", [])
                equipment = exercise.get("equipment", [])
                difficulty = exercise.get("difficulty", "")
                variations = exercise.get("variations", [])
                precautions = exercise.get("precautions", [])
                
                # Basic exercise info chunk (short)
                if name and description:
                    chunks.append({
                        "title": name,
                        "content": f"Exercise: {name}\nDescription: {description}",
                        "type": "exercise_basic_info",
                        "metadata": {
                            "difficulty": difficulty,
                            "equipment": equipment
                        }
                    })
                
                # Detailed instructions chunk (medium)
                if instructions:
                    # Group instructions into sets of 3-4 steps
                    for i in range(0, len(instructions), 3):
                        step_group = instructions[i:i+3]
                        chunks.append({
                            "title": f"{name} Instructions {i//3 + 1}",
                            "content": f"Exercise: {name}\nInstructions:\n" + "\n".join(f"{j+1}. {step}" for j, step in enumerate(step_group, i)),
                            "type": "exercise_instructions"
                        })
                
                # Muscles and equipment chunk (short)
                if muscles or equipment:
                    chunks.append({
                        "title": f"{name} Muscles and Equipment",
                        "content": f"Exercise: {name}\nTarget Muscles: {', '.join(muscles)}\nEquipment: {', '.join(equipment)}",
                        "type": "exercise_muscles_equipment"
                    })
                
                # Variations chunk (medium)
                if variations:
                    chunks.append({
                        "title": f"{name} Variations",
                        "content": f"Exercise: {name}\nVariations:\n" + "\n".join(f"- {var}" for var in variations),
                        "type": "exercise_variations"
                    })
                
                # Precautions chunk (short)
                if precautions:
                    chunks.append({
                        "title": f"{name} Precautions",
                        "content": f"Exercise: {name}\nPrecautions:\n" + "\n".join(f"- {prec}" for prec in precautions),
                        "type": "exercise_precautions"
                    })
        
        return chunks

    def _chunk_nutrition_data(self, data: Dict) -> List[Dict]:
        """Break down nutrition data into specialized chunks"""
        chunks = []
        
        # Handle different nutrition data structures
        if "foods" in data:
            items = data["foods"]
        elif "nutrients" in data:
            items = data["nutrients"]
        else:
            items = [data]
        
        for item in items:
            if isinstance(item, dict):
                # Extract nutrition details
                name = item.get("name", "")
                nutrients = item.get("nutrients", {})
                serving = item.get("serving_size", "")
                calories = item.get("calories", "")
                macros = item.get("macronutrients", {})
                vitamins = item.get("vitamins", {})
                minerals = item.get("minerals", {})
                
                # Basic nutrition info chunk (short)
                if name and calories:
                    chunks.append({
                        "title": name,
                        "content": f"Food: {name}\nServing Size: {serving}\nCalories: {calories}",
                        "type": "nutrition_basic_info"
                    })
                
                # Macronutrients chunk (short)
                if macros:
                    chunks.append({
                        "title": f"{name} Macronutrients",
                        "content": f"Food: {name}\nMacronutrients:\n" + "\n".join(f"- {k}: {v}" for k, v in macros.items()),
                        "type": "nutrition_macros"
                    })
                
                # Vitamins chunk (short)
                if vitamins:
                    chunks.append({
                        "title": f"{name} Vitamins",
                        "content": f"Food: {name}\nVitamins:\n" + "\n".join(f"- {k}: {v}" for k, v in vitamins.items()),
                        "type": "nutrition_vitamins"
                    })
                
                # Minerals chunk (short)
                if minerals:
                    chunks.append({
                        "title": f"{name} Minerals",
                        "content": f"Food: {name}\nMinerals:\n" + "\n".join(f"- {k}: {v}" for k, v in minerals.items()),
                        "type": "nutrition_minerals"
                    })
                
                # Detailed nutrients chunk (medium)
                if nutrients:
                    # Group nutrients into categories
                    nutrient_categories = {
                        "Proteins": ["protein", "amino acids"],
                        "Fats": ["fat", "fatty acids"],
                        "Carbohydrates": ["carb", "sugar", "fiber"],
                        "Other": []
                    }
                    
                    for category, keywords in nutrient_categories.items():
                        category_nutrients = {
                            k: v for k, v in nutrients.items()
                            if any(keyword in k.lower() for keyword in keywords)
                        }
                        if category_nutrients:
                            chunks.append({
                                "title": f"{name} {category}",
                                "content": f"Food: {name}\n{category}:\n" + "\n".join(f"- {k}: {v}" for k, v in category_nutrients.items()),
                                "type": f"nutrition_{category.lower()}"
                            })
        
        return chunks

    def _chunk_injury_data(self, data: Dict) -> List[Dict]:
        """Break down injury data into specialized chunks"""
        chunks = []
        
        # Handle different injury data structures
        if "injuries" in data:
            injuries = data["injuries"]
        elif "conditions" in data:
            injuries = data["conditions"]
        else:
            injuries = [data]
        
        for injury in injuries:
            if isinstance(injury, dict):
                # Extract injury details
                name = injury.get("name", "")
                description = injury.get("description", "")
                symptoms = injury.get("symptoms", [])
                treatment = injury.get("treatment", [])
                prevention = injury.get("prevention", [])
                risk_factors = injury.get("risk_factors", [])
                recovery_time = injury.get("recovery_time", "")
                severity = injury.get("severity", "")
                
                # Basic injury info chunk (short)
                if name and description:
                    chunks.append({
                        "title": name,
                        "content": f"Injury: {name}\nDescription: {description}\nSeverity: {severity}\nRecovery Time: {recovery_time}",
                        "type": "injury_basic_info"
                    })
                
                # Symptoms chunk (short)
                if symptoms:
                    chunks.append({
                        "title": f"{name} Symptoms",
                        "content": f"Injury: {name}\nSymptoms:\n" + "\n".join(f"- {symptom}" for symptom in symptoms),
                        "type": "injury_symptoms"
                    })
                
                # Treatment chunk (medium)
                if treatment:
                    # Group treatments into phases if available
                    if isinstance(treatment, dict):
                        for phase, steps in treatment.items():
                            chunks.append({
                                "title": f"{name} {phase} Treatment",
                                "content": f"Injury: {name}\n{phase} Treatment:\n" + "\n".join(f"- {step}" for step in steps),
                                "type": "injury_treatment_phase"
                            })
                    else:
                        chunks.append({
                            "title": f"{name} Treatment",
                            "content": f"Injury: {name}\nTreatment:\n" + "\n".join(f"- {step}" for step in treatment),
                            "type": "injury_treatment"
                        })
                
                # Prevention chunk (short)
                if prevention:
                    chunks.append({
                        "title": f"{name} Prevention",
                        "content": f"Injury: {name}\nPrevention:\n" + "\n".join(f"- {step}" for step in prevention),
                        "type": "injury_prevention"
                    })
                
                # Risk factors chunk (short)
                if risk_factors:
                    chunks.append({
                        "title": f"{name} Risk Factors",
                        "content": f"Injury: {name}\nRisk Factors:\n" + "\n".join(f"- {factor}" for factor in risk_factors),
                        "type": "injury_risk_factors"
                    })
        
        return chunks

    def _chunk_recovery_data(self, data: Dict) -> List[Dict]:
        """Break down recovery data into specialized chunks"""
        chunks = []
        
        # Handle different recovery data structures
        if "methods" in data:
            methods = data["methods"]
        elif "techniques" in data:
            methods = data["techniques"]
        else:
            methods = [data]
        
        for method in methods:
            if isinstance(method, dict):
                # Extract recovery details
                name = method.get("name", "")
                description = method.get("description", "")
                steps = method.get("steps", [])
                benefits = method.get("benefits", [])
                duration = method.get("duration", "")
                frequency = method.get("frequency", "")
                contraindications = method.get("contraindications", [])
                equipment = method.get("equipment", [])
                
                # Basic recovery info chunk (short)
                if name and description:
                    chunks.append({
                        "title": name,
                        "content": f"Recovery Method: {name}\nDescription: {description}\nDuration: {duration}\nFrequency: {frequency}",
                        "type": "recovery_basic_info"
                    })
                
                # Steps chunk (medium)
                if steps:
                    # Group steps into phases if available
                    if isinstance(steps, dict):
                        for phase, phase_steps in steps.items():
                            chunks.append({
                                "title": f"{name} {phase} Steps",
                                "content": f"Recovery Method: {name}\n{phase} Steps:\n" + "\n".join(f"- {step}" for step in phase_steps),
                                "type": "recovery_steps_phase"
                            })
                    else:
                        chunks.append({
                            "title": f"{name} Steps",
                            "content": f"Recovery Method: {name}\nSteps:\n" + "\n".join(f"- {step}" for step in steps),
                            "type": "recovery_steps"
                        })
                
                # Benefits chunk (short)
                if benefits:
                    chunks.append({
                        "title": f"{name} Benefits",
                        "content": f"Recovery Method: {name}\nBenefits:\n" + "\n".join(f"- {benefit}" for benefit in benefits),
                        "type": "recovery_benefits"
                    })
                
                # Equipment and contraindications chunk (short)
                if equipment or contraindications:
                    chunks.append({
                        "title": f"{name} Equipment and Contraindications",
                        "content": f"Recovery Method: {name}\nEquipment: {', '.join(equipment)}\nContraindications:\n" + "\n".join(f"- {contra}" for contra in contraindications),
                        "type": "recovery_equipment_contraindications"
                    })
        
        return chunks

    def _chunk_general_data(self, data: Dict) -> List[Dict]:
        """Break down general data into specialized chunks"""
        chunks = []
        
        if isinstance(data, dict):
            # Try to identify sections in the data
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 50:
                    # Split long text into smaller chunks
                    sentences = nltk.sent_tokenize(value)
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        if current_length + len(sentence) > 800:  # General knowledge chunks can be longer
                            if current_chunk:
                                chunks.append({
                                    "title": f"{key.replace('_', ' ').title()} Part {len(chunks) + 1}",
                                    "content": " ".join(current_chunk),
                                    "type": "general_section"
                                })
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence)
                    
                    if current_chunk:
                        chunks.append({
                            "title": f"{key.replace('_', ' ').title()} Part {len(chunks) + 1}",
                            "content": " ".join(current_chunk),
                            "type": "general_section"
                        })
                elif isinstance(value, list):
                    # Process list items in groups
                    for i in range(0, len(value), 5):  # Group every 5 items
                        group = value[i:i+5]
                        if all(isinstance(item, str) for item in group):
                            chunks.append({
                                "title": f"{key.replace('_', ' ').title()} Group {i//5 + 1}",
                                "content": "\n".join(f"- {item}" for item in group),
                                "type": "general_list"
                            })
        
        return chunks

    def _collect_from_download(self, source: Dict):
        """Download and process data from direct download sources"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*'
            }
            
            response = requests.get(source["url"], headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            if response.status_code == 200:
                # Create a temporary file to store the download
                temp_file = self.data_dir / source["category"] / f"temp_{source['name'].lower().replace(' ', '_')}.{source['data_format']}"
                
                # Download the file in chunks
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                try:
                    # Process the downloaded file
                    self._process_downloaded_file(temp_file, source)
                finally:
                    # Clean up the temporary file
                    if temp_file.exists():
                        temp_file.unlink()
                        
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error downloading from {source['name']}: {str(e)}")
        except Exception as e:
            logging.error(f"Error downloading from {source['name']}: {str(e)}")

    def _process_downloaded_file(self, filepath: Path, source: Dict):
        """Process downloaded files based on their format"""
        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed_items = self._process_api_data(data, source)
                    if processed_items:
                        self.categories[source["category"]].extend(processed_items)
                        logging.info(f"Successfully processed JSON file from {source['name']}")
                    else:
                        logging.warning(f"No valid data processed from JSON file {source['name']}")
                        
            elif filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
                processed_items = []
                for _, row in df.iterrows():
                    item = {
                        "title": str(row.get('title', source["name"])),
                        "content": str(row.to_dict()),
                        "source": source["name"],
                        "category": source["category"],
                        "reliability": source["reliability"],
                        "url": source["url"],
                        "collection_date": datetime.now().isoformat(),
                        "cross_references": source.get("cross_references", []),
                        "practical_examples": source.get("practical_examples", False),
                        "metadata": {
                            "content_length": len(str(row.to_dict())),
                            "extraction_method": "csv",
                            "columns": list(df.columns)
                        }
                    }
                    if self.validate_item(item):
                        processed_items.append(item)
                
                if processed_items:
                    self.categories[source["category"]].extend(processed_items)
                    logging.info(f"Successfully processed CSV file from {source['name']}")
                else:
                    logging.warning(f"No valid data processed from CSV file {source['name']}")
                    
            elif filepath.suffix == '.pdf':
                # For PDF files, we'll just store the file path for now
                item = {
                    "title": source["name"],
                    "content": f"PDF file downloaded from {source['url']}",
                    "source": source["name"],
                    "category": source["category"],
                    "reliability": source["reliability"],
                    "url": source["url"],
                    "collection_date": datetime.now().isoformat(),
                    "cross_references": source.get("cross_references", []),
                    "practical_examples": source.get("practical_examples", False),
                    "metadata": {
                        "file_path": str(filepath),
                        "file_type": "pdf",
                        "extraction_method": "direct_download"
                    }
                }
                if self.validate_item(item):
                    self.categories[source["category"]].append(item)
                    logging.info(f"Successfully processed PDF file from {source['name']}")
                else:
                    logging.warning(f"PDF file from {source['name']} did not pass validation")
                    
        except Exception as e:
            logging.error(f"Error processing downloaded file from {source['name']}: {str(e)}")

if __name__ == "__main__":
    # Create data directory structure
    data_dir = Path("fitness_app/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = FitnessDataCollector()
    
    # Collect data for each category
    categories = ["exercises", "nutrition", "injuries", "recovery", "fitness_knowledge"]
    for category in categories:
        print(f"\nCollecting {category} data...")
        collector.collect_category_data(category)
    
    # Log final summary
    collector.log_collection_summary() 