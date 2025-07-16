import os
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.vector_db_path = Path("fitness_app/data/vector_db")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the FAISS index
        self.index = faiss.read_index(str(self.vector_db_path / "injury_recovery.index"))
        
        # Load metadata
        with open(self.vector_db_path / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Initialize DeepSeek model
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base")
        self.llm = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def search(self, query: str, k: int = 5):
        """Search for relevant documents"""
        # Create query embedding
        query_embedding = self.model.encode([query])[0].astype('float32')
        
        # Search the index
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        # Get results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                result = {
                    'metadata': self.metadata[idx],
                    'relevance_score': float(1 / (1 + distance)),  # Convert distance to similarity score
                    'index': int(idx)
                }
                results.append(result)
        
        return results
    
    def get_context(self, results):
        """Get the full context for the results"""
        contexts = []
        for result in results:
            # Load the original data
            category = result['metadata']['category']
            source = result['metadata']['source']
            
            # Find the source file
            data_dir = Path("fitness_app/data") / category
            source_file = next(data_dir.glob(f"*{source.lower().replace(' ', '_')}.json"), None)
            
            if source_file:
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Find the specific item
                    for item in data['data']:
                        if (item.get('title') == result['metadata']['title'] and 
                            item.get('type') == result['metadata']['type']):
                            contexts.append({
                                'content': item.get('content', ''),
                                'metadata': result['metadata'],
                                'relevance_score': result['relevance_score']
                            })
                            break
        
        return contexts
    
    def _generate_answer(self, query: str, contexts: list):
        """Generate an answer using DeepSeek and the contexts"""
        # Combine contexts
        combined_context = "\n\n".join([
            f"Source: {ctx['metadata']['source']}\n"
            f"Title: {ctx['metadata']['title']}\n"
            f"Content: {ctx['content']}"
            for ctx in contexts
        ])
        
        # Create prompt
        prompt = f"""You are a medical and fitness expert. Use the following context to answer the question.
        If the context doesn't contain enough information, say so.

        Context:
        {combined_context}

        Question: {query}

        Answer:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        outputs = self.llm.generate(
            **inputs,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        try:
            answer = response.split("Answer:")[-1].strip()
        except:
            answer = response
        
        return answer
    
    def answer_query(self, query: str, k: int = 5):
        """Answer a query using the RAG system"""
        # Search for relevant documents
        results = self.search(query, k)
        
        # Get context
        contexts = self.get_context(results)
        
        # Format the response
        response = {
            'query': query,
            'sources': [],
            'answer': self._generate_answer(query, contexts)
        }
        
        # Add sources
        for context in contexts:
            response['sources'].append({
                'title': context['metadata']['title'],
                'source': context['metadata']['source'],
                'relevance_score': context['relevance_score']
            })
        
        return response

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example queries
    queries = [
        "What are the symptoms of a rotator cuff injury?",
        "How long does ACL recovery take?",
        "What are the best exercises for lower back pain?"
    ]
    
    # Test the system
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag.answer_query(query)
        print(f"Answer: {response['answer']}")
        print("\nSources:")
        for source in response['sources']:
            print(f"- {source['title']} (from {source['source']})")

if __name__ == "__main__":
    main() 