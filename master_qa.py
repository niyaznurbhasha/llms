#!/usr/bin/env python3
"""
Master Q&A System: Audio File Processing with Optimized Transcription and Response Pipeline
"""

import os
import numpy as np
import torch
import whisper
import webrtcvad
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging
import gc
from functools import lru_cache
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_system.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MasterQA:
    # Constants for better performance
    SAMPLE_RATE = 16000
    CHANNELS = 1
    VAD_AGGRESSIVENESS = 3
    MAX_WORKERS = 3

    def __init__(
        self,
        model_name="deepseek-ai/deepseek-llm-7b-base",
        offload_folder="model_offload",
        vector_db_path="vector_db",
        use_rag=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Master Q&A system with optimized settings
        """
        self.device = device
        self.use_rag = use_rag
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)
        
        # Initialize models and components
        self._initialize_models(model_name, offload_folder, vector_db_path)
        
        # Enable garbage collection with optimized settings
        gc.enable()
        gc.set_threshold(100, 5, 5)

    def _initialize_models(self, model_name, offload_folder, vector_db_path):
        """Initialize all required models with optimized settings"""
        # Initialize Whisper model with optimized settings
        self.whisper_model = whisper.load_model(
            "large-v3",
            device=self.device,
            download_root=offload_folder
        )
        
        # Initialize punctuation correction
        self.punct = pipeline(
            "text2text-generation",
            model="flexudy/t5-small-wav2vec2-grammar-fixer",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize spec rewriter for better question structure
        self.rewriter = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if torch.cuda.is_available() else -1
        )
        
        if self.use_rag:
            from topic_llm import TopicLLM
            
            self.llm = TopicLLM(
                model_name=model_name,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                offload_folder=offload_folder,
                vector_db_path=vector_db_path,
                system_prompt=self._get_system_prompt()
            )
            
            self.llm.create_topic_collection("general_qa")
        else:
            # Initialize basic LLM with optimized settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=offload_folder
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder=offload_folder,
                offload_state_dict=True,
                offload_buffers=True
            )
            self.system_prompt = self._get_system_prompt()

    def has_speech(self, pcm):
        """Check if audio contains speech using VAD"""
        raw = pcm.tobytes()
        frame_sz = int(self.SAMPLE_RATE * 30/1000) * 2
        for i in range(0, len(raw), frame_sz):
            chunk = raw[i:i+frame_sz]
            if len(chunk) < frame_sz:
                break
            if self.vad.is_speech(chunk, self.SAMPLE_RATE):
                return True
        return False

    def transcribe_audio(self, audio_path):
        """Transcribe audio file with optimized settings"""
        try:
            # Load and normalize audio
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
            
            # Convert to int16 for VAD
            audio_int16 = (audio * 32768).astype(np.int16)
            
            # Check for speech
            if not self.has_speech(audio_int16):
                logging.warning("No speech detected in audio file")
                return ""
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio,
                fp16=torch.cuda.is_available(),
                language="en"
            )
            
            raw_text = result.get("text", "").strip()
            if not raw_text:
                return ""
            
            # Apply punctuation correction
            corrected = self.punct("fix: " + raw_text)[0]["generated_text"].strip()
            
            # Rewrite for better structure
            rewritten = self.rewrite_question(corrected)
            
            return rewritten
            
        except Exception as e:
            logging.error(f"Error in transcription: {str(e)}")
            return ""

    def rewrite_question(self, text):
        """Rewrite question for better structure"""
        prompt = (
            "Rewrite this spoken question into a clear, well-structured question.\n"
            "Maintain the original meaning but improve clarity and structure.\n\n"
            f"{text}"
        )
        out = self.rewriter(prompt, max_length=512, do_sample=False)[0]["generated_text"]
        return out.strip()

    def get_response(self, question):
        """Get response from LLM"""
        if self.use_rag:
            answer, context = self.llm.query_topic(
                "general_qa",
                question,
                num_results=3
            )
            return answer, context
        else:
            # Format prompt with system message
            prompt = f"{self.system_prompt}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the answer part
            answer = response.split("Answer:")[-1].strip()
            return answer, None

    @lru_cache(maxsize=1)
    def _get_system_prompt(self):
        """Get system prompt for the LLM"""
        return (
            "You are a helpful AI assistant. Provide clear, accurate, and informative "
            "responses to questions. If you're unsure about something, acknowledge the "
            "limitations of your knowledge."
        )

    def process_audio_file(self, audio_path):
        """Process audio file and get response"""
        # Transcribe audio
        question = self.transcribe_audio(audio_path)
        if not question:
            return "Could not transcribe audio or no speech detected."
        
        logging.info(f"Transcribed question: {question}")
        
        # Get response
        answer, context = self.get_response(question)
        
        return {
            "question": question,
            "answer": answer,
            "context": context
        }

def main():
    # Example usage
    qa_system = MasterQA(
        use_rag=True,  # Enable RAG for better responses
        offload_folder="model_offload"
    )
    
    # Process audio file
    audio_path = "input.wav"  # Replace with your audio file path
    result = qa_system.process_audio_file(audio_path)
    
    print("\nQuestion:", result["question"])
    print("\nAnswer:", result["answer"])
    if result["context"]:
        print("\nRelevant Context:")
        for item in result["context"]:
            print(f"- {item['content']}")

if __name__ == "__main__":
    main() 