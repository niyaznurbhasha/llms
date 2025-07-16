import sounddevice as sd
import numpy as np
import wave
import threading
import queue
import time
import os
from datetime import datetime
import json
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import io
import torchaudio
import webrtcvad
from speechbrain.pretrained import EncoderClassifier
import whisper
from concurrent.futures import ThreadPoolExecutor
import gc
from functools import lru_cache
import tempfile
import soundfile as sf
import re
from collections import deque
import hashlib
import sqlite3
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversational_assistant.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ConversationMemory:
    """Advanced conversation memory with SQLite storage and semantic search"""
    
    def __init__(self, db_path="conversation_memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for conversation memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                context_summary TEXT,
                importance_score REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create session metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                topic_summary TEXT,
                user_preferences TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON conversations(content_hash)')
        
        conn.commit()
        conn.close()
    
    def add_message(self, session_id: str, role: str, content: str, context_summary: str = None, importance_score: float = 1.0):
        """Add a message to conversation memory"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (session_id, timestamp, role, content, content_hash, context_summary, importance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, timestamp, role, content, content_hash, context_summary, importance_score))
        
        conn.commit()
        conn.close()
    
    def get_recent_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, timestamp, importance_score
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'importance_score': row[3]
            })
        
        conn.close()
        return list(reversed(results))  # Return in chronological order
    
    def get_semantic_context(self, query: str, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get semantically relevant context based on query"""
        # Simple keyword-based search for now
        # In a production system, you'd use embeddings for semantic search
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search for messages containing keywords from the query
        keywords = query.lower().split()
        placeholders = ','.join(['?' for _ in keywords])
        
        cursor.execute(f'''
            SELECT role, content, timestamp, importance_score
            FROM conversations
            WHERE session_id = ? AND (
                {' OR '.join([f'LOWER(content) LIKE ?' for _ in keywords])}
            )
            ORDER BY importance_score DESC, timestamp DESC
            LIMIT ?
        ''', (session_id, *[f'%{keyword}%' for keyword in keywords], limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'importance_score': row[3]
            })
        
        conn.close()
        return results
    
    def create_session(self, session_id: str, topic_summary: str = None, user_preferences: str = None):
        """Create a new conversation session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO session_metadata (session_id, start_time, topic_summary, user_preferences)
            VALUES (?, ?, ?, ?)
        ''', (session_id, datetime.now().isoformat(), topic_summary, user_preferences))
        
        conn.commit()
        conn.close()
    
    def end_session(self, session_id: str, topic_summary: str = None):
        """End a conversation session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE session_metadata
            SET end_time = ?, topic_summary = ?
            WHERE session_id = ?
        ''', (datetime.now().isoformat(), topic_summary, session_id))
        
        conn.commit()
        conn.close()

class ConversationalAssistant:
    """Advanced conversational assistant with memory and follow-up handling"""
    
    # Audio processing constants
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = np.int16
    FRAME_MS = 20
    SILENCE_THRESHOLD = 0.75
    MAX_QUEUE_SIZE = 50
    MAX_WORKERS = 3
    VAD_AGGRESSIVENESS = 3
    MAX_SILENCE_MS = 1000
    BUFFER_SIZE = 1024

    def __init__(
        self,
        model_name="deepseek-ai/deepseek-coder-6.7b-base",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        offload_folder="model_offload",
        vector_db_path="vector_db",
        use_rag=True,
        voice_ref_path="sample.wav",
        device="cuda" if torch.cuda.is_available() else "cpu",
        input_device=1,
        session_id=None
    ):
        """
        Initialize the Conversational Assistant
        """
        self.device = device
        self.use_rag = use_rag
        self.input_device = input_device
        
        # Generate session ID if not provided
        if session_id is None:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.session_id = session_id
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        self.memory.create_session(self.session_id)
        
        # Initialize conversation state
        self.current_topic = None
        self.user_preferences = {}
        self.conversation_flow = deque(maxlen=20)
        
        # Initialize audio settings
        self.audio_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.bytes_per_frame = int(self.SAMPLE_RATE * self.FRAME_MS / 1000) * 2
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)
        self.vad_buffer = bytearray()
        
        # Initialize models and components
        self._initialize_models(model_name, embedding_model, offload_folder, vector_db_path)
        self._initialize_speaker_verification(voice_ref_path)
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        
        # Start VAD processing thread
        self.vad_thread = threading.Thread(target=self._vad_consumer, daemon=True)
        self.vad_thread.start()
        
        # Enable garbage collection
        gc.enable()
        gc.set_threshold(100, 5, 5)

    @staticmethod
    def list_audio_devices():
        """List all available audio input devices"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_info = {
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate'],
                    'is_bluetooth': 'bluetooth' in device['name'].lower() or 'bth' in device['name'].lower(),
                    'is_hfp': 'hands-free' in device['name'].lower() or 'hfp' in device['name'].lower()
                }
                input_devices.append(device_info)
        return input_devices

    @staticmethod
    def test_audio_device(device_idx):
        """Test if an audio device can be opened"""
        try:
            device_info = sd.query_devices(device_idx)
            logging.info(f"Testing device: {device_info['name']}")
            
            with sd.InputStream(
                device=device_idx,
                channels=2,
                samplerate=44100,
                dtype=np.int16,
                blocksize=1024,
                latency='high'
            ) as stream:
                logging.info("Successfully opened test stream")
                return True
        except Exception as e:
            logging.error(f"Device test failed: {str(e)}")
            return False

    def setup_audio_stream(self):
        """Set up audio stream with 44100 Hz, 2 channels"""
        try:
            if self.input_device is None:
                devices = self.list_audio_devices()
                if not devices:
                    raise RuntimeError("No input devices found")
                self.input_device = devices[0]['index']
                logging.info(f"Using default input device: {sd.query_devices(self.input_device)['name']}")

            if not self.test_audio_device(self.input_device):
                raise RuntimeError("Device test failed")

            sample_rate = 44100
            channels = 2
            logging.info(f"Opening stream with sample_rate={sample_rate}, channels={channels}")

            try:
                self.stream = sd.InputStream(
                    device=self.input_device,
                    channels=channels,
                    samplerate=sample_rate,
                    dtype=np.int16,
                    callback=self._audio_callback,
                    blocksize=1024,
                    latency='high'
                )
                self.stream.start()
                logging.info(f"Successfully started audio stream with device: {sd.query_devices(self.input_device)['name']}")
                self.SAMPLE_RATE = sample_rate
                self.CHANNELS = channels
                return True
            except Exception as e:
                logging.error(f"Failed to initialize stream: {str(e)}")
                return False
        except Exception as e:
            logging.error(f"Error setting up audio stream: {str(e)}")
            return False

    def _initialize_models(self, model_name, embedding_model, offload_folder, vector_db_path):
        """Initialize all required models"""
        # Initialize Whisper model
        self.whisper_model = whisper.load_model(
            "base",
            device=self.device,
            download_root=offload_folder
        )
        
        if self.use_rag:
            try:
                from topic_llm import TopicLLM
                
                self.llm = TopicLLM(
                    model_name=model_name,
                    embedding_model=embedding_model,
                    offload_folder=offload_folder,
                    vector_db_path=vector_db_path,
                    system_prompt=self._get_system_prompt()
                )
                
                self.llm.create_topic_collection("conversational_assistant")
                self._initialize_conversation_data()
            except ImportError:
                logging.warning("TopicLLM not available, falling back to basic LLM")
                self.use_rag = False
        
        if not self.use_rag:
            # Initialize basic LLM
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

    def _initialize_speaker_verification(self, voice_ref_path):
        """Initialize speaker verification system"""
        self.spk_enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )
        
        if os.path.exists(voice_ref_path):
            ref_waveform, _ = torchaudio.load(voice_ref_path)
            if ref_waveform.shape[0] > 1:
                ref_waveform = torch.mean(ref_waveform, dim=0, keepdim=True)
            self.ref_emb = self.spk_enc.encode_batch(ref_waveform)
        else:
            logging.warning(f"Voice reference file {voice_ref_path} not found. Speaker verification disabled.")
            self.ref_emb = None

    def _initialize_conversation_data(self):
        """Initialize conversation data for RAG"""
        if not self.use_rag:
            return
            
        data_dir = Path("conversation_data")
        if data_dir.exists() and any(data_dir.iterdir()):
            for data_file in data_dir.glob("*.json"):
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        self.llm.add_documents("conversational_assistant", data)
                except Exception as e:
                    logging.error(f"Error loading data file {data_file}: {str(e)}")

    def _vad_consumer(self):
        """Process audio stream using VAD"""
        current_segment = []
        silence_counter = 0
        frame_ms = 20
        sample_rate = self.SAMPLE_RATE
        bytes_per_sample = 2
        frame_size = int(sample_rate * frame_ms / 1000)
        self.bytes_per_frame = frame_size * bytes_per_sample
        max_silence_frames = int(self.MAX_SILENCE_MS / frame_ms)
        
        while True:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                data = chunk if isinstance(chunk, bytes) else chunk.tobytes()
                self.vad_buffer += data

                while len(self.vad_buffer) >= self.bytes_per_frame:
                    frame = self.vad_buffer[:self.bytes_per_frame]
                    self.vad_buffer = self.vad_buffer[self.bytes_per_frame:]

                    if len(frame) != self.bytes_per_frame:
                        continue
                    
                    try:
                        is_speech = self.vad.is_speech(frame, sample_rate)
                    except Exception as e:
                        logging.error(f"VAD error: {str(e)}")
                        continue

                    if is_speech:
                        current_segment.append(frame)
                        silence_counter = 0
                    elif current_segment:
                        silence_counter += 1
                        if silence_counter > max_silence_frames:
                            audio_data = b"".join(current_segment)
                            current_segment = []
                            silence_counter = 0
                            self.executor.submit(self._process_segment, audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in VAD consumer: {str(e)}")
                continue

    def safe_delete(self, filename, retries=3, delay=0.2):
        """Safely delete temporary files"""
        for _ in range(retries):
            try:
                os.unlink(filename)
                return
            except Exception as e:
                time.sleep(delay)
        logging.warning(f"Could not delete temporary file {filename} after {retries} retries.")

    def _process_segment(self, audio_bytes):
        """Process a detected speech segment"""
        temp_files = []
        try:
            if self.ref_emb is not None:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, audio_array, self.SAMPLE_RATE, subtype='PCM_16')
                    temp_files.append(temp_file.name)
                    
                    waveform, sr = torchaudio.load(temp_file.name)
                    if sr != self.SAMPLE_RATE:
                        waveform = torchaudio.functional.resample(waveform, sr, self.SAMPLE_RATE)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    seg_emb = self.spk_enc.encode_batch(waveform)
                    similarity = torch.cosine_similarity(self.ref_emb, seg_emb)
                    if similarity.mean().item() > self.SILENCE_THRESHOLD:
                        return

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                sf.write(temp_file.name, audio_array, self.SAMPLE_RATE, subtype='PCM_16')
                temp_files.append(temp_file.name)
                
                try:
                    result = self.whisper_model.transcribe(temp_file.name)
                    if result["text"].strip():
                        self.executor.submit(self._handle_conversation, result["text"])
                finally:
                    if temp_file.name in temp_files:
                        temp_files.remove(temp_file.name)
                        self.safe_delete(temp_file.name)
                    
        except Exception as e:
            logging.error(f"Error processing segment: {str(e)}")
        finally:
            for temp_file in temp_files:
                self.safe_delete(temp_file)

    def _handle_conversation(self, user_input):
        """Handle conversation with advanced memory and follow-up processing"""
        try:
            # Add user input to memory
            self.memory.add_message(self.session_id, "user", user_input)
            
            # Update conversation flow
            self.conversation_flow.append({"role": "user", "content": user_input})
            
            # Analyze input for intent and context
            intent = self._analyze_intent(user_input)
            context = self._get_conversation_context(user_input)
            
            # Generate response with context awareness
            response = self._generate_contextual_response(user_input, intent, context)
            
            # Add response to memory
            self.memory.add_message(self.session_id, "assistant", response)
            self.conversation_flow.append({"role": "assistant", "content": response})
            
            # Update conversation state
            self._update_conversation_state(user_input, response)
            
            # Display response
            self._display_response(user_input, response)
            
        except Exception as e:
            logging.error(f"Error handling conversation: {str(e)}")
            error_response = "I apologize, but I encountered an error processing your message. Could you please try again?"
            self.memory.add_message(self.session_id, "assistant", error_response)
            print(f"\n❌ Error: {error_response}")

    def _analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for intent and sentiment"""
        input_lower = user_input.lower()
        
        intent = {
            "type": "general",
            "sentiment": "neutral",
            "requires_follow_up": False,
            "topic": None,
            "urgency": "normal"
        }
        
        # Analyze intent type
        if any(word in input_lower for word in ["help", "assist", "support"]):
            intent["type"] = "help_request"
        elif any(word in input_lower for word in ["question", "ask", "what", "how", "why", "when", "where"]):
            intent["type"] = "question"
        elif any(word in input_lower for word in ["thank", "thanks", "appreciate"]):
            intent["type"] = "gratitude"
            intent["sentiment"] = "positive"
        elif any(word in input_lower for word in ["goodbye", "bye", "exit", "quit", "stop"]):
            intent["type"] = "farewell"
        elif any(word in input_lower for word in ["code", "program", "algorithm", "function"]):
            intent["type"] = "coding"
            intent["topic"] = "programming"
        elif any(word in input_lower for word in ["ai", "machine learning", "ml", "neural", "model"]):
            intent["type"] = "ai_ml"
            intent["topic"] = "artificial_intelligence"
        
        # Analyze sentiment
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "wrong", "error"]
        
        if any(word in input_lower for word in positive_words):
            intent["sentiment"] = "positive"
        elif any(word in input_lower for word in negative_words):
            intent["sentiment"] = "negative"
        
        # Check if follow-up is needed
        if intent["type"] in ["question", "help_request", "coding", "ai_ml"]:
            intent["requires_follow_up"] = True
        
        return intent

    def _get_conversation_context(self, user_input: str) -> Dict[str, Any]:
        """Get relevant conversation context"""
        # Get recent context
        recent_context = self.memory.get_recent_context(self.session_id, limit=10)
        
        # Get semantic context
        semantic_context = self.memory.get_semantic_context(user_input, self.session_id, limit=5)
        
        # Combine contexts
        context = {
            "recent_messages": recent_context,
            "semantic_matches": semantic_context,
            "current_topic": self.current_topic,
            "user_preferences": self.user_preferences,
            "conversation_flow": list(self.conversation_flow)
        }
        
        return context

    def _generate_contextual_response(self, user_input: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate contextual response with memory awareness"""
        try:
            # Build context string
            context_str = self._build_context_string(context)
            
            if self.use_rag:
                # Use RAG with conversation context
                full_query = f"Context: {context_str}\n\nUser: {user_input}\n\nIntent: {intent}\n\nResponse:"
                return self.llm.query_topic(
                    "conversational_assistant",
                    full_query,
                    filters={
                        "type": ["conversation", "general", "coding", "ai_ml"],
                        "difficulty": ["easy", "medium", "hard"]
                    }
                )
            
            # Direct LLM response with context
            prompt = f"{self.system_prompt}\n\n{context_str}\n\nUser: {user_input}\n\nIntent Analysis: {intent}\n\nAssistant:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=2000,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
            
            # Clean up response
            response = re.sub(r'^[^a-zA-Z]*', '', response)
            return response
            
        except Exception as e:
            logging.error(f"Error generating contextual response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Could you please try rephrasing your question?"

    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string for LLM"""
        context_parts = []
        
        # Add recent conversation
        if context["recent_messages"]:
            context_parts.append("Recent conversation:")
            for msg in context["recent_messages"][-5:]:  # Last 5 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content']}")
        
        # Add current topic
        if context["current_topic"]:
            context_parts.append(f"Current topic: {context['current_topic']}")
        
        # Add user preferences
        if context["user_preferences"]:
            context_parts.append(f"User preferences: {context['user_preferences']}")
        
        return "\n".join(context_parts)

    def _update_conversation_state(self, user_input: str, response: str):
        """Update conversation state based on interaction"""
        # Update current topic based on content
        if "programming" in user_input.lower() or "code" in user_input.lower():
            self.current_topic = "programming"
        elif "ai" in user_input.lower() or "machine learning" in user_input.lower():
            self.current_topic = "artificial_intelligence"
        elif "help" in user_input.lower():
            self.current_topic = "help"
        
        # Extract user preferences
        if "prefer" in user_input.lower() or "like" in user_input.lower():
            # Simple preference extraction
            if "detailed" in user_input.lower():
                self.user_preferences["detail_level"] = "detailed"
            elif "simple" in user_input.lower():
                self.user_preferences["detail_level"] = "simple"

    def _display_response(self, user_input: str, response: str):
        """Display the conversation exchange"""
        print("\n" + "="*60)
        print(f"👤 You: {user_input}")
        print(f"🤖 Assistant: {response}")
        print("="*60)

    def _audio_callback(self, indata, frames, time, status):
        """Audio callback with optimized queue handling"""
        if status:
            logging.warning(f"Audio callback status: {status}")
            if status.input_overflow:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                return

        try:
            audio_data = indata.tobytes()
            
            if indata.shape[1] > 1:
                audio_array = np.mean(indata, axis=1, dtype=np.int16)
                audio_data = audio_array.tobytes()
            
            if len(audio_data) > 0:
                self.audio_queue.put(audio_data, timeout=0.1)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put(audio_data, timeout=0.1)
            except:
                pass
        except Exception as e:
            logging.error(f"Error in audio callback: {str(e)}")

    @lru_cache(maxsize=1)
    def _get_system_prompt(self):
        """Get the system prompt for the LLM"""
        return """You are an advanced conversational AI assistant with the following capabilities:

1. **Natural Conversation**: Engage in natural, flowing dialogue
2. **Memory Awareness**: Remember and reference previous conversation context
3. **Contextual Responses**: Provide responses that build on previous exchanges
4. **Follow-up Questions**: Ask relevant follow-up questions when appropriate
5. **Multi-domain Knowledge**: Programming, AI/ML, general knowledge, problem-solving
6. **Personalization**: Adapt to user preferences and communication style

Conversation Guidelines:
- Be warm, engaging, and conversational
- Reference previous context when relevant
- Ask clarifying questions when needed
- Provide detailed explanations when appropriate
- Show enthusiasm and personality
- Maintain consistency in your responses
- Adapt your communication style to the user's preferences

When responding:
- Acknowledge the user's input naturally
- Build on previous conversation context
- Provide helpful, accurate information
- Ask follow-up questions to deepen the conversation
- Show that you remember previous interactions
- Be conversational, not robotic

Remember: You're having a natural conversation, not just answering questions."""

    def run_conversation_session(self):
        """Run the conversational assistant session"""
        print("\n" + "="*60)
        print("💬 Advanced Conversational Assistant")
        print("="*60)
        print("I'm your AI conversation partner with memory and context awareness!")
        print("Features:")
        print("• Natural conversation flow")
        print("• Memory of our entire conversation")
        print("• Context-aware responses")
        print("• Follow-up questions and engagement")
        print("• Multi-domain knowledge")
        print("• Personalized interaction")
        print(f"\nSession ID: {self.session_id}")
        print("\nPress Ctrl+C to end our conversation")
        print("="*60)
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nSaving conversation and ending session...")
            
            # Generate session summary
            recent_messages = self.memory.get_recent_context(self.session_id, limit=20)
            if recent_messages:
                summary = f"Conversation covered topics including {self.current_topic or 'various subjects'}. Total exchanges: {len(recent_messages)}"
            else:
                summary = "Brief conversation session"
            
            # End session
            self.memory.end_session(self.session_id, summary)
            
            # Cleanup
            self.stream.stop()
            self.stream.close()
            self.executor.shutdown(wait=True)
            gc.collect()
            
            print("Thank you for the conversation! I've saved our chat for future reference. 👋")

if __name__ == "__main__":
    # List available audio devices
    print("\n🎤 Available audio input devices:")
    devices = ConversationalAssistant.list_audio_devices()
    for device in devices:
        print(f"{device['index']}: {device['name']}")
        print(f"   Channels: {device['channels']}")
        print(f"   Sample Rate: {device['default_samplerate']}")
        print(f"   Type: {'Bluetooth HFP' if device['is_hfp'] else 'Bluetooth' if device['is_bluetooth'] else 'Other'}")
        print()
    
    # Device selection
    device_idx = 1
    selection = input("\nEnter the number of your input device (or press Enter for default 1): ").strip()
    if selection:
        try:
            device_idx = int(selection)
            if device_idx not in [device['index'] for device in devices]:
                print("Invalid device number. Using default device 1.")
                device_idx = 1
        except ValueError:
            print("Invalid input. Using default device 1.")
    
    # Test the selected device
    print("\n🔧 Testing selected audio device...")
    if not ConversationalAssistant.test_audio_device(device_idx):
        print("❌ Device test failed. Please try a different device.")
        exit(1)
    
    # Initialize the conversational assistant
    print("\n🚀 Initializing Advanced Conversational Assistant...")
    assistant = ConversationalAssistant(use_rag=True, input_device=device_idx)
    
    # Set up audio stream
    if not assistant.setup_audio_stream():
        print("❌ Failed to set up audio stream. Please check your microphone.")
        exit(1)
    
    # Run the conversation session
    assistant.run_conversation_session() 