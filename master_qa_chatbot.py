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
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_qa_chatbot.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MasterQAChatbot:
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
        conversation_history_file="conversation_history.json"
    ):
        """
        Initialize the Master QA Chatbot with comprehensive capabilities
        """
        self.device = device
        self.use_rag = use_rag
        self.input_device = input_device
        self.conversation_history_file = conversation_history_file
        
        # Initialize conversation memory
        self.conversation_history = []
        self.load_conversation_history()
        
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

    def load_conversation_history(self):
        """Load conversation history from file"""
        try:
            if os.path.exists(self.conversation_history_file):
                with open(self.conversation_history_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logging.info(f"Loaded {len(self.conversation_history)} conversation entries")
        except Exception as e:
            logging.error(f"Error loading conversation history: {str(e)}")
            self.conversation_history = []

    def save_conversation_history(self):
        """Save conversation history to file"""
        try:
            with open(self.conversation_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving conversation history: {str(e)}")

    def add_to_conversation(self, role, content, timestamp=None):
        """Add a message to conversation history"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        entry = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.conversation_history.append(entry)
        
        # Keep only last 50 messages to prevent memory issues
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        # Save periodically
        if len(self.conversation_history) % 10 == 0:
            self.save_conversation_history()

    def get_conversation_context(self, max_messages=10):
        """Get recent conversation context for LLM"""
        recent_messages = self.conversation_history[-max_messages:] if self.conversation_history else []
        context = ""
        
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        return context

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
                
                self.llm.create_topic_collection("master_qa")
                self._initialize_qa_data()
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

    def _initialize_qa_data(self):
        """Initialize QA data for RAG"""
        if not self.use_rag:
            return
            
        data_dir = Path("qa_data")
        if data_dir.exists() and any(data_dir.iterdir()):
            for data_file in data_dir.glob("*.json"):
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        self.llm.add_documents("master_qa", data)
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
                        self.executor.submit(self._handle_response, result["text"])
                finally:
                    if temp_file.name in temp_files:
                        temp_files.remove(temp_file.name)
                        self.safe_delete(temp_file.name)
                    
        except Exception as e:
            logging.error(f"Error processing segment: {str(e)}")
        finally:
            for temp_file in temp_files:
                self.safe_delete(temp_file)

    def _handle_response(self, question):
        """Handle the response generation with conversation memory"""
        try:
            # Add user question to conversation history
            self.add_to_conversation("user", question)
            
            # Get response with conversation context
            response = self.get_llm_response(question)
            
            if response:
                # Add assistant response to conversation history
                self.add_to_conversation("assistant", response)
                
                print("\n" + "="*50)
                print(f"User: {question}")
                print(f"Assistant: {response}")
                print("="*50)
                
                # Save conversation after each exchange
                self.save_conversation_history()
        except Exception as e:
            logging.error(f"Error handling response: {str(e)}")

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

    def get_llm_response(self, question, context=None):
        """Get LLM response with conversation context"""
        try:
            # Get conversation context
            conversation_context = self.get_conversation_context()
            
            if self.use_rag:
                # Use RAG with conversation context
                full_query = f"Conversation context:\n{conversation_context}\n\nCurrent question: {question}"
                return self.llm.query_topic(
                    "master_qa",
                    full_query,
                    filters={
                        "type": ["coding", "conceptual", "system_design", "general"],
                        "difficulty": ["easy", "medium", "hard"]
                    }
                )
            
            # Direct LLM response with conversation context
            prompt = f"{self.system_prompt}\n\nConversation History:\n{conversation_context}\n\nCurrent Question: {question}\n\nResponse:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=1500,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Response:")[-1].strip()
            
            # Clean up response
            response = re.sub(r'^[^a-zA-Z]*', '', response)
            return response
        except Exception as e:
            logging.error(f"Error getting LLM response: {str(e)}")
            return "I apologize, but I encountered an error processing your question. Could you please try again?"

    @lru_cache(maxsize=1)
    def _get_system_prompt(self):
        """Get the system prompt for the LLM"""
        return """You are Master QA, an expert AI assistant with comprehensive knowledge across multiple domains. Your capabilities include:

1. **Programming & Software Development**: Python, JavaScript, Java, C++, algorithms, data structures, system design
2. **Machine Learning & AI**: ML algorithms, deep learning, NLP, computer vision, data science
3. **General Knowledge**: Science, technology, history, current events, problem-solving
4. **Conversational Skills**: Natural dialogue, follow-up questions, context awareness

When responding:
- Be conversational and engaging
- Reference previous conversation context when relevant
- Provide detailed, accurate information
- For coding questions: include clear, well-commented code
- For conceptual questions: explain clearly with examples
- Ask follow-up questions when appropriate
- Maintain a helpful, professional tone

Format responses naturally in conversation style, adapting to the user's needs and the context of the discussion."""

    def run_chatbot_session(self):
        """Run the chatbot session"""
        print("\n" + "="*60)
        print("🤖 Master QA Chatbot - Your AI Assistant")
        print("="*60)
        print("I'm listening for your questions and ready to help!")
        print("Features:")
        print("• Voice recognition and natural conversation")
        print("• Memory of our conversation")
        print("• Comprehensive knowledge across domains")
        print("• Context-aware responses")
        print("\nPress Ctrl+C to quit")
        print("="*60)
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nSaving conversation and shutting down...")
            self.save_conversation_history()
            self.stream.stop()
            self.stream.close()
            self.executor.shutdown(wait=True)
            gc.collect()
            print("Goodbye! 👋")

if __name__ == "__main__":
    # List available audio devices
    print("\n🎤 Available audio input devices:")
    devices = MasterQAChatbot.list_audio_devices()
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
    if not MasterQAChatbot.test_audio_device(device_idx):
        print("❌ Device test failed. Please try a different device.")
        exit(1)
    
    # Initialize the chatbot
    print("\n🚀 Initializing Master QA Chatbot...")
    chatbot = MasterQAChatbot(use_rag=True, input_device=device_idx)
    
    # Set up audio stream
    if not chatbot.setup_audio_stream():
        print("❌ Failed to set up audio stream. Please check your microphone.")
        exit(1)
    
    # Run the chatbot session
    chatbot.run_chatbot_session() 