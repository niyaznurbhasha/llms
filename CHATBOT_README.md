# 🤖 Advanced Chatbot Applications

This repository contains two advanced chatbot applications with voice interaction, memory, and conversational AI capabilities.

## 📁 Files Overview

### 1. `master_qa_chatbot.py` - Master QA Chatbot
A comprehensive question-answering chatbot with conversation memory and voice interaction.

**Features:**
- 🎤 Voice recognition and natural speech processing
- 💾 Conversation history persistence
- 🧠 Context-aware responses
- 🔍 Multi-domain knowledge (programming, AI/ML, general)
- 📝 Automatic conversation logging

### 2. `conversational_assistant.py` - Advanced Conversational Assistant
A sophisticated conversational AI with advanced memory, intent analysis, and follow-up handling.

**Features:**
- 🗄️ SQLite-based conversation memory
- 🎯 Intent analysis and sentiment detection
- 🔄 Natural conversation flow with follow-ups
- 🎨 Personalized interaction based on user preferences
- 📊 Semantic context retrieval
- 🏷️ Session management and topic tracking

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Audio input device** (microphone) connected
3. **Sufficient disk space** for model downloads (several GB)

### Installation

1. **Clone or download** the chatbot files to your local machine

2. **Install dependencies:**
   ```bash
   pip install -r chatbot_requirements.txt
   ```

3. **Install system audio dependencies:**

   **Windows:**
   ```bash
   # Usually no additional installation needed
   ```

   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install portaudio19-dev python3-pyaudio
   ```

   **macOS:**
   ```bash
   brew install portaudio
   ```

### Running the Chatbots

#### Option 1: Master QA Chatbot
```bash
python master_qa_chatbot.py
```

#### Option 2: Conversational Assistant
```bash
python conversational_assistant.py
```

## 🎤 Audio Device Setup

Both chatbots will automatically detect and list available audio input devices. You'll be prompted to select your preferred microphone.

**Device Selection Tips:**
- Choose a device with good audio quality
- Avoid Bluetooth HFP devices for better performance
- Ensure the device is not being used by other applications

## 🔧 Configuration

### Model Configuration

Both chatbots use the `deepseek-ai/deepseek-coder-6.7b-base` model by default. You can modify this in the constructor:

```python
# For different models
chatbot = MasterQAChatbot(
    model_name="your-preferred-model",
    use_rag=True,  # Enable/disable RAG
    device="cuda"  # Use "cpu" if no GPU available
)
```

### Memory Configuration

**Master QA Chatbot:**
- Conversation history stored in JSON format
- Configurable history length (default: 50 messages)
- Automatic periodic saving

**Conversational Assistant:**
- SQLite database for persistent storage
- Session-based memory management
- Semantic search capabilities

## 📊 Features Comparison

| Feature | Master QA Chatbot | Conversational Assistant |
|---------|------------------|-------------------------|
| Voice Recognition | ✅ | ✅ |
| Conversation Memory | ✅ (JSON) | ✅ (SQLite) |
| Intent Analysis | ❌ | ✅ |
| Follow-up Questions | Basic | Advanced |
| Session Management | ❌ | ✅ |
| Semantic Search | ❌ | ✅ |
| User Preferences | ❌ | ✅ |
| Topic Tracking | ❌ | ✅ |

## 🎯 Use Cases

### Master QA Chatbot
- **Technical interviews** and coding questions
- **Educational assistance** for programming concepts
- **Quick Q&A sessions** with memory
- **Voice-based learning** environments

### Conversational Assistant
- **Long-term conversations** with context retention
- **Personal AI assistant** with learning capabilities
- **Research discussions** with topic continuity
- **Interactive learning** with personalized responses

## 🔍 Troubleshooting

### Common Issues

1. **Audio Device Not Found**
   - Check microphone connections
   - Ensure device is set as default in system settings
   - Try different USB ports for USB microphones

2. **Model Download Issues**
   - Check internet connection
   - Ensure sufficient disk space (5-10 GB)
   - Try using a different model

3. **Performance Issues**
   - Use GPU if available (`device="cuda"`)
   - Reduce model size for faster responses
   - Close other applications using the microphone

4. **Memory Issues**
   - Reduce conversation history length
   - Clear old conversation files
   - Monitor system memory usage

### Performance Optimization

1. **GPU Usage:**
   ```python
   # Enable GPU acceleration
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Model Offloading:**
   ```python
   # Use model offloading for large models
   offload_folder="model_offload"
   ```

3. **Audio Quality:**
   - Use high-quality microphone
   - Minimize background noise
   - Ensure proper audio levels

## 📝 Logging

Both chatbots create detailed logs:
- `master_qa_chatbot.log` - Master QA Chatbot logs
- `conversational_assistant.log` - Conversational Assistant logs
- `conversation_history.json` - Master QA conversation history
- `conversation_memory.db` - Conversational Assistant database

## 🔒 Privacy and Data

- **Local Processing:** All audio and conversation data processed locally
- **No Cloud Storage:** Conversations stored only on your device
- **Optional Logging:** Can disable logging for privacy
- **Data Control:** Full control over conversation data

## 🛠️ Development

### Extending the Chatbots

1. **Add New Models:**
   ```python
   # Modify the model_name parameter
   model_name="your-custom-model"
   ```

2. **Custom System Prompts:**
   ```python
   # Override the _get_system_prompt method
   def _get_system_prompt(self):
       return "Your custom system prompt"
   ```

3. **Add New Features:**
   - Extend the conversation memory classes
   - Add new intent types
   - Implement custom response generation

### Testing

```bash
# Run basic tests
python -m pytest tests/

# Test audio devices
python -c "from master_qa_chatbot import MasterQAChatbot; MasterQAChatbot.list_audio_devices()"
```

## 📚 API Reference

### MasterQAChatbot Class

```python
class MasterQAChatbot:
    def __init__(self, model_name, use_rag=True, input_device=1, ...)
    def setup_audio_stream(self) -> bool
    def run_chatbot_session(self)
    def get_llm_response(self, question: str) -> str
    def add_to_conversation(self, role: str, content: str)
    @staticmethod
    def list_audio_devices() -> List[Dict]
```

### ConversationalAssistant Class

```python
class ConversationalAssistant:
    def __init__(self, model_name, session_id=None, ...)
    def setup_audio_stream(self) -> bool
    def run_conversation_session(self)
    def _analyze_intent(self, user_input: str) -> Dict[str, Any]
    def _get_conversation_context(self, user_input: str) -> Dict[str, Any]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check individual file headers for specific licensing information.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Ensure all dependencies are properly installed
4. Test with different audio devices

---

**Happy Chatting! 🎉** 