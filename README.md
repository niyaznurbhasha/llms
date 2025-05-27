# LLMs Repository

This repository contains various tools and applications for working with Large Language Models (LLMs), focusing on fitness, transcription, and general LLM interactions.

## Projects

### 1. Fitness App
A comprehensive fitness application that includes:
- Data collection from various fitness sources
- Exercise database
- Nutrition information
- Injury prevention and recovery data
- Fitness knowledge base

### 2. Transcription and Summarization
- Lightweight transcription and summarization tool
- Supports various audio formats
- Generates concise summaries

### 3. General LLM Tools
- General-purpose LLM interaction module
- Data collection utilities
- Question-answering system

## Directory Structure

```
llms/
├── fitness_app/              # Fitness application
│   ├── data/                # Data storage
│   └── data_collector.py    # Data collection module
├── transcribeAndSummarizeLightWeight.py  # Transcription tool
├── master_qa.py             # Question-answering system
├── general_llm.py           # General LLM utilities
└── general_data_collector.py # Data collection utilities
```

## Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
- Create a `.env` file in the root directory
- Add your API keys:
```
EXERCISEDB_API_KEY=your_key_here
USDA_API_KEY=your_key_here
NUTRITIONIX_API_KEY=your_key_here
OPENFDA_API_KEY=your_key_here
PUBMED_API_KEY=your_key_here
```

## Usage

### Fitness App
```python
from fitness_app.data_collector import FitnessDataCollector

collector = FitnessDataCollector()
collector.collect_category_data("exercises")
```

### Transcription Tool
```python
python transcribeAndSummarizeLightWeight.py --input audio_file.mp3
```

### General LLM Tools
```python
from general_llm import LLMHandler

handler = LLMHandler()
response = handler.process_query("Your question here")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

- The `interview_assistant` directory is excluded from this repository
- API keys should never be committed to the repository
- Always use virtual environments for development 