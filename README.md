# LLMs Repository

This repository contains various tools and applications for working with Large Language Models (LLMs), focusing on fitness, transcription, and general LLM interactions.
# Check out the chatbot readme for new chatbot functionalities!

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
 
## Data Sources

### Exercise Data Sources
- [WGER Exercise API](https://wger.de/api/v2/exercise/) - Open source exercise database
- [FitDB API](https://api.fitdb.com/v1/exercises) - Free exercise database
- [Kaggle Fitness Dataset](https://www.kaggle.com/datasets/fmendes/fitness-exercises-with-animations) - Fitness exercises with animations
- [UCI Fitness Dataset](https://archive.ics.uci.edu/ml/datasets/Fitness+Exercises) - Academic fitness dataset

### Nutrition Data Sources
- [USDA Food Database API](https://api.nal.usda.gov/fdc/v1/foods/search) - Official USDA food database
- [Open Food Facts API](https://world.openfoodfacts.org/api/v2/search) - Open source food database

### Injury Data Sources
- [OpenFDA API](https://api.fda.gov/drug/event.json) - FDA adverse event reports
- [WHO Injury Database](https://www.who.int/data/gho/data/themes/topics/injuries) - WHO injury statistics

### Fitness Knowledge Sources
- [PubMed API](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/) - Medical research database
- [CORE API](https://core.ac.uk/api/v3/search/works) - Academic research database
- [FitnessGram Dataset](https://www.fitnessgram.net/datasets/) - Physical fitness dataset

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
USDA_API_KEY=your_key_here
OPENFDA_API_KEY=your_key_here
PUBMED_API_KEY=your_key_here
CORE_API_KEY=your_key_here
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

## Data Collection Statistics

### Expected Data Volume
- Exercises: ~113,500 records
- Nutrition: ~1,300,000 items
- Injuries: ~70,000 records
- Fitness Knowledge: ~30,000 records
- Total Size: ~500-600 MB

### Data Quality
- All sources are verified and reliable
- Data is validated and deduplicated
- Quality scores are calculated for each item
- Cross-references are maintained

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
- All data sources are free to use and don't require paid subscriptions 
