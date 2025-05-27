# Fitness App Setup Guide

## Quick Start

1. **Create and activate virtual environment**:
```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# OR Activate (Mac/Linux)
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Project Structure
```
fitness_app/
├── data/                      # Data storage directory
│   ├── exercises/            # Exercise-related data
│   ├── nutrition/            # Nutrition-related data
│   ├── injuries/             # Injury-related data
│   ├── recovery/             # Recovery-related data
│   └── fitness_knowledge/    # General fitness knowledge
├── requirements.txt          # Project dependencies
├── data_collector.py         # Data collection script
└── SETUP.md                  # This setup guide
```

## Data Collection

### 1. Create Data Directories
```bash
# From the fitness_app directory
mkdir -p data/{exercises,nutrition,injuries,recovery,fitness_knowledge}
```

### 2. Start Data Collection
```bash
# Run the collector
python data_collector.py
```

The collector will:
- Create necessary directories
- Download data from verified sources
- Apply quality filters
- Save data in JSON format
- Generate quality reports

### 3. Monitor Progress
- Check the console output for progress
- Review quality metrics in the data files
- Check the `data` directory for collected files

### 4. Verify Data
Each category directory will contain:
- `{category}_data.json`: Main data file
- Individual source files
- Quality metrics and statistics

## Data Sources

### Nutrition Data
- USDA Food Database (Government)
- Nutrition.gov (Government)
- Harvard Nutrition Source (Academic)

### Exercise Data
- ACSM Exercise Guidelines (Professional)
- ExRx.net Exercise Database (Professional)
- NSCA Exercise Technique Manual (Professional)

### Injury Data
- Physiopedia (Medical)
- PubMed Central (Academic)
- AAOS Sports Medicine (Medical)

### Recovery Data
- Journal of Sports Rehabilitation (Academic)
- APTA Sports Physical Therapy (Professional)

### General Fitness
- WHO Physical Activity Guidelines (International)
- CDC Physical Activity Guidelines (Government)
- European Journal of Sport Science (Academic)

## Troubleshooting

### Common Issues

1. **NLTK Data Not Found**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

2. **Permission Errors**
- Run as administrator/with sudo if needed
- Check directory permissions

3. **Memory Issues**
- The collector processes data in batches
- Monitor system resources

### Getting Help

If you encounter issues:
1. Check error messages in console
2. Verify directory permissions
3. Ensure all dependencies are installed
4. Check disk space availability

## Next Steps

After data collection:
1. Review the quality metrics in each JSON file
2. Check the data distribution across categories
3. Verify source attribution
4. Begin model training

## Maintenance

Regular tasks:
1. Check for new data sources
2. Update dependencies
3. Monitor disk space
4. Backup collected data 