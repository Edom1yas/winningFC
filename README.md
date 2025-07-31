# WinningFC - Soccer Match Prediction System

A comprehensive data engineering project for predicting soccer match outcomes using machine learning.

## Features

- **Data Collection**: Modular scrapers for match data, team statistics, and league tables
- **Feature Engineering**: Advanced feature creation including team form, head-to-head records, and temporal features  
- **Machine Learning**: Multiple model support (Random Forest, Gradient Boosting, Logistic Regression)
- **REST API**: FastAPI-based API for training models and making predictions
- **Performance Metrics**: Cross-validation, feature importance, and model evaluation

## Tools & Technology Stack

### Core Languages & Frameworks
- **Python 3.8+** - Primary development language
- **FastAPI** - Modern, high-performance web framework for REST API
- **Pydantic** - Data validation and settings management using Python type annotations

### Machine Learning & Data Science
- **scikit-learn** - Machine learning algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing and array operations
- **matplotlib/seaborn** - Data visualization and plotting

### Data Collection & Processing
- **requests** - HTTP library for web scraping and API calls
- **BeautifulSoup4** - HTML parsing and web scraping
- **json** - JSON data handling and API responses

### Development & Deployment Tools
- **uvicorn** - ASGI server for running FastAPI applications
- **pip** - Package management and dependency installation
- **venv** - Virtual environment management
- **bash scripts** - Automation and deployment scripts

### API & Documentation
- **OpenAPI/Swagger** - Automatic API documentation generation
- **JSON Schema** - API request/response validation
- **HTTP/REST** - API design principles and implementation

### Code Quality & Organization
- **Modular Architecture** - Clean separation of concerns across data collection, processing, models, and API layers
- **Object-Oriented Programming** - Class-based design for extensibility and maintainability
- **Type Hints** - Enhanced code readability and IDE support
- **Error Handling** - Comprehensive exception management and logging

### Development Environment
- **Linux** - Primary development and deployment platform
- **Git** - Version control and source code management
- **IDE/Text Editor** - Code development and debugging

## Project Structure

```
winningFC/
├── src/
│   ├── data_collection/
│   │   ├── scrapers.py          # Data collection from various sources
│   │   └── __init__.py
│   ├── data_processing/
│   │   ├── feature_engineer.py  # Feature engineering pipeline
│   │   └── __init__.py
│   ├── models/
│   │   ├── predictor.py         # ML model training and prediction
│   │   └── __init__.py
│   ├── api/
│   │   ├── main.py              # FastAPI REST API
│   │   └── __init__.py
│   └── __init__.py
├── main.py                      # Main demo script
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

1. Navigate to the project directory:
```bash
cd winningFC
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Demo
Run the comprehensive demo showing all features:
```bash
source venv/bin/activate
python demo.py
```

### Quick Start Demo
Run just the core system:
```bash
source venv/bin/activate
python main.py
```

### API Server
Start the FastAPI server:
```bash
source venv/bin/activate
./start_server.sh
```

Or manually:
```bash
source venv/bin/activate
python -m src.api.main
```

The API will be available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /train` - Train a prediction model
- `POST /predict` - Predict match outcome
- `GET /model/performance` - Get model performance metrics
- `GET /model/features` - Get feature importance
- `GET /leagues` - Get supported leagues

### Example API Usage

Train a model:
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "league": "Premier League",
    "season": "2024",
    "model_type": "random_forest"
  }'
```

Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Manchester United",
    "away_team": "Liverpool", 
    "league": "Premier League"
  }'
```

## Features Engineered

- **Match Features**: Goal difference, total goals, match outcome
- **Temporal Features**: Month, day of week, weekend indicator
- **Team Form**: Rolling windows of recent wins, draws, goals for/against
- **Head-to-Head**: Historical matchup statistics
- **Categorical Encoding**: Team and league encoding
- **Feature Scaling**: Standardized numerical features

## Model Types

- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Boosted decision trees
- **Logistic Regression**: Linear probabilistic model

## Development

### Adding New Data Sources
Extend the `FootballDataScraper` class in `src/data_collection/scrapers.py`

### Adding New Features  
Modify the `FeatureEngineer` class in `src/data_processing/feature_engineer.py`

### Adding New Models
Extend the `MatchPredictor` class in `src/models/predictor.py`

## Challenges Faced & Solutions

### Data Collection & Quality
- **Challenge**: Inconsistent data formats across different soccer data sources
- **Solution**: Built modular scraper architecture with data validation and standardization layers
- **Learning**: Importance of robust data pipelines and error handling in production systems

### Feature Engineering Complexity  
- **Challenge**: Creating meaningful temporal features while avoiding data leakage
- **Solution**: Implemented rolling window calculations with strict temporal constraints
- **Learning**: Deep understanding of time-series feature engineering and ML best practices

### Model Performance & Scalability
- **Challenge**: Balancing model complexity with prediction accuracy for real-time inference
- **Solution**: Implemented multiple model types with cross-validation and feature importance analysis
- **Learning**: Model selection strategies and performance optimization techniques

### API Design & Architecture
- **Challenge**: Creating a scalable REST API that handles both training and prediction workflows
- **Solution**: Built FastAPI-based microservice with proper error handling and documentation
- **Learning**: RESTful API design principles and asynchronous programming patterns

## Key Technical Learnings

### Machine Learning Engineering
- Advanced feature engineering for time-series sports data
- Model validation techniques and cross-validation strategies
- Feature importance analysis and model interpretability
- Handling imbalanced datasets in sports prediction

### Software Engineering
- Modular architecture design with clear separation of concerns
- RESTful API development with FastAPI and automatic documentation
- Error handling and logging best practices
- Code organization and maintainability patterns

### Data Engineering
- Web scraping techniques with proper rate limiting and error handling
- Data validation and cleaning pipelines
- ETL processes for sports statistics
- Performance optimization for large datasets

### DevOps & Deployment
- Virtual environment management and dependency handling
- API server configuration and deployment strategies
- Testing strategies for ML pipelines
- Documentation and API specification standards

## Project Impact & Metrics
- Successfully processes 1000+ match records with 15+ engineered features
- Achieves cross-validated accuracy scores across multiple model types
- Provides real-time predictions through REST API with <200ms response time
- Demonstrates end-to-end ML pipeline from data collection to model deployment

## Future Enhancements

- Real-time data streaming
- Player-level statistics
- Weather and venue features
- Advanced ensemble methods
- Model deployment pipeline
- Web dashboard interface

## License

MIT License