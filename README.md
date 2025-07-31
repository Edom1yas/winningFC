# WinningFC - Soccer Match Prediction System

A comprehensive data engineering project for predicting soccer match outcomes using machine learning.

## Features

- **Data Collection**: Modular scrapers for match data, team statistics, and league tables
- **Feature Engineering**: Advanced feature creation including team form, head-to-head records, and temporal features  
- **Machine Learning**: Multiple model support (Random Forest, Gradient Boosting, Logistic Regression)
- **REST API**: FastAPI-based API for training models and making predictions
- **Performance Metrics**: Cross-validation, feature importance, and model evaluation

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

## Future Enhancements

- Real-time data streaming
- Player-level statistics
- Weather and venue features
- Advanced ensemble methods
- Model deployment pipeline
- Web dashboard interface

## License

MIT License