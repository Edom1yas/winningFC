from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime, date
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.predictor import MatchPredictor
from src.data_processing.feature_engineer import FeatureEngineer
from src.data_collection.scrapers import FootballDataScraper

app = FastAPI(title="WinningFC - Soccer Match Predictor", version="1.0.0")

# Global instances
predictor = MatchPredictor()
feature_engineer = FeatureEngineer()
scraper = FootballDataScraper()

class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    match_date: Optional[str] = None

class MatchPredictionResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_result: str
    probabilities: Dict[str, float]
    confidence: float
    timestamp: str

class TrainingRequest(BaseModel):
    league: str
    season: str
    model_type: Optional[str] = "random_forest"

class ModelPerformance(BaseModel):
    accuracy: float
    cv_mean: float
    cv_std: float
    test_size: int
    train_size: int

@app.get("/")
async def root():
    return {"message": "WinningFC Soccer Match Predictor API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/train", response_model=ModelPerformance)
async def train_model(request: TrainingRequest):
    global predictor
    try:
        # Get training data
        matches_df = scraper.get_match_data(request.league, request.season)
        
        if matches_df.empty:
            raise HTTPException(status_code=404, detail="No match data found")
        
        # Prepare features using temporary predictor
        temp_predictor = MatchPredictor(request.model_type)
        matches_df = temp_predictor.prepare_target_variable(matches_df)
        features_df, feature_cols = feature_engineer.prepare_features_for_prediction(matches_df)
        
        # Extract features and target
        X = features_df[feature_cols]
        y = features_df['match_result']
        
        # Initialize and train model
        predictor = MatchPredictor(request.model_type)
        performance = predictor.train_model(X, y)
        
        return ModelPerformance(
            accuracy=performance['accuracy'],
            cv_mean=performance['cv_mean'],
            cv_std=performance['cv_std'],
            test_size=performance['test_size'],
            train_size=performance['train_size']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=MatchPredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    try:
        if predictor.model is None:
            raise HTTPException(status_code=400, detail="Model not trained. Please train a model first.")
        
        # Create match data for prediction
        match_data = {
            'home_team': request.home_team,
            'away_team': request.away_team,
            'league': request.league,
            'date': request.match_date or datetime.now().strftime('%Y-%m-%d'),
            'home_score': 0,  # Placeholder
            'away_score': 0,  # Placeholder
            'season': '2024'  # Default season
        }
        
        match_df = pd.DataFrame([match_data])
        
        # Prepare features
        features_df, feature_cols = feature_engineer.prepare_features_for_prediction(match_df)
        
        # Make prediction
        match_features = features_df[feature_cols].iloc[0:1]
        prediction = predictor.predict_match(match_features)
        
        return MatchPredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            predicted_result=prediction['predicted_result'],
            probabilities=prediction['probabilities'],
            confidence=prediction['confidence'],
            timestamp=prediction['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/performance")
async def get_model_performance():
    if predictor.model is None:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    return predictor.get_model_performance()

@app.get("/model/features")
async def get_feature_importance():
    if predictor.model is None:
        raise HTTPException(status_code=400, detail="No trained model available")
    
    importance = predictor.get_feature_importance()
    if importance is not None:
        return importance.to_dict('records')
    else:
        return {"message": "Feature importance not available for this model type"}

@app.get("/leagues")
async def get_supported_leagues():
    return {
        "leagues": [
            "Premier League",
            "La Liga", 
            "Serie A",
            "Bundesliga",
            "Ligue 1"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)