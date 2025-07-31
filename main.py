#!/usr/bin/env python3

import pandas as pd
import sys
import os
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection.scrapers import FootballDataScraper
from src.data_processing.feature_engineer import FeatureEngineer
from src.models.predictor import MatchPredictor

def main():
    print("🏆 WinningFC - Soccer Match Prediction System")
    print("=" * 50)
    
    # Initialize components
    scraper = FootballDataScraper()
    feature_engineer = FeatureEngineer()
    predictor = MatchPredictor(model_type='random_forest')
    
    # Example workflow
    try:
        print("\n1. Collecting match data...")
        matches_df = scraper.get_match_data("Premier League", "2024")
        print(f"   ✓ Collected {len(matches_df)} matches")
        
        print("\n2. Engineering features...")
        matches_df = predictor.prepare_target_variable(matches_df)
        features_df, feature_cols = feature_engineer.prepare_features_for_prediction(matches_df)
        print(f"   ✓ Created {len(feature_cols)} features")
        
        print("\n3. Training prediction model...")
        X = features_df[feature_cols]
        y = features_df['match_result']
        
        performance = predictor.train_model(X, y)
        print(f"   ✓ Model accuracy: {performance['accuracy']:.3f}")
        print(f"   ✓ Cross-validation: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        
        print("\n4. Making sample prediction...")
        # Create a sample match for prediction using teams from training data
        sample_match = pd.DataFrame([{
            'home_team': 'Arsenal',
            'away_team': 'Liverpool',
            'league': 'Premier League',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'home_score': 0,
            'away_score': 0,
            'season': '2024'
        }])
        
        sample_features_df, _ = feature_engineer.prepare_features_for_prediction(sample_match)
        sample_features = sample_features_df[feature_cols].iloc[0:1]
        
        prediction = predictor.predict_match(sample_features)
        print(f"   ✓ Prediction: {prediction['predicted_result']}")
        print(f"   ✓ Confidence: {prediction['confidence']:.3f}")
        
        # Display probabilities
        probs = prediction['probabilities']
        print(f"   ✓ Probabilities:")
        print(f"     - Home Win: {probs['home_win']:.3f}")
        print(f"     - Draw: {probs['draw']:.3f}")
        print(f"     - Away Win: {probs['away_win']:.3f}")
        
        print(f"\n✅ System ready! Run 'python -m src.api.main' to start the API server")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())