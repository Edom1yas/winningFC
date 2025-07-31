import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.model_performance = {}
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_target_variable(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        df = matches_df.copy()
        
        # Create target variable: 0=Away Win, 1=Draw, 2=Home Win
        conditions = [
            df['home_score'] < df['away_score'],  # Away win
            df['home_score'] == df['away_score'], # Draw
            df['home_score'] > df['away_score']   # Home win
        ]
        choices = [0, 1, 2]
        df['match_result'] = np.select(conditions, choices)
        
        return df
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        logger.info(f"Training {self.model_type} model with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance = feature_importance
        
        # Store performance metrics
        self.model_performance = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        logger.info(f"Model trained. Accuracy: {accuracy:.3f}, CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return self.model_performance
    
    def predict_match(self, match_features: pd.DataFrame) -> Dict:
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Make prediction
        prediction = self.model.predict(match_features)[0]
        probabilities = self.model.predict_proba(match_features)[0]
        
        # Map prediction to readable format
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        return {
            'predicted_result': result_map[prediction],
            'prediction_code': int(prediction),
            'probabilities': {
                'away_win': float(probabilities[0]),
                'draw': float(probabilities[1]),
                'home_win': float(probabilities[2])
            },
            'confidence': float(max(probabilities)),
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_upcoming_matches(self, matches_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        predictions = []
        
        for idx, match in matches_df.iterrows():
            match_features = match[feature_cols].values.reshape(1, -1)
            match_features_df = pd.DataFrame(match_features, columns=feature_cols)
            
            prediction_result = self.predict_match(match_features_df)
            
            predictions.append({
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'date': match['date'],
                'predicted_result': prediction_result['predicted_result'],
                'home_win_prob': prediction_result['probabilities']['home_win'],
                'draw_prob': prediction_result['probabilities']['draw'],
                'away_win_prob': prediction_result['probabilities']['away_win'],
                'confidence': prediction_result['confidence']
            })
        
        return pd.DataFrame(predictions)
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        logger.info("Optimizing hyperparameters...")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            }
        }
        
        if self.model_type in param_grids:
            param_grid = param_grids[self.model_type]
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        
        return {}
    
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'performance': self.model_performance,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data.get('feature_importance')
        self.model_performance = model_data.get('performance', {})
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self.feature_importance
    
    def get_model_performance(self) -> Dict:
        return self.model_performance