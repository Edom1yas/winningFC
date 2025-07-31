import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_match_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        df = matches_df.copy()
        
        # Basic features
        df['goal_difference'] = df['home_score'] - df['away_score']
        df['total_goals'] = df['home_score'] + df['away_score']
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['draw'] = (df['home_score'] == df['away_score']).astype(int)
        df['away_win'] = (df['home_score'] < df['away_score']).astype(int)
        
        # Date features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def create_team_form_features(self, matches_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        df = matches_df.copy()
        df = df.sort_values(['date'])
        
        # Calculate rolling form for home and away teams
        for team_type in ['home', 'away']:
            team_col = f'{team_type}_team'
            
            # Rolling wins, draws, losses
            df[f'{team_type}_recent_wins'] = df.groupby(team_col)[f'{team_type}_win'].rolling(window=window, min_periods=1).sum().values
            df[f'{team_type}_recent_draws'] = df.groupby(team_col)['draw'].rolling(window=window, min_periods=1).sum().values
            
            # Rolling goals for/against
            score_col = f'{team_type}_score'
            opp_score_col = 'away_score' if team_type == 'home' else 'home_score'
            
            df[f'{team_type}_recent_gf'] = df.groupby(team_col)[score_col].rolling(window=window, min_periods=1).mean().values
            df[f'{team_type}_recent_ga'] = df.groupby(team_col)[opp_score_col].rolling(window=window, min_periods=1).mean().values
        
        return df
    
    def create_head_to_head_features(self, matches_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        df = matches_df.copy()
        
        # Create h2h key
        df['h2h_key'] = df.apply(lambda x: tuple(sorted([x['home_team'], x['away_team']])), axis=1)
        
        # Historical head-to-head results
        df['h2h_home_wins'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_draws'] = 0
        
        for idx, row in df.iterrows():
            h2h_matches = df[(df['h2h_key'] == row['h2h_key']) & (df.index < idx)].tail(window)
            
            if len(h2h_matches) > 0:
                home_wins = len(h2h_matches[
                    ((h2h_matches['home_team'] == row['home_team']) & (h2h_matches['home_win'] == 1)) |
                    ((h2h_matches['away_team'] == row['home_team']) & (h2h_matches['away_win'] == 1))
                ])
                
                away_wins = len(h2h_matches) - home_wins - len(h2h_matches[h2h_matches['draw'] == 1])
                draws = len(h2h_matches[h2h_matches['draw'] == 1])
                
                df.at[idx, 'h2h_home_wins'] = home_wins
                df.at[idx, 'h2h_away_wins'] = away_wins
                df.at[idx, 'h2h_draws'] = draws
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str], fit: bool = True) -> pd.DataFrame:
        df_scaled = df.copy()
        
        if fit:
            scaled_values = self.scaler.fit_transform(df[numerical_cols])
        else:
            scaled_values = self.scaler.transform(df[numerical_cols])
        
        for i, col in enumerate(numerical_cols):
            df_scaled[f'{col}_scaled'] = scaled_values[:, i]
        
        return df_scaled
    
    def prepare_features_for_prediction(self, matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        # Apply all feature engineering steps
        df = self.create_match_features(matches_df)
        df = self.create_team_form_features(df)
        df = self.create_head_to_head_features(df)
        
        # Define feature columns
        categorical_cols = ['home_team', 'away_team', 'league']
        numerical_cols = [
            'home_recent_wins', 'home_recent_draws', 'home_recent_gf', 'home_recent_ga',
            'away_recent_wins', 'away_recent_draws', 'away_recent_gf', 'away_recent_ga',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'month', 'day_of_week', 'is_weekend'
        ]
        
        # Encode and scale features
        df = self.encode_categorical_features(df, categorical_cols)
        df = self.scale_numerical_features(df, numerical_cols)
        
        # Define final feature set
        feature_cols = []
        feature_cols.extend([f'{col}_encoded' for col in categorical_cols])
        feature_cols.extend([f'{col}_scaled' for col in numerical_cols])
        
        return df, feature_cols