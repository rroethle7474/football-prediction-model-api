from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from typing import Dict, Any
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, login, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import os
from dotenv import load_dotenv
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
from storage_helper import AzureStorageHelper
# Load environment variables

load_dotenv()

# Initialize the storage helper (put this with your other initializations)
storage_helper = AzureStorageHelper()
USE_AZURE_STORAGE = os.getenv('USE_AZURE_STORAGE', 'True').lower() == 'true'


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_password = os.getenv('API_PASSWORD')
        if not auth_password:
            return jsonify({"error": "Server authentication not configured"}), 500
            
        # For GET requests, check query parameters
        if request.method == 'GET':
            provided_password = request.args.get('password')
        # For POST requests, check JSON body
        else:
            provided_password = request.json.get('password') if request.is_json else None
            
        if not provided_password:
            return jsonify({"error": "No password provided"}), 401
            
        if provided_password != auth_password:
            return jsonify({"error": "Invalid password"}), 401
            
        return f(*args, **kwargs)
    return decorated



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class NFLPredictor:
    def __init__(self):
        self.hf_api = HfApi()
        # Get credentials from environment variables
        self.hf_username = os.getenv('HUGGINGFACE_USERNAME')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.team_stats = None
        self.last_stats_update = None
        
        if not self.hf_username:
            raise ValueError("HUGGINGFACE_USERNAME not found in environment variables")

    def _ensure_login(self):
        """Ensure we're logged in before making HuggingFace API calls"""
        if self.hf_token:
            login(token=self.hf_token)
        else:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
        
    @lru_cache(maxsize=1)
    def _load_raw_data(self) -> tuple:
        """Load raw CSV files with caching"""
        print("Loading raw data from CSV files")
        return (
            pd.read_csv('features/PassingOffense-Yards.csv'),
            pd.read_csv('features/PassingOffense-Attempts.csv'),
            pd.read_csv('features/RushingOffense-Yards.csv'),
            pd.read_csv('features/RushingOffense-Attempts.csv'),
            pd.read_csv('features/PassingDefense.csv'),
            pd.read_csv('features/RushingDefense.csv'),
            pd.read_csv('features/TOP.csv')
        )

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess data with caching"""
        # Check if we have fresh cached stats
        if self._are_stats_fresh():
            print("Using cached team stats")
            return self.team_stats
        
        # Load raw data using cached method
        (
            passing_off_yards,
            passing_off_attempts,
            rushing_off_yards,
            rushing_off_attempts,
            passing_def,
            rushing_def,
            top
        ) = self._load_raw_data()
        
        # Merge all dataframes on Team
        data = passing_off_yards.merge(passing_off_attempts, on='Team', suffixes=('_py', '_pa'))
        data = data.merge(rushing_off_yards, on='Team', suffixes=('', '_ry'))
        data = data.merge(rushing_off_attempts, on='Team', suffixes=('', '_ra'))
        data = data.merge(passing_def, on='Team', suffixes=('', '_pd'))
        data = data.merge(rushing_def, on='Team', suffixes=('', '_rd'))
        data = data.merge(top, on='Team', suffixes=('', '_top'))
        
        # Update cache
        self.team_stats = data
        self.last_stats_update = datetime.now()
        
        return data

    def train_model(self, model_name: str, training_split: float = 0.2) -> Dict[str, Any]:
        try:
            print("Starting train_model method")  # Debug print
            
            # Validate training_split
            if not 0 < training_split < 1:
                return {
                    "status": "error",
                    "message": "Training split must be between 0 and 1"
                }

            # Ensure we're logged in before making API calls
            self._ensure_login()
            print("Login successful")  # Debug print

            # Check if model already exists
            try:
                print(f"Checking if model {model_name} exists")  # Debug print
                self.hf_api.model_info(
                    repo_id=f"{self.hf_username}/{model_name}",
                    token=self.hf_token
                )
                # If we get here, the model exists
                return {
                    "status": "error", 
                    "message": f"Model '{model_name}' already exists. Please choose a different name or delete the existing model first."
                }
            except RepositoryNotFoundError:
                print("Model does not exist, proceeding with creation")  # Debug print
                pass

            # Create repository
            try:
                print("Creating repository")  # Debug print
                self.hf_api.create_repo(
                    repo_id=f"{self.hf_username}/{model_name}",
                    repo_type="model",
                    private=True
                )
            except Exception as repo_error:
                print(f"Repository creation error: {str(repo_error)}")  # Debug print
                return {"status": "error", "message": f"Error creating repository: {str(repo_error)}"}

            # Load and preprocess data
            print("Loading and preprocessing data")  # Debug print
            team_stats = self.load_and_preprocess_data()
            actual_results = self._load_actual_results()

            print("Preparing features and targets")  # Debug print
            # Prepare features
            X = self._prepare_features(actual_results, team_stats)
            
            # Prepare multiple targets
            y = self._prepare_targets(actual_results)

            print("Splitting data")  # Debug print
            # Split data using provided training_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=training_split, random_state=42
            )

            print("Training model")  # Debug print
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize multi-output model
            model = MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))

            # Train model
            model.fit(X_train_scaled, y_train)

            # Save model and scaler
            model_filename = f"{model_name}_model.joblib"
            scaler_filename = f"{model_name}_scaler.joblib"
            
            joblib.dump(model, model_filename)
            joblib.dump(scaler, scaler_filename)
            print("ABOUT TO UPLOAD TO HUGGINGFACE")
            # Upload to HuggingFace
            self.hf_api.upload_file(
                path_or_fileobj=model_filename,
                path_in_repo=model_filename,
                repo_id=f"{self.hf_username}/{model_name}",
                repo_type="model"
            )
            
            self.hf_api.upload_file(
                path_or_fileobj=scaler_filename,
                path_in_repo=scaler_filename,
                repo_id=f"{self.hf_username}/{model_name}",
                repo_type="model"
            )

            # Cleanup local files
            os.remove(model_filename)
            os.remove(scaler_filename)

            return {"status": "success", "message": "Model trained and uploaded successfully"}

        except Exception as train_error:
            print(f"Training error occurred: {str(train_error)}")  # Debug print
            # If training fails, attempt to delete the repository
            try:
                print(f"Attempting to clean up repository after error")  # Debug print
                self.hf_api.delete_repo(
                    repo_id=f"{self.hf_username}/{model_name}",
                    repo_type="model",
                    token=self.hf_token
                )
            except Exception as delete_error:
                print(f"Cleanup error occurred: {str(delete_error)}")  # Debug print
                return {
                    "status": "error", 
                    "message": f"Training failed and cleanup failed. Training error: {str(train_error)}. Cleanup error: {str(delete_error)}"
                }
            
            return {"status": "error", "message": f"Training failed and repository was cleaned up. Error: {str(train_error)}"}
        
    def update_team_stats(self) -> Dict[str, Any]:
        """Update team statistics cache"""
        try:
            self.team_stats = self.load_and_preprocess_data()
            self.last_stats_update = datetime.now()
            return {"status": "success", "message": "Team stats updated successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_team_stats(self) -> Dict[str, Any]:
        """Get current team statistics and last update time"""
        if self.team_stats is None:
            return {
                "status": "not_initialized",
                "message": "Team stats have not been loaded yet",
                "last_update": None,
                "teams": None
            }
        
        return {
            "status": "success",
            "message": "Team stats retrieved successfully",
            "last_update": self.last_stats_update.isoformat(),
            "teams": list(self.team_stats['Team'])
        }

    def predict(self, home_team: str, away_team: str, model_name: str) -> Dict[str, Any]:
        """Make predictions for a game"""
        try:
            # Ensure we're logged in before making API calls
            self._ensure_login()
            
            # Download model and scaler from Hugging Face
            try:
                model_path = hf_hub_download(
                    repo_id=f"{self.hf_username}/{model_name}",
                    filename=f"{model_name}_model.joblib",
                    token=self.hf_token
                )
                scaler_path = hf_hub_download(
                    repo_id=f"{self.hf_username}/{model_name}",
                    filename=f"{model_name}_scaler.joblib",
                    token=self.hf_token
                )
            except Exception as download_error:
                return {
                    "status": "error",
                    "message": f"Error downloading model files: {str(download_error)}"
                }

            # Load model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Get team stats
            team_stats = self.load_and_preprocess_data()
            
            # Prepare features for both home and away predictions
            X_home = self._prepare_prediction_features(home_team, away_team, team_stats, is_home=True)
            X_away = self._prepare_prediction_features(away_team, home_team, team_stats, is_home=False)
            
            # Scale features
            X_home_scaled = scaler.transform(X_home)
            X_away_scaled = scaler.transform(X_away)
            
            # Make predictions
            home_predictions = model.predict(X_home_scaled)
            away_predictions = model.predict(X_away_scaled)
            
            # Format predictions
            result = {
                "status": "success",
                "predictions": {
                    "home_team": {
                        "team": home_team,
                        "passing_yards": round(float(home_predictions[0][0]), 2),
                        "passing_attempts": round(float(home_predictions[0][1]), 2),
                        "rushing_yards": round(float(home_predictions[0][2]), 2),
                        "rushing_attempts": round(float(home_predictions[0][3]), 2),
                        "time_of_possession": round(float(home_predictions[0][4]), 2)
                    },
                    "away_team": {
                        "team": away_team,
                        "passing_yards": round(float(away_predictions[0][0]), 2),
                        "passing_attempts": round(float(away_predictions[0][1]), 2),
                        "rushing_yards": round(float(away_predictions[0][2]), 2),
                        "rushing_attempts": round(float(away_predictions[0][3]), 2),
                        "time_of_possession": round(float(away_predictions[0][4]), 2)
                    }
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _prepare_features(self, actual_results: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive feature set capturing game dynamics"""
        try:
            print("Preparing features")
            print("Team stats columns:", team_stats.columns)
            print("Actual results columns:", actual_results.columns)
            
            features = pd.DataFrame()
            
            for idx, row in actual_results.iterrows():
                try:
                    team = row['Team']
                    print(f"Processing team: {team}")
                    
                    # Get opponent
                    opponent = actual_results[
                        (actual_results['MatchupId'] == row['MatchupId']) & 
                        (actual_results['Team'] != team)
                    ]['Team'].iloc[0]
                    print(f"Opponent: {opponent}")
                    
                    # Get team and opponent data
                    team_data = team_stats[team_stats['Team'] == team].iloc[0]
                    opponent_data = team_stats[team_stats['Team'] == opponent].iloc[0]
                    
                    # Print IsHomeTeam value before conversion
                    print(f"IsHomeTeam value: {row['IsHomeTeam']}, type: {type(row['IsHomeTeam'])}")
                    
                    feature_row = {
                        # Team offensive metrics
                        'Team_PassOff_Yards': team_data['2024_py'],
                        'Team_PassOff_Attempts': team_data['2024_pa'],
                        'Team_RushOff_Yards': team_data['2024'],
                        'Team_RushOff_Attempts': team_data['2024_ra'],
                        'Team_TOP_min': team_data['2024-min'],
                        
                        # Team defensive metrics
                        'Team_PassDef': team_data['2024_pd'],
                        'Team_RushDef': team_data['2024_rd'],
                        
                        # Opponent metrics
                        'Opp_PassOff_Yards': opponent_data['2024_py'],
                        'Opp_PassOff_Attempts': opponent_data['2024_pa'],
                        'Opp_RushOff_Yards': opponent_data['2024'],
                        'Opp_RushOff_Attempts': opponent_data['2024_ra'],
                        'Opp_TOP_min': opponent_data['2024-min'],
                        'Opp_PassDef': opponent_data['2024_pd'],
                        'Opp_RushDef': opponent_data['2024_rd'],
                        
                        # Recent performance (Last 3 games)
                        'Team_Last3_PassYards': team_data['Last3_py'],
                        'Team_Last3_RushYards': team_data['Last3'],
                        'Opp_Last3_PassYards': opponent_data['Last3_py'],
                        'Opp_Last3_RushYards': opponent_data['Last3'],
                        
                        # Home/Away splits
                        'Team_Home_PassYards': team_data['Home_py'],
                        'Team_Away_PassYards': team_data['Away_py'],
                        'Team_Home_RushYards': team_data['Home'],
                        'Team_Away_RushYards': team_data['Away'],
                        
                        # Game context
                        'Is_Home': 1 if str(row['IsHomeTeam']).lower() == 'yes' else 0,
                        
                        # Derived features
                        'Team_YardsPerPass': team_data['2024_py'] / team_data['2024_pa'],
                        'Team_YardsPerRush': team_data['2024'] / team_data['2024_ra'],
                        'TOP_Ratio': team_data['2024-min'] / opponent_data['2024-min']
                    }
                    
                    features = pd.concat([features, pd.DataFrame([feature_row])], ignore_index=True)
                    
                except Exception as row_error:
                    print(f"Error processing row {idx}: {str(row_error)}")
                    print(f"Row data: {row}")
                    raise
                    
            print("Features shape:", features.shape)
            return features
            
        except Exception as e:
            print(f"Error in _prepare_features: {str(e)}")
            raise

    def _prepare_targets(self, actual_results: pd.DataFrame) -> np.ndarray:
        """Prepare multi-output target variables"""
        try:
            print("Preparing targets with columns:", actual_results.columns)
            print("Sample of actual_results data:\n", actual_results.head())
            
            # Convert TOP to total minutes for easier prediction
            top_minutes = actual_results['TOP-min'] + actual_results['TOP-sec'] / 60
            
            targets = np.column_stack([
                actual_results['PassingYards'],
                actual_results['PassingAttempts'],
                actual_results['RushingYards'],
                actual_results['RushingAttempts'],
                top_minutes
            ])
            
            print("Target shape:", targets.shape)
            return targets
            
        except Exception as e:
            print(f"Error in _prepare_targets: {str(e)}")
            print(f"Data types of columns:\n{actual_results.dtypes}")
            raise

    def _prepare_prediction_features(self, team: str, opponent: str, team_stats: pd.DataFrame, is_home: bool) -> pd.DataFrame:
        """Prepare features for a single prediction"""
        team_data = team_stats[team_stats['Team'] == team].iloc[0]
        opponent_data = team_stats[team_stats['Team'] == opponent].iloc[0]
        
        feature_row = {
            # Team offensive metrics
            'Team_PassOff_Yards': team_data['2024_py'],
            'Team_PassOff_Attempts': team_data['2024_pa'],
            'Team_RushOff_Yards': team_data['2024'],
            'Team_RushOff_Attempts': team_data['2024_ra'],
            'Team_TOP_min': team_data['2024-min'],
            
            # Team defensive metrics
            'Team_PassDef': team_data['2024_pd'],
            'Team_RushDef': team_data['2024_rd'],
            
            # Opponent metrics
            'Opp_PassOff_Yards': opponent_data['2024_py'],
            'Opp_PassOff_Attempts': opponent_data['2024_pa'],
            'Opp_RushOff_Yards': opponent_data['2024'],
            'Opp_RushOff_Attempts': opponent_data['2024_ra'],
            'Opp_TOP_min': opponent_data['2024-min'],
            'Opp_PassDef': opponent_data['2024_pd'],
            'Opp_RushDef': opponent_data['2024_rd'],
            
            # Recent performance (Last 3 games)
            'Team_Last3_PassYards': team_data['Last3_py'],
            'Team_Last3_RushYards': team_data['Last3'],
            'Opp_Last3_PassYards': opponent_data['Last3_py'],
            'Opp_Last3_RushYards': opponent_data['Last3'],
            
            # Home/Away splits
            'Team_Home_PassYards': team_data['Home_py'],
            'Team_Away_PassYards': team_data['Away_py'],
            'Team_Home_RushYards': team_data['Home'],
            'Team_Away_RushYards': team_data['Away'],
            
            # Game context
            'Is_Home': int(is_home),
            
            # Derived features
            'Team_YardsPerPass': team_data['2024_py'] / team_data['2024_pa'],
            'Team_YardsPerRush': team_data['2024'] / team_data['2024_ra'],
            'TOP_Ratio': team_data['2024-min'] / opponent_data['2024-min']
        }
        
        return pd.DataFrame([feature_row])

    def get_user_models(self) -> Dict[str, Any]:
        """Get all models (private and public) associated with the user"""
        try:
            # Ensure we're logged in before making API calls
            self._ensure_login()
            
            # List all models for the user
            models = self.hf_api.list_models(
                author=self.hf_username,  # Filter by username
                use_auth_token=self.hf_token  # Include private models
            )
            
            # Extract model names from the results
            model_names = [model.modelId.split('/')[-1] for model in models]
            
            return {
                "status": "success",
                "models": model_names
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _are_stats_fresh(self) -> bool:
        """Check if cached stats are available and fresh (less than 1 day old)"""
        return (
            self.team_stats is not None 
            and self.last_stats_update is not None
            and (datetime.now() - self.last_stats_update) < timedelta(days=1)
        )

    @lru_cache(maxsize=1)
    def _load_actual_results(self) -> pd.DataFrame:
        """Load actual results with caching"""
        return pd.read_csv('training_data/ActualResults.csv')

    def clear_cache(self) -> Dict[str, Any]:
        """Clear all caches (both LRU and instance properties)"""
        try:
            # Clear LRU caches
            self._load_raw_data.cache_clear()
            self._load_actual_results.cache_clear()
            
            # Reset instance properties
            self.team_stats = None
            self.last_stats_update = None
            
            return {
                "status": "success",
                "message": "All caches cleared successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model from Hugging Face Hub"""
        try:
            # Ensure we're logged in before making API calls
            self._ensure_login()
            
            # Delete the repository
            self.hf_api.delete_repo(
                repo_id=f"{self.hf_username}/{model_name}",
                repo_type="model",
                token=self.hf_token
            )
            
            return {
                "status": "success",
                "message": f"Model '{model_name}' successfully deleted"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

predictor = NFLPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    home_team = data.get('homeTeam')
    away_team = data.get('awayTeam')
    model_name = data.get('modelName')
    print("HOME_TEAM: ", home_team)
    print("AWAY_TEAM: ", away_team)
    print("MODEL_NAME: ", model_name)
    if not all([home_team, away_team, model_name]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    result = predictor.predict(home_team, away_team, model_name)
    return jsonify(result)

@app.route('/trainModel', methods=['POST'])
@requires_auth
def train_model():
    print("TRAINING MODEL")
    data = request.get_json()
    print("DATA: ", data)
    model_name = data.get('modelName')
    training_split = data.get('trainingSplit', 0.2)
    print("TRAINING SPLIT: ", training_split)# Default to 0.2 if not provided
    
    if not model_name:
        return jsonify({"error": "Missing modelName parameter"}), 400
    
    try:
        training_split = float(training_split)
        if not 0 < training_split < 1:
            return jsonify({"error": "trainingSplit must be between 0 and 1"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "trainingSplit must be a valid number between 0 and 1"}), 400
    
    result = predictor.train_model(model_name, training_split)
    return jsonify(result)

@app.route('/updateStats', methods=['POST'])
@requires_auth
def update_stats():
    """Endpoint to manually trigger team stats update"""
    result = predictor.update_team_stats()
    return jsonify(result)

@app.route('/statsStatus', methods=['GET'])
def stats_status():
    """Endpoint to check current status of team stats"""
    result = predictor.get_team_stats()
    return jsonify(result)

@app.route('/models', methods=['GET'])
def get_models():
    """Endpoint to get all models associated with the user"""
    result = predictor.get_user_models()
    return jsonify(result)

@app.route('/clearCache', methods=['POST'])
@requires_auth
def clear_cache():
    """Endpoint to clear all caches"""
    result = predictor.clear_cache()
    return jsonify(result)

@app.route('/deleteModel', methods=['POST'])
@requires_auth
def delete_model():
    """Endpoint to delete a model from Hugging Face Hub"""
    data = request.get_json()
    model_name = data.get('modelName')
    
    if not model_name:
        return jsonify({
                "status": "error",
                "message": "No modelName parameter provided"
            })
    
    result = predictor.delete_model(model_name)
    return jsonify(result)

@app.route('/savePassingDefense', methods=['POST'])
@requires_auth
def save_passing_defense():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
        
                # Upload to Azure Storage
        storage_helper.upload_file(
            'features',
            'PassingDefense.csv',
            file.read()
        )
        # Save the file, overwriting if it exists
        # file.save('features/PassingDefense.csv')
        
        # Clear the predictor's cache since data has changed
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "PassingDefense.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/savePassingOffenseYards', methods=['POST'])
@requires_auth
def save_passing_offense_yards():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        file.save('features/PassingOffense-Yards.csv')
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "PassingOffense-Yards.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/savePassingOffenseAttempts', methods=['POST'])
@requires_auth
def save_passing_offense_attempts():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        file.save('features/PassingOffense-Attempts.csv')
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "PassingOffense-Attempts.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/saveRushingDefense', methods=['POST'])
@requires_auth
def save_rushing_defense():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        file.save('features/RushingDefense.csv')
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "RushingDefense.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/saveRushingOffenseYards', methods=['POST'])
@requires_auth
def save_rushing_offense_yards():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        file.save('features/RushingOffense-Yards.csv')
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "RushingOffense-Yards.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/saveRushingOffenseAttempts', methods=['POST'])
@requires_auth
def save_rushing_offense_attempts():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        file.save('features/RushingOffense-Attempts.csv')
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "RushingOffense-Attempts.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/saveTimeOfPossession', methods=['POST'])
@requires_auth
def save_time_of_possession():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        file.save('features/TOP.csv')
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "TOP.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/saveActualResults', methods=['POST'])
@requires_auth
def save_actual_results():
    try:
        print("SAVING ACTUAL RESULTS")
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
            
        # Save the file, overwriting if it exists
        file.save('training_data/ActualResults.csv')
        
        # Clear the predictor's cache since data has changed
        predictor.clear_cache()
        
        return jsonify({"status": "success", "message": "ActualResults.csv updated successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)