from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from typing import Dict, Any
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, ModelCard, login, hf_hub_download
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
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize the storage helper
storage_helper = AzureStorageHelper()
USE_AZURE_STORAGE = os.getenv('USE_AZURE_STORAGE', 'True').lower() == 'true'

# Initialize Flask app with Swagger UI
app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', 
          title='Football Prediction Model API',
          description='API for predicting NFL game statistics using machine learning models',
          doc='/swagger')  # This will host the Swagger UI at /swagger

# Define request/response models for Swagger
prediction_input = api.model('PredictionInput', {
    'home_team': fields.String(required=True, description='Name of the home team'),
    'away_team': fields.String(required=True, description='Name of the away team'),
    'model_name': fields.String(required=True, description='Name of the model to use for prediction')
})

training_input = api.model('TrainingInput', {
    'model_name': fields.String(required=True, description='Name for the new model'),
    'training_split': fields.Float(required=False, default=0.2, description='Training/test split ratio'),
    'model_description': fields.String(required=False, description='Description of the model'),
    'password': fields.String(required=True, description='API password for authentication')
})

auth_input = api.model('AuthInput', {
    'password': fields.String(required=True, description='API password for authentication')
})

file_upload_input = api.model('FileUploadInput', {
    'file': fields.Raw(required=True, description='CSV file to upload'),
    'password': fields.String(required=True, description='API password for authentication')
})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_password = os.getenv('API_PASSWORD')
        if not auth_password:
            return jsonify({"error": "Server authentication not configured"}), 500
            
        # Handle different request types
        if request.method == 'GET':
            provided_password = request.args.get('password')
        else:
            # Check form data first (for multipart/form-data requests)
            provided_password = request.form.get('password')
            
            # If not in form data, check JSON (for application/json requests)
            if provided_password is None and request.is_json:
                provided_password = request.json.get('password')
            
        if not provided_password:
            return jsonify({"error": "No password provided"}), 401
            
        if provided_password != auth_password:
            return jsonify({"error": "Invalid password"}), 401
            
        return f(*args, **kwargs)
    return decorated

class NFLPredictor:
    def __init__(self):
        self.hf_api = HfApi()
        self.hf_username = os.getenv('HUGGINGFACE_USERNAME')
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.team_stats = None
        self.last_stats_update = None
        
        if not self.hf_username:
            raise ValueError("HUGGINGFACE_USERNAME not found in environment variables")

    def _ensure_login(self):
        if self.hf_token:
            login(token=self.hf_token)
        else:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
        
    def _read_csv_file(self, container: str, filename: str) -> pd.DataFrame:
        """Read CSV file from either Azure Storage or local filesystem"""
        if USE_AZURE_STORAGE:
            print("Downloading file from Azure Storage")
            content = storage_helper.download_file(container, filename)
            return pd.read_csv(BytesIO(content))
        else:
            print("Downloading file from local filesystem")
            # For local storage, use the directory structure from container parameter
            filepath = os.path.join(container, filename)
            return pd.read_csv(filepath)

    def _save_csv_file(self, container: str, filename: str, file_content) -> None:
        """Save CSV file to either Azure Storage or local filesystem"""
        if USE_AZURE_STORAGE:
            print("Uploading file to Azure Storage")
            storage_helper.upload_file(container, filename, file_content)
        else:
            print("Uploading file to local filesystem")
            # Ensure directory exists
            os.makedirs(container, exist_ok=True)
            filepath = os.path.join(container, filename)
            # If file_content is bytes, write in binary mode
            if isinstance(file_content, bytes):
                with open(filepath, 'wb') as f:
                    f.write(file_content)
            else:
                # Assume it's a file object from request.files
                file_content.save(filepath)

    @lru_cache(maxsize=1)
    def _load_raw_data(self) -> tuple:
        """Load raw CSV files with caching"""
        print("Loading raw data from storage")
        
        return (
            self._read_csv_file('features', 'PassingOffense-Yards.csv'),
            self._read_csv_file('features', 'PassingOffense-Attempts.csv'),
            self._read_csv_file('features', 'RushingOffense-Yards.csv'),
            self._read_csv_file('features', 'RushingOffense-Attempts.csv'),
            self._read_csv_file('features', 'PassingDefense.csv'),
            self._read_csv_file('features', 'RushingDefense.csv'),
            self._read_csv_file('features', 'TOP.csv')
        )

    def load_and_preprocess_data(self) -> pd.DataFrame:
        if self._are_stats_fresh():
            print("Using cached team stats")
            return self.team_stats
        
        (
            passing_off_yards,
            passing_off_attempts,
            rushing_off_yards,
            rushing_off_attempts,
            passing_def,
            rushing_def,
            top
        ) = self._load_raw_data()
        
        data = passing_off_yards.merge(passing_off_attempts, on='Team', suffixes=('_py', '_pa'))
        data = data.merge(rushing_off_yards, on='Team', suffixes=('', '_ry'))
        data = data.merge(rushing_off_attempts, on='Team', suffixes=('', '_ra'))
        data = data.merge(passing_def, on='Team', suffixes=('', '_pd'))
        data = data.merge(rushing_def, on='Team', suffixes=('', '_rd'))
        data = data.merge(top, on='Team', suffixes=('', '_top'))
        
        self.team_stats = data
        self.last_stats_update = datetime.now()
        
        return data
    
    def train_model(self, model_name: str, training_split: float = 0.2, model_description: str = '') -> Dict[str, Any]:
        try:
            print("Starting train_model method")
            
            if not 0 < training_split < 1:
                return {
                    "status": "error",
                    "message": "Training split must be between 0 and 1"
                }

            self._ensure_login()

            try:
                print(f"Checking if model {model_name} exists")
                self.hf_api.model_info(
                    repo_id=f"{self.hf_username}/{model_name}",
                    token=self.hf_token
                )
                return {
                    "status": "error", 
                    "message": f"Model '{model_name}' already exists. Please choose a different name or delete the existing model first."
                }
            except RepositoryNotFoundError:
                print("Model does not exist, proceeding with creation")
                pass

            try:
                print("Creating repository")
                self.hf_api.create_repo(
                    repo_id=f"{self.hf_username}/{model_name}",
                    repo_type="model",
                    private=True
                )
            except Exception as repo_error:
                print(f"Repository creation error: {str(repo_error)}")
                return {"status": "error", "message": f"Error creating repository: {str(repo_error)}"}

            try:
                readme_filename = "README.md"
                with open(readme_filename, "w", encoding="utf-8") as f:
                    f.write(model_description)  # Write the model_description to README.md
            
                # Upload README.md to Hugging Face
                self.hf_api.upload_file(
                    path_or_fileobj=readme_filename,
                    path_in_repo=readme_filename,
                    repo_id=f"{self.hf_username}/{model_name}",
                    repo_type="model"
                )
                os.remove(readme_filename)  # Clean up local file
            except Exception as readme_error:
                print(f"Model card upload failed: {str(readme_error)}")
                raise  # This will trigger the cleanup in the outer exception handler 
                      
            print("Loading and preprocessing data")
            team_stats = self.load_and_preprocess_data()
            actual_results = self._load_actual_results()

            print("Preparing features and targets")
            X = self._prepare_features(actual_results, team_stats)
            y = self._prepare_targets(actual_results)

            print("Splitting data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=training_split, random_state=42
            )

            print("Training model")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = MultiOutputRegressor(xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))

            model.fit(X_train_scaled, y_train)

            model_filename = f"{model_name}_model.joblib"
            scaler_filename = f"{model_name}_scaler.joblib"
            
            joblib.dump(model, model_filename)
            joblib.dump(scaler, scaler_filename)

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

            os.remove(model_filename)
            os.remove(scaler_filename)

            return {"status": "success", "message": "Model trained and uploaded successfully"}

        except Exception as train_error:
            print(f"Training error occurred: {str(train_error)}")
            try:
                print(f"Attempting to clean up repository after error")
                self.hf_api.delete_repo(
                    repo_id=f"{self.hf_username}/{model_name}",
                    repo_type="model",
                    token=self.hf_token
                )
            except Exception as delete_error:
                print(f"Cleanup error occurred: {str(delete_error)}")
                return {
                    "status": "error", 
                    "message": f"Training failed and cleanup failed. Training error: {str(train_error)}. Cleanup error: {str(delete_error)}"
                }
            
            return {"status": "error", "message": f"Training failed and repository was cleaned up. Error: {str(train_error)}"}
        
    def update_team_stats(self) -> Dict[str, Any]:
        try:
            self.team_stats = self.load_and_preprocess_data()
            self.last_stats_update = datetime.now()
            return {"status": "success", "message": "Team stats updated successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_team_stats(self) -> Dict[str, Any]:
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
        try:
            self._ensure_login()
            
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

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            team_stats = self.load_and_preprocess_data()
            
            X_home = self._prepare_prediction_features(home_team, away_team, team_stats, is_home=True)
            X_away = self._prepare_prediction_features(away_team, home_team, team_stats, is_home=False)
            
            X_home_scaled = scaler.transform(X_home)
            X_away_scaled = scaler.transform(X_away)
            
            home_predictions = model.predict(X_home_scaled)
            away_predictions = model.predict(X_away_scaled)
            
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
        try:
            print("Preparing features")

            features = pd.DataFrame()
            
            for idx, row in actual_results.iterrows():
                try:
                    team = row['Team']
                    print(f"Processing team: {team}")
                    
                    opponent = actual_results[
                        (actual_results['MatchupId'] == row['MatchupId']) & 
                        (actual_results['Team'] != team)
                    ]['Team'].iloc[0]
                    print(f"Opponent: {opponent}")
                    
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
        try:
            self._ensure_login()
            models = self.hf_api.list_models(
                author=self.hf_username,
                use_auth_token=self.hf_token
            )
            model_info = []
            for model in models:
                model_name = model.modelId.split('/')[-1]
                try:
                    # Get the model card/README content
                    readme_info = self.hf_api.model_info(
                        repo_id=f"{self.hf_username}/{model_name}",
                        token=self.hf_token
                    )
                    card = ModelCard.load(model.modelId)

                    model_info.append({
                        "name": model_name,
                        "description": [],
                        "tags": model.tags,
                        "lastModified": model.lastModified,
                        "readme": card.content if card.content else None
                    })
                except Exception as e:
                    print(f"Error fetching info for model {model_name}: {str(e)}")
                    # Still include the model but with minimal information
                    model_info.append({
                        "name": model_name,
                        "description": [],
                        "tags": model.tags,
                        "lastModified": model.lastModified,
                        "readme": None
                    })
                    
            return {
                "status": "success",
                "models": model_info
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _are_stats_fresh(self) -> bool:
        return (
            self.team_stats is not None 
            and self.last_stats_update is not None
            and (datetime.now() - self.last_stats_update) < timedelta(days=1)
        )

    @lru_cache(maxsize=1)
    def _load_actual_results(self) -> pd.DataFrame:
        return self._read_csv_file('training_data', 'ActualResults.csv')

    def clear_cache(self) -> Dict[str, Any]:
        try:
            self._load_raw_data.cache_clear()
            self._load_actual_results.cache_clear()
            
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
        try:
            self._ensure_login()
            
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

# Create namespaces for API organization
main_ns = api.namespace('', description='Main operations')
stats_ns = api.namespace('', description='Statistics operations')
file_ns = api.namespace('', description='File upload operations')

@main_ns.route('/predict')
class PredictEndpoint(Resource):
    @main_ns.expect(prediction_input)
    @main_ns.doc(description='Get predictions for a game between two teams')
    def post(self):
        data = request.json
        return predictor.predict(data['home_team'], data['away_team'], data['model_name'])

@main_ns.route('/trainModel')
class TrainModelEndpoint(Resource):
    @main_ns.expect(training_input)
    @main_ns.doc(description='Train a new prediction model')
    @requires_auth
    def post(self):
        data = request.json
        return predictor.train_model(
            data['model_name'],
            data.get('training_split', 0.2),
            data.get('model_description', '')
        )

@stats_ns.route('/updateStats')
class UpdateStatsEndpoint(Resource):
    @stats_ns.expect(auth_input)
    @stats_ns.doc(description='Update team statistics')
    @requires_auth
    def post(self):
        return predictor.update_team_stats()

@stats_ns.route('/statsStatus')
class StatsStatusEndpoint(Resource):
    @stats_ns.doc(description='Get status of loaded team statistics')
    def get(self):
        return predictor.get_team_stats()

@main_ns.route('/models')
class ModelsEndpoint(Resource):
    @main_ns.doc(description='List all available models')
    def get(self):
        return predictor.get_user_models()

@main_ns.route('/clearCache')
class ClearCacheEndpoint(Resource):
    @main_ns.expect(auth_input)
    @main_ns.doc(description='Clear data caches')
    @requires_auth
    def post(self):
        return predictor.clear_cache()

@main_ns.route('/deleteModel')
class DeleteModelEndpoint(Resource):
    @main_ns.expect(auth_input)
    @main_ns.doc(description='Delete a trained model')
    @requires_auth
    def post(self):
        data = request.json
        return predictor.delete_model(data.get('model_name'))

@file_ns.route('/savePassingDefense')
class SavePassingDefenseEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload passing defense statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'PassingDefense.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/savePassingOffenseYards')
class SavePassingOffenseYardsEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload passing offense yards statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'PassingOffense-Yards.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/savePassingOffenseAttempts')
class SavePassingOffenseAttemptsEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload passing offense attempts statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'PassingOffense-Attempts.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/saveRushingDefense')
class SaveRushingDefenseEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload rushing defense statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'RushingDefense.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/saveRushingOffenseYards')
class SaveRushingOffenseYardsEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload rushing offense yards statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'RushingOffense-Yards.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/saveRushingOffenseAttempts')
class SaveRushingOffenseAttemptsEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload rushing offense attempts statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'RushingOffense-Attempts.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/saveTimeOfPossession')
class SaveTimeOfPossessionEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload time of possession statistics CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('features', 'TOP.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

@file_ns.route('/saveActualResults')
class SaveActualResultsEndpoint(Resource):
    @file_ns.expect(file_upload_input)
    @file_ns.doc(description='Upload actual game results CSV')
    @requires_auth
    def post(self):
        if 'file' not in request.files:
            return {"status": "error", "message": "No file provided"}, 400
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No file selected"}, 400
        
        predictor._save_csv_file('training_data', 'ActualResults.csv', file)
        predictor.clear_cache()
        return {"status": "success", "message": "File uploaded successfully"}

if __name__ == '__main__':
    app.run(debug=True)