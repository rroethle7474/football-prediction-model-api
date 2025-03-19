# Football Prediction Model API

This API supports the Frontend for predicting total yards by various statistics for future NFL games. Frontend project found here: https://github.com/rroethle7474/football-prediction-model-web.

Production URL found for web found here: https://www.jclleague.com/

## Table of Contents
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
- [Technologies Used](#technologies-used)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Code Architecture and Key Components](#code-architecture-and-key-components)
- [Future Improvements](#future-improvements)

## Setup and Installation

### Prerequisites
- Python 3.9+ 
- pip (Python package installer)
- A Hugging Face account (for model storage)
- Azure Storage Account (optional - for data storage)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd football-prediction-model-api
   ```

2. **Create a virtual environment**
   
   For better IDE performance, create the virtual environment outside the project directory to prevent indexing slowdowns:
   
   ```bash
   # Create a virtual environment in a parallel directory
   python -m venv ../football-prediction-venv
   
   # Activate the virtual environment
   # On Windows:
   ..\football-prediction-venv\Scripts\activate
   
   # On macOS/Linux:
   source ../football-prediction-venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file with your credentials
   # Add your Hugging Face credentials and other settings
   ```

5. **Run the application**
   ```bash
   # For development
   flask run --debug
   
   # For production
   gunicorn app:app
   ```

## Technologies Used

This API is built with the following technologies and packages:

- **Core Framework**:
  - Flask - Web framework
  - Flask-CORS - Cross-Origin Resource Sharing

- **Machine Learning**:
  - scikit-learn - Machine learning library
  - XGBoost - Gradient boosting framework
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing

- **Storage and Cloud**:
  - Azure Storage Blob - Cloud storage (optional)
  - Hugging Face Hub - Model repository and versioning

- **Utilities**:
  - python-dotenv - Environment variable management
  - joblib - Model serialization
  - gunicorn - WSGI HTTP Server for production

## Environment Variables

The `.env.example` file contains template environment variables that need to be configured:

- `HUGGINGFACE_USERNAME`: Your Hugging Face username, required for saving and retrieving models.
- `HUGGINGFACE_TOKEN`: Your Hugging Face API token, required for authentication with the Hugging Face model hub.
- `API_PASSWORD`: Password protection for sensitive API endpoints.
- `AZURE_STORAGE_CONNECTION_STRING`: Connection string for Azure Blob Storage.
- `USE_AZURE_STORAGE`: Set to "True" to use Azure Storage for storing data files, or "False" to use local file system storage.

When `USE_AZURE_STORAGE` is set to "False", the application will store all data files in local directories within the project structure. This is useful for local development. When set to "True", the application will store files in Azure Blob Storage containers, which is recommended for production deployments.

## API Endpoints

The API provides several endpoints for model training, prediction, and data management:

- `/predict` - Get predictions for NFL team stats
- `/trainModel` - Train a new prediction model
- `/updateStats` - Update team statistics
- `/statsStatus` - Check status of loaded team statistics
- `/models` - List all user models
- `/clearCache` - Clear data caches
- `/deleteModel` - Delete a trained model
- Various endpoints for updating CSV data files

## Code Architecture and Key Components

This section provides an overview of how the application is structured and works to help new developers understand the codebase.

### Main Components

1. **NFLPredictor Class**: The core class that handles all prediction and model management functionality:
   - Manages connections to Hugging Face for model storage and retrieval
   - Processes NFL team stats and generates machine learning features
   - Trains XGBoost regression models to predict game statistics
   - Handles caching of team stats for performance optimization

2. **Data Management**:
   - The application works with CSV files containing team statistics
   - Files are stored either locally or in Azure Blob Storage based on configuration
   - The system uses multiple CSV files to represent different aspects of team performance:
     - `PassingOffense-Yards.csv` / `PassingOffense-Attempts.csv`
     - `RushingOffense-Yards.csv` / `RushingOffense-Attempts.csv`
     - `PassingDefense.csv` / `RushingDefense.csv`
     - `TOP.csv` (Time of Possession)
     - `ActualResults.csv` (Training data with actual game results)

3. **Caching System**:
   - The application uses Python's `lru_cache` decorator for efficient data loading
   - Team statistics are cached in memory with a 24-hour expiration
   - Routes like `/clearCache` allow manual cache invalidation when needed

4. **Authentication**:
   - A simple password-based authentication system protects sensitive API routes
   - The `requires_auth` decorator is applied to routes that modify data or train models
   - The password is stored in the environment variable `API_PASSWORD`

5. **Machine Learning Pipeline**:
   - Features are created from team and opponent statistics
   - Models predict five outputs: passing yards, passing attempts, rushing yards, rushing attempts, and time of possession
   - Models are trained using `MultiOutputRegressor` with `XGBRegressor` as the base estimator
   - Trained models are stored on Hugging Face's model hub for easy retrieval

### Data Flow

1. **Training Flow**:
   - Load and preprocess team statistics from CSV files
   - Load actual game results for training
   - Generate feature matrix and target values
   - Split data into training and testing sets
   - Train model and save to Hugging Face
  
2. **Prediction Flow**:
   - Retrieve team statistics for both teams
   - Download model and scaler from Hugging Face
   - Generate feature vectors for home and away team
   - Make predictions for both teams and return formatted results

### Directory Structure

- `features/`: Contains CSV files with team statistics for offense and defense
- `training_data/`: Contains the actual results used for model training
- `app.py`: Main application file with all code (monolithic design)
- `storage_helper.py`: Helper for Azure Blob Storage operations
- `.env.example`: Template for required environment variables

### Performance Considerations

- Data is cached to minimize storage access operations
- Team stats are refreshed at most once per day automatically
- Models are stored remotely but downloaded only when needed for predictions

## Future Improvements

**TODO:**
- Separate the monolithic app.py into modular components:
  - Split into model, controller, and service layers
  - Create separate files for different functionality (auth, prediction, training, etc.)
  - Implement proper logging throughout the application
  - Add comprehensive error handling
  - Add testing framework and unit tests 