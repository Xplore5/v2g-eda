# V2G EDA: Vehicle-to-Grid Exploratory Data Analysis

Comprehensive exploratory data analysis and demand response modeling for Vehicle-to-Grid (V2G) optimizations, focusing on EV charging patterns, user behavior analysis, and grid integration strategies.

## 📋 Project Overview

This project analyzes EV charging transaction data from January-June 2025 to understand charging patterns, identify optimization opportunities, and develop demand response strategies for Vehicle-to-Grid systems. The analysis covers 196,580+ charging transactions across multiple zones and charging sites, with advanced user persona development and production-ready DR event forecasting models.

### Project Goals

1. **User Segregation and Analysis**: Understanding customer behavior patterns through advanced clustering, geographic analysis, and LLM-enhanced persona generation
2. **DR Event Forecasting**: Building predictive models for Demand Response events and electricity price forecasting with ensemble ML/DL approaches

### Model Structure

The project implements a **dual-model system**:

- **Price Prediction Model**: Regression models forecasting USEP electricity prices
- **DR Event Prediction Model**: Binary classification models detecting price spikes for demand response
- **Ensemble Approach**: Combining Random Forest and Neural Network models for optimal performance

### Key Features

- **Comprehensive EDA**: Time-series analysis, temporal patterns, and usage trends
- **Advanced Customer Segmentation**: Multi-stage persona development with behavioral clustering, geographic analysis, and LLM-enhanced descriptive personas
- **Demand Response Strategy**: Peak-hour throttling candidate identification with ensemble ML prediction models
- **Grid Integration Analysis**: Site utilization, zone-based load distribution with geographic POI correlation
- **Revenue Modeling**: V2G discharge potential and dynamic pricing analysis
- **Retention Analysis**: Customer loyalty and repeat usage patterns
- **External Data Integration**: Fuel pricing (JKM LNG, Brent crude, coal), weather data, and system demand metrics
- **Production ML Models**: Dual-model system for price prediction and DR event forecasting with ensemble approaches
- **Geographic Intelligence**: POI-based landmark analysis for enhanced user persona development
- **Weather-Enhanced Forecasting**: Integration of meteorological data for improved prediction accuracy

### Key Findings

#### User Persona Analysis

- **Primary Persona Clusters**: 3-5 distinct behavioral groups identified through DBSCAN clustering
- **Enhanced Secondary Personas**: 10+ descriptive lifestyle patterns generated through LLM analysis including "Urban Commuter", "Home-Base Charger", "Multi-Zone Resident", "Business District User", "Mall Shopper"
- **Geographic-Persona Correlation**: Strong correlation between charging locations and user lifestyle patterns (e.g., residential areas vs. commercial districts)
- **Peak-Hour Specialists**: Only ~1.5% of users charge during peak hours (12 PM-8 PM) on 90%+ of days, representing prime DR candidates
- **Customer Loyalty Patterns**: High site loyalty with many users frequenting single locations, enabling targeted DR strategies

#### Data Patterns and Insights

- **Temporal Patterns**: Peak demand occurs on weekday evenings (18:00-22:00) with weekend demand more evenly distributed
- **Geographic Distribution**: West and Central zones dominate both session counts and energy delivery
- **Session Duration Variability**: Highly skewed distribution with median ~2 hours but long tail extending to multi-day charging
- **Seasonal Trends**: Increasing demand trend from January through June, particularly strong in May-June
- **Customer Retention**: Only ~8% of customers used charging services all 6 months, suggesting need for improved loyalty programs

#### DR Prediction Model Performance

- **Price Prediction Accuracy**: Ensemble model achieves RMSE ~6.2 $/MWh and R² ~0.78
- **DR Event Detection**: High recall of ~87% for capturing potential price spikes critical for grid stability
- **Weather Enhancement**: Meteorological features significantly improve prediction accuracy
- **Feature Importance**: Temporal lags and rolling statistics are most predictive features

#### External Data Integration Impact

- **Fuel Price Correlation**: JKM LNG prices show strong correlation with electricity price movements
- **Weather Influence**: Temperature and cloud coverage significantly impact charging patterns and price predictions
- **Multi-Source Synergy**: Integration of fuel prices, weather, and demand data improves overall model performance

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd v2g-eda
   ```
2. **Install uv** (if not already installed):

   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or via pip
   pip install uv
   ```
3. **Create and activate virtual environment with Python 3.10**:

   ```bash
   uv python install 3.10
   uv venv --python 3.10
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. **Install project dependencies**:

   ```bash
   uv sync --all-groups
   ```

   This will install all required packages:

   - `matplotlib>=3.10.5` - Data visualization
   - `numpy>=2.2.6` - Numerical computing
   - `pandas>=2.3.1` - Data manipulation and analysis
   - `scikit-learn>=1.7.1` - Machine learning and clustering
   - `seaborn>=0.13.2` - Statistical data visualization
   - `ipykernel` - Running Python interactively/notebooks

### Data Requirements

Place your data files in the appropriate directories:

#### uploads/ directory (primary transaction data)

- `TransactionLogs_Jan2025-June2025.xlsx` - Main EV charging transaction data (Jan-Jun 2025)
- `TransactionLogs_Jan2025-June2025_updated.xlsx` - Updated transaction data with charger details
- `transaction_data_tagged.csv` - Pre-processed tagged transaction data
- `user_full_personas_generated.csv` - Generated user persona profiles
- `charger_persona_correlation_generated.csv` - Charger-persona correlation data
- `charger_landmarks_heuristic.csv` - Geographic landmark data for chargers

#### Model files (generated after training)

- `models/price_model_Random Forest.joblib` - Price prediction Random Forest model
- `models/price_model_MLP.h5` - Price prediction Neural Network model
- `models/dr_model_Random Forest.joblib` - DR event detection Random Forest model
- `models/dr_model_MLP.h5` - DR event detection Neural Network model
- `models/scaler_price.joblib` - Feature scaler for price prediction
- `models/scaler_dr.joblib` - Feature scaler for DR prediction
- `models/model_metadata.pkl` - Model metadata and feature information

## 🚀 Usage

### Running the Analysis

1. **Open and run the analysis notebooks in sequence**:

   - `EDA.ipynb` - Initial exploratory data analysis and consumer behavior analysis
   - `Demand Response Data Preparation.ipynb` - Primary persona modeling and geographic analysis
   - `LLM Feature Generation.ipynb` - Advanced persona enhancement with local LLM
   - `Natural Markets Scraping Data.ipynb` - External market data collection
   - `Weather.ipynb` - Meteorological data collection and processing
   - `DR Prediction - Data Preparation.ipynb` - Comprehensive ML model development

### Key Notebooks

#### **Phase 1: Initial Analysis**

- **`EDA.ipynb`**: Main exploratory analysis containing:
  - Data cleaning and preprocessing of 196,580+ transaction records
  - Temporal pattern analysis (daily, weekly, monthly trends)
  - Customer behavior clustering and segmentation
  - Site utilization and zone analysis
  - Peak-hour throttling candidate identification
  - Revenue potential modeling

#### **Phase 2: User Persona Development**

- **`Demand Response Data Preparation.ipynb`**: Primary persona modeling:

  - Advanced feature engineering for behavioral clustering
  - DBSCAN clustering for primary persona identification
  - Geographic analysis using Nominatim and Overpass APIs
  - POI (Point of Interest) analysis and landmark categorization
  - User-charger correlation mapping
- **`LLM Feature Generation.ipynb`**: Advanced persona enhancement:

  - Local LLM deployment using Ollama with Gemma3-4B model
  - Comprehensive user profile generation
  - Batch processing with fallback mechanisms
  - Descriptive secondary persona creation

#### **Phase 3: External Data Integration**

- **`Natural Markets Scraping Data.ipynb`**: Market data collection:

  - Fuel pricing data (JKM LNG, Brent crude, coal) from World Bank and public sources
  - System demand data integration
  - Data quality validation and interpolation
  - Comprehensive market dataset creation
- **`Weather.ipynb`**: Meteorological data collection:

  - Changi station weather data (primary dataset)
  - Five-region weather coverage for Singapore
  - Solar radiation and cloud coverage integration
  - 30-minute temporal resolution processing

#### **Phase 4: ML Model Development**

- **`DR Prediction - Data Preparation.ipynb`**: Comprehensive modeling:
  - Dual-model system development (price + DR prediction)
  - Traditional ML models (Random Forest, Linear Regression, SVR)
  - Deep learning models (MLP, LSTM, GRU)
  - Ensemble approach with model selection
  - Production-ready prediction pipeline

### Output Files

#### Data Files (Outputs/)

- `USEP-Data_Jan2025-June2025.csv` - Electricity market pricing data (48 half-hourly periods)
- `weather_changi.csv` - Primary weather station data from Changi (temperature, humidity, etc.)
- `weather_by_region.csv` - Regional weather data for 5 Singapore regions
- `Fuel_Prices_Jan2025-Jun2025.csv` - Fuel pricing data (JKM LNG, Brent crude, coal)
- `Comprehensive_Market_Data_Jan2025-Jun2025.csv` - Integrated market dataset with all external data
- **`electricity_price_prediction_data_raw.csv` -** **Raw data prepared for ML model training (final prediction data)**

#### User Persona Files (Outputs/)

- `user_primary_personas.csv` - Primary persona clustering results from DBSCAN
- `user_full_personas.csv` - Enhanced user profiles with geographic and behavioral data
- `user_personas_llm_enhanced.csv` - LLM-enhanced user personas with descriptive names
- **`user_detailed_profiles.csv` - Comprehensive user profiles with all features**
- `user_tagging.xlsx` - User tagging data with persona classifications
- `charger_tagging.xlsx` - Charger tagging data with dominant personas

#### Geographic Analysis Files (Outputs/)

- `charger_landmarks.csv` - POI analysis results for all charging locations
- `charger_persona_correlation.csv` - Correlation between chargers and user personas

#### Model Files (models/)

- `price_model_Random Forest.joblib` - Trained Random Forest model for price prediction
- `price_model_MLP.h5` - Trained Neural Network model for price prediction
- `dr_model_Random Forest.joblib` - Trained Random Forest model for DR event detection
- `dr_model_MLP.h5` - Trained Neural Network model for DR event detection
- `scaler_price.joblib` - Feature scaler for price prediction models
- `scaler_dr.joblib` - Feature scaler for DR prediction models
- `model_metadata.pkl` - Model metadata including feature columns and configuration

## 📊 Analysis Components

### 1. Data Preprocessing and Integration

- Cleaning 196,580+ transaction records
- Advanced feature engineering (temporal features, duration calculations, behavioral metrics)
- Missing data handling and outlier detection
- Multi-source data integration (fuel prices, weather, system demand)
- Geographic data processing with POI analysis

### 2. Temporal Analysis

- Daily/weekly/monthly energy consumption trends
- Hour-of-day and day-of-week usage patterns
- Seasonal trend identification
- Peak demand pattern analysis
- Time-series forecasting with weather enhancement

### 3. Advanced Customer Segmentation

- **Primary Persona Development**: DBSCAN clustering with behavioral features
- **Geographic Enhancement**: POI-based secondary persona generation
- **LLM-Enhanced Personas**: Local Gemma3-4B model for descriptive persona names
- **User-Charger Correlation**: Mapping between users and their preferred charging locations
- **Loyalty Analysis**: Customer retention and repeat usage patterns

### 4. Infrastructure and Geographic Analysis

- EVSE (charger gun) utilization rates
- Site-level performance metrics
- Zone-based load distribution
- Geographic correlation analysis between chargers and landmark categories
- Regional weather impact on charging patterns

### 5. Demand Response Strategy and Prediction

- **Traditional DR Analysis**: Peak-hour throttling candidate identification
- **Revenue Modeling**: V2G discharge potential and dynamic pricing analysis
- **ML-Based DR Prediction**: Binary classification models for price spike detection
- **Price Forecasting**: Regression models for USEP electricity price prediction
- **Ensemble Modeling**: Combining Random Forest and Neural Network approaches

### 6. External Data Integration and Analysis

- **Fuel Pricing Integration**: JKM LNG, Brent crude, coal price correlation analysis
- **Weather Enhancement**: Temperature, humidity, solar radiation impact on predictions
- **Market Data Analysis**: System demand and electricity market dynamics
- **Multi-Source Synergy**: Combined impact of external factors on DR predictions

### 7. Production ML Pipeline

- **Feature Engineering Pipeline**: 50+ engineered features including temporal lags, rolling statistics, cyclical encoding
- **Model Training and Selection**: Traditional ML and Deep Learning model comparison
- **Ensemble Approach**: Optimal model combination for price and DR prediction
- **Production Deployment**: Serialized models with prediction functions for 24-hour forecasting

## Key Insights for V2G Optimization

1. **Load Shifting Opportunities**: Peak demand concentration in evening hours presents clear opportunities for demand response programs
2. **Customer Segmentation**: Distinct user patterns enable targeted incentive programs for different customer personas
3. **Infrastructure Focus**: West and Central zones show highest utilization, suggesting priority areas for V2G infrastructure deployment
4. **Retention Strategy**: Low customer retention rates indicate need for improved loyalty programs to ensure V2G participation

## DR Prediction System

The project now includes a comprehensive Demand Response (DR) prediction system that can forecast electricity prices and detect potential DR events for the next 24 hours.

### Model Components

1. **Electricity Price Prediction Model** - Optimized for low MAE and RMSE
2. **DR Event Prediction Model** - Binary classification optimized for recall
3. **Comprehensive Feature Engineering** - Time-based, lagged, and rolling statistics
4. **Multiple ML/DL Approaches** - Traditional ML and deep learning models

### DR Event Definition

DR events are triggered when:

- Current price > 2-hour moving average * (1 + 5% threshold)
- Evaluated every 2 hours (4 periods of 30 minutes)
- Optimized for recall to capture as many potential DR events as possible

### Saved Models

The trained models are saved in the `models/` directory:

- `best_electricity_prediction_model.pkl/.h5` - Price prediction model
- `best_dr_prediction_model.pkl/.h5` - DR event prediction model
- `model_metadata.pkl` - Feature columns and model metadata
- `*_scaler.pkl` - Feature scalers for both models

### Usage Instructions

#### 1. Load and Use Models

```python
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

# Load models and metadata
metadata = joblib.load('models/model_metadata.pkl')

# Load price model
if metadata['price_model_type'] == 'sklearn':
    price_model = joblib.load('models/best_electricity_prediction_model.pkl')
    price_scaler = joblib.load('models/best_electricity_prediction_model_scaler.pkl')
else:
    price_model = tf.keras.models.load_model('models/best_electricity_prediction_model.h5')
    price_scaler = joblib.load('models/best_electricity_prediction_model_scaler.pkl')

# Load DR model
if metadata['dr_model_type'] == 'sklearn':
    dr_model = joblib.load('models/best_dr_prediction_model.pkl')
    dr_scaler = joblib.load('models/best_dr_prediction_model_scaler.pkl')
else:
    dr_model = tf.keras.models.load_model('models/best_dr_prediction_model.h5')
    dr_scaler = joblib.load('models/best_dr_prediction_model_scaler.pkl')
```

#### 2. Prepare Input Features

```python
# Create features for next 24 hours (48 periods of 30 minutes)
def create_features_for_prediction():
    dates = pd.date_range(datetime.now(), periods=48, freq='30T')
  
    features = {
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'is_peak_hour': ((dates.hour >= 7) & (dates.hour <= 19)).astype(int),
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
        'day_sin': np.sin(2 * np.pi * dates.dayofweek / 7),
        'day_cos': np.cos(2 * np.pi * dates.dayofweek / 7),
        # ... add all other required features from metadata['feature_columns']
    }
  
    return pd.DataFrame(features)

input_features = create_features_for_prediction()
```

#### 3. Make Predictions

```python
# Ensure input has all required features
X = input_features[metadata['feature_columns']].copy()

# Scale features
X_scaled_price = price_scaler.transform(X)
X_scaled_dr = dr_scaler.transform(X)

# Predict prices
price_predictions = price_model.predict(X_scaled_price)

# Predict DR events
if hasattr(dr_model, 'predict_proba'):
    dr_probabilities = dr_model.predict_proba(X_scaled_dr)[:, 1]
    dr_predictions = (dr_probabilities > 0.5).astype(int)
else:
    dr_predictions = dr_model.predict(X_scaled_dr)
    dr_probabilities = dr_predictions.copy()

# Create results
results = input_features.copy()
results['predicted_price'] = price_predictions
results['dr_probability'] = dr_probabilities
results['dr_event_prediction'] = dr_predictions
results['timestamp'] = pd.date_range(datetime.now(), periods=48, freq='30T')
```

#### 4. Sample Usage Script

A complete sample script is provided as `sample_prediction_usage.py`:

```bash
python sample_prediction_usage.py
```

This script demonstrates:

- Loading the trained models
- Creating sample features
- Making predictions for the next 24 hours
- Displaying results in a user-friendly format

### Model Performance

The system automatically selects the best models based on:

- **Price Prediction**: Lowest combined RMSE and MAE
- **DR Prediction**: Highest recall (to capture as many DR events as possible)

### Features Used

- **Time-based**: hour, day_of_week, month, weekend, peak_hour indicators
- **Cyclical Encoding**: sin/cos transformations for temporal features
- **Price Lags**: 1, 2, 4, 8, 12, 24 periods
- **Rolling Statistics**: mean, std, min, max for various windows
- **Momentum & Volatility**: Price changes and volatility measures
- **Demand & Solar**: If available in the source data

### Important Notes

- Models are trained on 30-minute interval data
- DR events are defined based on price spikes above 2-hour moving average
- The system uses class weighting to handle imbalanced DR events
- LSTM models require sequence data (minimum 5 timesteps)
- All features must be scaled using the provided scalers

## Future Development

- Enhanced demand response modeling [improve precision and recall of price prediction and DR prediction both]
- Real-time USEP pricing + weather + global market pricing integration **[need an API to access live pricing, not available currently]**
- **V2G revenue optimization algorithms**
- Extended forecast periods (7 days, 14 days, 30 days)
