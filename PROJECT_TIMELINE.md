# V2G-EDA Project Timeline

## Project Overview

This project focuses on Vehicle-to-Grid (V2G) Electric Vehicle (EV) charging data analysis with two primary goals:

1. **User Segregation and Analysis**: Understanding customer behavior patterns and creating detailed user personas
2. **DR Event Forecasting**: Building predictive models for Demand Response events and electricity price forecasting

---

## Phase 1: Initial Exploratory Data Analysis

### EDA.ipynb - Initial Data Analysis and Consumer Behavior Analysis

**Objectives:**

- Perform initial exploratory data analysis on EV charging transaction data
- Identify consumer behavior patterns
- Analyze temporal and geographical charging patterns
- Identify potential DR event candidates

**Key Activities:**

- Data loading and cleaning from TransactionLogs_Jan2025-June2025.xlsx
- Feature engineering: session duration, time since last charge, weekend indicators, user loyalty ratios
- Time-series analysis of energy consumption and charging sessions
- Heatmap analysis of charging patterns by day/hour
- Customer behavior clustering using KMeans
- Charger utilization analysis
- Peak-hour user identification for DR throttling candidates

**Key Findings:**

- **Session Duration Distribution**: Highly skewed with median ~2 hours, but long tail extending to multi-day sessions
- **Temporal Patterns**: Peak demand on weekday evenings (18:00-22:00) and mid-morning
- **Geographical Concentration**: West and Central zones dominate session counts and energy delivery
- **User Loyalty**: Many customers show high loyalty to single sites; presence of opportunistic users
- **Seasonal Trend**: Increasing trend from January through June, with strong acceleration in May-June
- **Peak-Hour Daily Users**: Only ~1.5% of users charge during peak hours (12 PM-8 PM) on 90%+ of days

**DR Strategy Insights:**

- Identified top throttling candidates based on peak-hour usage patterns
- Proposed revenue model for V2G discharging to grid
- Developed customer grouping criteria for demand response programs

---

## Phase 2: User Persona Development

### Demand Response Data Preparation.ipynb - Primary Persona Modeling

**Objectives:**

- Develop primary user personas using clustering algorithms
- Incorporate geographical context through landmark analysis
- Create user-charger correlation mapping

**Key Activities:**

- Advanced feature engineering: session duration, average power, DC/AC usage patterns
- DBSCAN clustering for primary persona identification
- Geocoding of charging sites using Nominatim API
- POI (Point of Interest) analysis using Overpass API
- Landmark categorization: hospitals, schools, malls, industrial, residential areas
- User-charger frequency analysis (90th percentile threshold)
- Secondary persona development based on landmark categories

**Technical Implementation:**

- **Clustering Algorithm**: DBSCAN with eps=0.1, min_samples=10
- **Geographic Analysis**: 1,500m radius POI search around each charger
- **Feature Set**: total_sessions, total_energy, avg_energy, avg_duration, dc_fraction, loyalty_ratio
- **Output Files**: user_primary_personas.csv, charger_landmarks.csv, user_full_personas.csv

**Key Outputs:**

- Primary persona clusters for all users
- Geographic correlation between chargers and landmark categories
- User-tagging and charger-tagging datasets
- Heatmap visualization of geographic-persona correlations

---

## Phase 2.1: LLM-Enhanced Persona Generation (March 2025)

### LLM Feature Generation.ipynb - Advanced Persona Enhancement

**Objectives:**

- Generate detailed secondary personas using local LLM analysis
- Create comprehensive user profiles combining behavioral, geographic, and activity contexts
- Implement sandboxed LLM processing for data privacy

**Key Activities:**

- Integration of multiple datasets: user personas, charger correlations, transaction data, landmarks
- Frequent charger analysis (90th percentile usage patterns)
- User activity context extraction: temporal patterns, energy consumption, loyalty metrics
- Local LLM deployment using Ollama with Gemma3-4B model
- Batch processing of user profiles for persona generation
- Fallback rule-based persona generation for edge cases

**Technical Architecture:**

- **LLM Setup**: Local Ollama server with Gemma3-4B model
- **Processing Strategy**: Batch processing (20 users per batch) with retry mechanisms
- **Feature Integration**: Combined behavioral, geographic, and temporal features
- **Output**: Enhanced secondary personas with descriptive names (e.g., "Urban Commuter", "Home-Base Charger")

**Key Results:**

- **Processing Coverage**: Successfully generated personas for eligible users with sufficient data
- **Persona Diversity**: Created unique persona names capturing lifestyle patterns
- **Fallback System**: Rule-based personas for users with insufficient data
- **Output Files**: user_personas_llm_enhanced.csv, user_detailed_profiles.csv

**Top Generated Personas:**

- Home-Base Charger, Multi-Zone Resident, Business District User
- Mall Shopper, Urban Commuter, Multi-Location Traveler
- Regular Charger, Weekend Explorer, etc.

---

## Phase 3: External Data Integration

### Natural Markets Scraping Data.ipynb - Market Data Collection

**Objectives:**

- Collect comprehensive market data for enhanced DR prediction
- Integrate fuel pricing indexes and system demand data
- Create unified dataset for machine learning models

**Key Activities:**

- **Fuel Price Collection**: JKM LNG, Brent crude, coal prices from public sources
- **Data Sources**: World Bank Pink Sheet, GitHub commodity datasets, IMF data
- **Time Period**: January 1, 2025 to June 30, 2025 (48 half-hourly periods per day)
- **Data Processing**: Interpolation of missing dates, conversion to half-hourly format
- **Quality Assurance**: Data completeness validation, range verification

**Technical Implementation:**

- **API Integration**: World Bank, GitHub Datasets, Open-Meteo
- **Data Processing**: Pandas-based ETL pipeline with interpolation
- **Format Standardization**: 48 half-hourly periods per day structure
- **Output Files**: Fuel_Prices_Jan2025-Jun2025.csv, Comprehensive_Market_Data_Jan2025-Jun2025.csv

**Key Data Ranges:**

- **JKM LNG**: $8.0 - $18.5 /MMBtu (avg: $14.5)
- **Brent Crude**: $50.0 - $95.0 /barrel (avg: $75.0)
- **Coal**: $70.0 - $140.0 /ton (avg: $105.0)

### Weather.ipynb - Meteorological Data Collection

**Objectives:**

- Collect comprehensive weather data for Singapore regions
- Integrate solar radiation and cloud coverage data
- Support weather-enhanced forecasting models

**Key Activities:**

- **Station Selection**: Changi (primary), Five-region coverage (East, West, South, North, Central)
- **Data Sources**: Meteostat API, Open-Meteo API
- **Features**: Temperature, humidity, wind speed, pressure, cloud cover, solar radiation
- **Temporal Resolution**: Hourly to 30-minute interpolation
- **Quality Control**: Data validation and gap filling

**Technical Implementation:**

- **Primary Dataset**: Changi Station weather_data/weather_changi.csv
- **Regional Dataset**: Five-region coverage in weather_data/weather_region.csv
- **API Integration**: Meteostat for historical data, Open-Meteo for cloud/solar data
- **Time Processing**: UTC to SGT conversion, 30-minute resampling with linear interpolation

---

## Phase 4: Advanced DR Prediction Modeling

### DR Prediction - Data Preparation.ipynb - Comprehensive Model Development

**Objectives:**

- Develop dual-model system for price prediction and DR event forecasting
- Implement both traditional ML and deep learning approaches
- Create production-ready prediction pipeline

#### 4.1 Data Preparation and Feature Engineering

**Key Activities:**

- **DR Event Definition**: Binary classification based on 15% price increase over 4-hour moving average
- **Feature Engineering**: 50+ features including temporal lags, rolling statistics, cyclical encoding
- **Target Variables**:
  - Price prediction (regression): USEP ($/MWh)
  - DR event prediction (classification): binary 0/1 for price spikes

**Feature Categories:**

- **Temporal Features**: hour, day_of_week, month, is_weekend, is_peak_hour, cyclical encoding
- **Price Lags**: 1, 2, 4, 8, 12, 24 period lags
- **Rolling Statistics**: mean, std, max, min for windows 4, 8, 12, 24 periods
- **Momentum Features**: price changes over 1h, 2h, 6h periods
- **Volatility Measures**: rolling standard deviations
- **External Features**: demand, solar generation, weather data, fuel prices

#### 4.2 Model Development - Price Prediction

**Traditional ML Models:**

- **Linear Regression**: Baseline model (showed overfitting with R²=1.0)
- **Random Forest**: RMSE ~6.0-7.0, R² ~0.7-0.8
- **Support Vector Regressor**: Competitive performance

**Deep Learning Models:**

- **MLP**: 3-layer architecture with dropout, RMSE ~6.5-7.5
- **LSTM**: Time-series forecasting with 5-timestep sequences
- **GRU**: Alternative recurrent architecture for temporal patterns

**Model Selection:**

- **Best Performers**: Random Forest and MLP (ensemble approach)
- **Performance Metrics**: RMSE, MAE, R² for price prediction
- **Trade-off Analysis**: Accuracy vs. computational efficiency

#### 4.3 Model Development - DR Event Prediction

**Traditional ML Models:**

- **Logistic Regression**: Simple baseline with balanced class weights
- **Random Forest**: Highest recall, optimized for DR event detection
- **Support Vector Classifier**: RBF kernel with balanced classes

**Deep Learning Models:**

- **MLP**: Binary classification with dropout layers
- **LSTM**: Sequential classification with 5-timestep windows
- **GRU**: Alternative sequential classifier

**Optimization Strategy:**

- **Primary Metric**: Recall (critical for DR event detection)
- **Secondary Metrics**: Precision, F1-score
- **Class Imbalance Handling**: Balanced class weights, custom thresholds

#### 4.4 Final Model Architecture

**Ensemble Approach:**

- **Price Prediction**: Random Forest + MLP ensemble averaging
- **DR Prediction**: Random Forest + MLP probability averaging
- **Production Pipeline**: 24-hour prediction horizon with 30-minute granularity

**Performance Summary:**

| Model Type         | Price RMSE     | Price R²       | DR Recall       | DR Precision    | DR F1-Score     |
| ------------------ | -------------- | --------------- | --------------- | --------------- | --------------- |
| Random Forest      | ~6.5           | ~0.75           | ~0.85           | ~0.70           | ~0.76           |
| MLP                | ~7.0           | ~0.70           | ~0.80           | ~0.75           | ~0.77           |
| **Ensemble** | **~6.2** | **~0.78** | **~0.87** | **~0.73** | **~0.79** |

---

## Phase 5: Production Deployment

### Model Serialization and Prediction Service

**Key Activities:**

- **Model Persistence**: Saved best models in ./models/ directory
- **Scalable Architecture**: FastAPI backend for prediction serving
- **Prediction Pipeline**: 24-hour forecast with 30-minute granularity
- **Model Versioning**: Metadata tracking for model management

**Technical Implementation:**

- **Model Storage**:
  - Price models: Random Forest (.joblib) and MLP (.h5)
  - DR models: Random Forest (.joblib) and MLP (.h5)
  - Scalers: StandardScaler objects for feature preprocessing
- **Prediction Function**: `predict_next_24_hours()` with ensemble averaging
- **Input Requirements**: Full feature set matching training data structure
- **Output Format**: Timestamp, predicted_price, dr_probability, dr_event_prediction

**Production Features:**

- **Ensemble Predictions**: Averages predictions from RF and MLP models
- **Probability Calibration**: DR event probabilities from classification models
- **Error Handling**: Graceful fallbacks for model loading failures
- **Batch Processing**: Efficient handling of 48-period predictions

---

## Key Technical Innovations

### 1. Multi-Source Data Integration

- **Primary Data**: EV charging transactions (6 months, 30-minute intervals)
- **External Data**: Fuel prices, weather, solar radiation, system demand
- **Geographic Context**: POI analysis, landmark categorization
- **Temporal Features**: Cyclical encoding, lag features, rolling statistics

### 2. Advanced Persona Development

- **Primary Clustering**: DBSCAN algorithm with behavioral features
- **Geographic Enhancement**: POI-based secondary personas
- **LLM Integration**: Local Gemma3-4B model for descriptive persona names
- **Fallback Systems**: Rule-based persona generation for edge cases

### 3. Dual-Model Prediction System

- **Price Forecasting**: Regression models for USEP prediction
- **DR Event Detection**: Binary classification for price spike prediction
- **Ensemble Approach**: Combines traditional ML and deep learning models
- **Optimization Metrics**: RMSE/MAE for price, Recall for DR events

### 4. Production-Ready Pipeline

- **Model Serialization**: Standardized model storage and loading
- **Feature Engineering Pipeline**: Automated feature creation and scaling
- **Prediction Service**: Real-time 24-hour forecasting capability
- **Quality Assurance**: Comprehensive validation and error handling

---

## Performance Metrics Summary

### User Persona Analysis

- **Total Users Processed**: ~12,000 unique customers
- **Primary Persona Clusters**: 3-5 distinct behavioral groups
- **Secondary Persona Categories**: 10+ descriptive lifestyle patterns
- **Geographic Coverage**: 100% of charging sites with landmark analysis

### DR Prediction Performance

- **Price Prediction RMSE**: ~6.2 $/MWh (ensemble)
- **Price Prediction R²**: ~0.78 (ensemble)
- **DR Event Recall**: ~87% (critical for grid stability)
- **DR Event Precision**: ~73% (balanced approach)
- **Prediction Horizon**: 24 hours with 30-minute granularity

### Data Quality Metrics

- **Data Completeness**: >95% across all integrated sources
- **Temporal Coverage**: 6 months (Jan-Jun 2025)
- **Geographic Coverage**: All Singapore charging zones
- **Feature Engineering**: 50+ engineered features per time period

---

## Lessons Learned and Future Improvements

### Technical Challenges Overcome

1. **Data Integration**: Successfully merged heterogeneous data sources (transactions, weather, fuel prices)
2. **Class Imbalance**: Handled DR event rarity through balanced class weights
3. **Geographic Processing**: Implemented efficient POI analysis for thousands of charging sites
4. **LLM Privacy**: Maintained data privacy with local LLM deployment

### Model Performance Insights

1. **Ensemble Benefits**: Combining RF and MLP models consistently outperformed individual models
2. **Feature Importance**: Temporal lags and rolling statistics were most predictive
3. **Weather Impact**: Weather features provided significant improvement in price prediction
4. **Persona Utility**: Enhanced personas enabled more targeted DR strategies

### Future Enhancement Opportunities

1. **Real-Time Integration**: Live data feeds for current market conditions
2. **Advanced Architectures**: Transformer models for sequence prediction
3. **Multi-Objective Optimization**: Simultaneous optimization of accuracy and computational efficiency
4. **Explainability**: SHAP values for model interpretability and regulatory compliance

---

## Project Deliverables

### Core Datasets

- `user_personas_llm_enhanced.csv` - Complete user persona profiles
- `Comprehensive_Market_Data_Jan2025-Jun2025.csv` - Integrated market dataset
- `weather_changi.csv` - Primary weather station data
- `Fuel_Prices_Jan2025-Jun2025.csv` - Fuel price time series

### Trained Models

- `price_model_Random Forest.joblib` - Price prediction RF model
- `price_model_MLP.h5` - Price prediction neural network
- `dr_model_Random Forest.joblib` - DR event detection RF model
- `dr_model_MLP.h5` - DR event detection neural network

### Analysis Outputs

- `PROJECT_TIMELINE.md` - Comprehensive project documentation
- `README.md` - Project overview and usage instructions
- Visualization outputs: correlation heatmaps, performance plots, geographic analyses

### Production Artifacts

- Model scalers and preprocessing objects
- Prediction pipeline with ensemble averaging
- Comprehensive metadata and versioning information

---

*Project Timeline Completed: V1 - Oct 2025 | V2 - TBD*
*Team: Xplore5* **[Xie Fei, Alan Ruan, Zhai Yifan, Aamir Syed]**
*Primary Technologies: Python, TensorFlow, scikit-learn, Pandas, Ollama (for the local LLM)*
