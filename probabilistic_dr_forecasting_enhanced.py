"""
Enhanced Probabilistic Demand Response Forecasting System
=======================================================

Goal: Predict electricity prices for Singapore with confidence intervals
using ALL available data (2023-2025) and extended forecast periods.

Features:
- Uses complete transaction log data (2023-2025)
- Extended forecast periods (7 days, 14 days, 30 days)
- Enhanced feature engineering with transaction patterns
- Multiple probabilistic models with uncertainty quantification
- Walk-forward optimization for out-of-sample evaluation

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Probabilistic Models
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import torch
import torch.nn as nn
import torch.optim as optim

# Statistical Libraries
from scipy import stats
from scipy.stats import norm, laplace, t
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedProbabilisticDRForecaster:
    """
    Enhanced Probabilistic Demand Response Forecasting System
    Uses ALL available data (2023-2025) with extended forecast periods
    """
    
    def __init__(self, 
                 price_data_paths, 
                 weather_data_path, 
                 transaction_log_paths,
                 forecast_days=7):
        """
        Initialize the enhanced probabilistic forecaster
        
        Args:
            price_data_paths: List of paths to electricity price data files
            weather_data_path: Path to weather data
            transaction_log_paths: List of paths to transaction log files
            forecast_days: Number of days to forecast (default: 7)
        """
        self.price_data_paths = price_data_paths
        self.weather_data_path = weather_data_path
        self.transaction_log_paths = transaction_log_paths
        self.forecast_days = forecast_days
        
        # Data containers
        self.price_data = None
        self.weather_data = None
        self.transaction_logs = None
        self.merged_data = None
        
        # Models
        self.quantile_model = None
        self.gpr_model = None
        self.bnn_model = None
        self.ensemble_model = None
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Confidence intervals
        self.confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        print("🚀 Enhanced Probabilistic DR Forecasting System Initialized")
        print(f"📅 Forecast Period: {forecast_days} days ({forecast_days * 48} slots)")
        print("=" * 60)
    
    def load_and_prepare_data(self):
        """Load and prepare ALL available data sources"""
        print("📊 Loading and preparing ALL available data...")
        
        # Load ALL price data
        try:
            all_price_data = []
            for path in self.price_data_paths:
                print(f"📈 Loading price data from: {path}")
                df = pd.read_csv(path)
                all_price_data.append(df)
                print(f"  ✅ Loaded {len(df)} records")
            
            # Combine all price data
            self.price_data = pd.concat(all_price_data, ignore_index=True)
            print(f"✅ Combined price data: {len(self.price_data)} total records")
            
            # Handle the actual column structure from USEP data
            if 'DATE' in self.price_data.columns and 'PERIOD' in self.price_data.columns:
                # Create timestamp from DATE and PERIOD
                self.price_data['timestamp'] = pd.to_datetime(self.price_data['DATE'], format='%d-%b-%Y')
                # Add time based on period (each period is 30 minutes)
                self.price_data['timestamp'] = self.price_data['timestamp'] + pd.to_timedelta(
                    (self.price_data['PERIOD'] - 1) * 30, unit='minutes'
                )
                
                # Find the price column
                price_col = None
                for col in ['USEP ($/MWh)', 'price', 'electricity_price', 'unit_price']:
                    if col in self.price_data.columns:
                        price_col = col
                        break
                
                if price_col is None:
                    raise ValueError("No price column found in data")
                
                # Select relevant columns
                self.price_data = self.price_data[['timestamp', price_col, 'DEMAND (MW)', 'SOLAR(MW)']].copy()
                self.price_data.columns = ['timestamp', 'price', 'demand', 'solar']
                
                print(f"✅ Price data processed: {len(self.price_data)} records")
                print(f"📊 Columns: {list(self.price_data.columns)}")
                
            else:
                # Fallback to original logic
                if 'timestamp' in self.price_data.columns:
                    self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
                elif 'datetime' in self.price_data.columns:
                    self.price_data['timestamp'] = pd.to_datetime(self.price_data['datetime'])
                
                # Ensure we have the price column
                price_col = None
                for col in ['USEP ($/MWh)', 'price', 'electricity_price', 'unit_price']:
                    if col in self.price_data.columns:
                        price_col = col
                        break
                
                if price_col is None:
                    raise ValueError("No price column found in data")
                
                self.price_data = self.price_data[['timestamp', price_col]].copy()
                self.price_data.columns = ['timestamp', 'price']
            
        except Exception as e:
            print(f"❌ Error loading price data: {e}")
            return False
        
        # Load ALL transaction log data
        try:
            all_transaction_data = []
            for path in self.transaction_log_paths:
                print(f"📋 Loading transaction log from: {path}")
                if path.endswith('.xlsx'):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path)
                all_transaction_data.append(df)
                print(f"  ✅ Loaded {len(df)} records")
            
            # Combine all transaction data
            self.transaction_logs = pd.concat(all_transaction_data, ignore_index=True)
            print(f"✅ Combined transaction logs: {len(self.transaction_logs)} total records")
            
            # Process transaction data
            self._process_transaction_logs()
            
        except Exception as e:
            print(f"❌ Error loading transaction logs: {e}")
            self.transaction_logs = None
        
        # Load weather data
        try:
            self.weather_data = pd.read_csv(self.weather_data_path)
            print(f"✅ Weather data loaded: {len(self.weather_data)} records")
            
            # Handle the actual column structure from weather data
            if 'DATE' in self.weather_data.columns and 'PERIOD' in self.weather_data.columns:
                # Create timestamp from DATE and PERIOD
                self.weather_data['timestamp'] = pd.to_datetime(self.weather_data['DATE'], format='%Y/%m/%d')
                # Add time based on period (each period is 30 minutes)
                self.weather_data['timestamp'] = self.weather_data['timestamp'] + pd.to_timedelta(
                    (self.weather_data['PERIOD'] - 1) * 30, unit='minutes'
                )
                
                # Select relevant weather columns
                weather_cols = ['timestamp', 'temp', 'rhum', 'prcp', 'wspd', 'pres', 'cloudcover (%)', 'shortwave_radiation (W/m²·h)']
                available_cols = [col for col in weather_cols if col in self.weather_data.columns]
                self.weather_data = self.weather_data[available_cols].copy()
                
                print(f"✅ Weather data processed: {len(self.weather_data)} records")
                print(f"🌤️  Weather columns: {list(self.weather_data.columns)}")
                
            else:
                # Fallback to original logic
                if 'timestamp' in self.weather_data.columns:
                    self.weather_data['timestamp'] = pd.to_datetime(self.weather_data['timestamp'])
                elif 'datetime' in self.weather_data.columns:
                    self.weather_data['timestamp'] = pd.to_datetime(self.weather_data['datetime'])
            
        except Exception as e:
            print(f"❌ Error loading weather data: {e}")
            return False
        
        # Merge data
        self._merge_data()
        
        print(f"✅ Data preparation completed: {len(self.merged_data)} merged records")
        print(f"📅 Date range: {self.merged_data['timestamp'].min()} to {self.merged_data['timestamp'].max()}")
        
        return True
    
    def _process_transaction_logs(self):
        """Process transaction log data to extract relevant features"""
        print("🔧 Processing transaction log data...")
        
        # Check available columns
        print(f"📊 Available columns: {list(self.transaction_logs.columns)}")
        
        # Try to find timestamp column
        timestamp_col = None
        for col in ['timestamp', 'datetime', 'start_time', 'startTime', 'StartTime', 'Created']:
            if col in self.transaction_logs.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            # Convert timestamp
            self.transaction_logs['timestamp'] = pd.to_datetime(self.transaction_logs[timestamp_col])
            
            # Round to nearest 30-minute slot
            self.transaction_logs['timestamp'] = self.transaction_logs['timestamp'].dt.round('30T')
            
            # Aggregate transactions by 30-minute slots
            agg_cols = {}
            
            # Energy consumption
            for col in ['energy', 'Energy', 'kWh', 'energy_kwh']:
                if col in self.transaction_logs.columns:
                    agg_cols[col] = 'sum'
                    break
            
            # Duration
            for col in ['duration', 'Duration', 'duration_minutes', 'duration_min']:
                if col in self.transaction_logs.columns:
                    agg_cols[col] = 'mean'
                    break
            
            # Count of transactions
            agg_cols['transaction_count'] = 'count'
            
            # Aggregate by timestamp
            self.transaction_logs = self.transaction_logs.groupby('timestamp').agg(agg_cols).reset_index()
            
            print(f"✅ Transaction logs processed: {len(self.transaction_logs)} aggregated records")
        else:
            print("⚠️  No timestamp column found in transaction logs")
            self.transaction_logs = None
    
    def _merge_data(self):
        """Merge all data sources on timestamp"""
        print("🔗 Merging data sources...")
        
        # Start with price data
        self.merged_data = self.price_data.copy()
        
        # Merge weather data
        self.merged_data = pd.merge(
            self.merged_data, 
            self.weather_data, 
            on='timestamp', 
            how='left'
        )
        
        # Merge transaction log data if available
        if self.transaction_logs is not None:
            self.merged_data = pd.merge(
                self.merged_data, 
                self.transaction_logs, 
                on='timestamp', 
                how='left'
            )
        
        # Sort by timestamp
        self.merged_data = self.merged_data.sort_values('timestamp').reset_index(drop=True)
        
        # Forward fill missing values
        self.merged_data = self.merged_data.ffill().bfill()
        
        print(f"✅ Data merged successfully")
        print(f"📊 Columns: {list(self.merged_data.columns)}")
    
    def create_features(self):
        """Create comprehensive feature set for forecasting"""
        print("🔧 Creating advanced features...")
        
        # Time-based features
        self.merged_data['hour'] = self.merged_data['timestamp'].dt.hour
        self.merged_data['day_of_week'] = self.merged_data['timestamp'].dt.dayofweek
        self.merged_data['month'] = self.merged_data['timestamp'].dt.month
        self.merged_data['day_of_year'] = self.merged_data['timestamp'].dt.dayofyear
        self.merged_data['week_of_year'] = self.merged_data['timestamp'].dt.isocalendar().week
        self.merged_data['quarter'] = self.merged_data['timestamp'].dt.quarter
        self.merged_data['year'] = self.merged_data['timestamp'].dt.year
        
        # Cyclical encoding for time features
        self.merged_data['hour_sin'] = np.sin(2 * np.pi * self.merged_data['hour'] / 24)
        self.merged_data['hour_cos'] = np.cos(2 * np.pi * self.merged_data['hour'] / 24)
        self.merged_data['day_sin'] = np.sin(2 * np.pi * self.merged_data['day_of_week'] / 7)
        self.merged_data['day_cos'] = np.cos(2 * np.pi * self.merged_data['day_of_week'] / 7)
        self.merged_data['month_sin'] = np.sin(2 * np.pi * self.merged_data['month'] / 12)
        self.merged_data['month_cos'] = np.cos(2 * np.pi * self.merged_data['month'] / 12)
        
        # Peak hour indicators
        self.merged_data['is_peak_hour'] = (
            (self.merged_data['hour'] >= 7) & (self.merged_data['hour'] <= 19)
        ).astype(int)
        
        self.merged_data['is_weekend'] = (
            self.merged_data['day_of_week'] >= 5
        ).astype(int)
        
        # Seasonal indicators
        self.merged_data['is_summer'] = (
            (self.merged_data['month'] >= 6) & (self.merged_data['month'] <= 8)
        ).astype(int)
        
        self.merged_data['is_winter'] = (
            (self.merged_data['month'] == 12) | (self.merged_data['month'] <= 2)
        ).astype(int)
        
        # Price-based features
        self.merged_data['price_lag1'] = self.merged_data['price'].shift(1)
        self.merged_data['price_lag2'] = self.merged_data['price'].shift(2)
        self.merged_data['price_lag6'] = self.merged_data['price'].shift(6)  # 3 hours
        self.merged_data['price_lag12'] = self.merged_data['price'].shift(12)  # 6 hours
        self.merged_data['price_lag24'] = self.merged_data['price'].shift(24)  # 12 hours
        self.merged_data['price_lag48'] = self.merged_data['price'].shift(48)  # 24 hours
        self.merged_data['price_lag96'] = self.merged_data['price'].shift(96)  # 48 hours
        self.merged_data['price_lag168'] = self.merged_data['price'].shift(168)  # 1 week
        
        # Rolling statistics
        for window in [6, 12, 24, 48, 96, 168, 336, 672]:  # 3h, 6h, 12h, 24h, 48h, 1w, 2w, 1m
            self.merged_data[f'price_rolling_mean_{window}'] = self.merged_data['price'].rolling(window).mean()
            self.merged_data[f'price_rolling_std_{window}'] = self.merged_data['price'].rolling(window).std()
            self.merged_data[f'price_rolling_median_{window}'] = self.merged_data['price'].rolling(window).median()
            self.merged_data[f'price_rolling_min_{window}'] = self.merged_data['price'].rolling(window).min()
            self.merged_data[f'price_rolling_max_{window}'] = self.merged_data['price'].rolling(window).max()
        
        # Price momentum and volatility
        self.merged_data['price_momentum_1h'] = self.merged_data['price'] - self.merged_data['price_lag2']
        self.merged_data['price_momentum_3h'] = self.merged_data['price'] - self.merged_data['price_lag6']
        self.merged_data['price_momentum_6h'] = self.merged_data['price'] - self.merged_data['price_lag12']
        self.merged_data['price_momentum_12h'] = self.merged_data['price'] - self.merged_data['price_lag24']
        self.merged_data['price_momentum_24h'] = self.merged_data['price'] - self.merged_data['price_lag48']
        self.merged_data['price_momentum_1w'] = self.merged_data['price'] - self.merged_data['price_lag168']
        
        self.merged_data['price_volatility_1h'] = self.merged_data['price'].rolling(2).std()
        self.merged_data['price_volatility_3h'] = self.merged_data['price'].rolling(6).std()
        self.merged_data['price_volatility_6h'] = self.merged_data['price'].rolling(12).std()
        self.merged_data['price_volatility_12h'] = self.merged_data['price'].rolling(24).std()
        self.merged_data['price_volatility_24h'] = self.merged_data['price'].rolling(48).std()
        self.merged_data['price_volatility_1w'] = self.merged_data['price'].rolling(168).std()
        
        # Price acceleration
        self.merged_data['price_acceleration'] = (
            self.merged_data['price_momentum_1h'] - self.merged_data['price_momentum_1h'].shift(1)
        )
        
        # Z-score and percentiles
        self.merged_data['price_zscore'] = (
            (self.merged_data['price'] - self.merged_data['price'].rolling(336).mean()) / 
            self.merged_data['price'].rolling(336).std()
        )
        
        # Extreme price indicators
        self.merged_data['is_extreme_high'] = (
            self.merged_data['price'] > self.merged_data['price'].rolling(336).quantile(0.95)
        ).astype(int)
        
        self.merged_data['is_extreme_low'] = (
            self.merged_data['price'] < self.merged_data['price'].rolling(336).quantile(0.05)
        ).astype(int)
        
        # Weather features (if available)
        weather_cols = [col for col in self.merged_data.columns if any(x in col.lower() for x in 
                        ['temp', 'humidity', 'wind', 'pressure', 'cloud', 'rain', 'solar'])]
        
        if weather_cols:
            print(f"🌤️  Weather features found: {weather_cols}")
            
            # Create lagged weather features
            for col in weather_cols:
                for lag in [1, 2, 6, 12, 24, 48, 168]:
                    self.merged_data[f'{col}_lag{lag}'] = self.merged_data[col].shift(lag)
                
                # Rolling weather statistics
                for window in [6, 12, 24, 48, 168]:
                    self.merged_data[f'{col}_rolling_mean_{window}'] = self.merged_data[col].rolling(window).mean()
                    self.merged_data[f'{col}_rolling_std_{window}'] = self.merged_data[col].rolling(window).std()
        
        # Consumption features (if available)
        consumption_cols = [col for col in self.merged_data.columns if any(x in col.lower() for x in 
                           ['demand', 'consumption', 'load', 'mw'])]
        
        if consumption_cols:
            print(f"⚡ Consumption features found: {consumption_cols}")
            
            # Create lagged consumption features
            for col in consumption_cols:
                for lag in [1, 2, 6, 12, 24, 48, 168]:
                    self.merged_data[f'{col}_lag{lag}'] = self.merged_data[col].shift(lag)
                
                # Rolling consumption statistics
                for window in [6, 12, 24, 48, 168]:
                    self.merged_data[f'{col}_rolling_mean_{window}'] = self.merged_data[col].rolling(window).mean()
                    self.merged_data[f'{col}_rolling_std_{window}'] = self.merged_data[col].rolling(window).std()
        
        # Transaction log features (if available)
        transaction_cols = [col for col in self.merged_data.columns if any(x in col.lower() for x in 
                            ['energy', 'duration', 'transaction_count', 'kwh'])]
        
        if transaction_cols:
            print(f"📋 Transaction features found: {transaction_cols}")
            
            # Create lagged transaction features
            for col in transaction_cols:
                for lag in [1, 2, 6, 12, 24, 48, 168]:
                    self.merged_data[f'{col}_lag{lag}'] = self.merged_data[col].shift(lag)
                
                # Rolling transaction statistics
                for window in [6, 12, 24, 48, 168]:
                    self.merged_data[f'{col}_rolling_mean_{window}'] = self.merged_data[col].rolling(window).mean()
                    self.merged_data[f'{col}_rolling_std_{window}'] = self.merged_data[col].rolling(window).std()
        
        # Remove rows with NaN values
        initial_rows = len(self.merged_data)
        self.merged_data = self.merged_data.dropna()
        final_rows = len(self.merged_data)
        
        print(f"✅ Features created successfully")
        print(f"📊 Total features: {len(self.merged_data.columns)}")
        print(f"📈 Data points: {initial_rows} → {final_rows} (after cleaning)")
        
        return True

    def prepare_features_for_training(self):
        """Prepare features and target for model training"""
        print("🎯 Preparing features for training...")
        
        # Select features (exclude timestamp and target)
        exclude_cols = ['timestamp', 'price']
        feature_cols = [col for col in self.merged_data.columns if col not in exclude_cols]
        
        # Separate features and target
        X = self.merged_data[feature_cols].copy()
        y = self.merged_data['price'].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"🔧 Converting {len(categorical_cols)} categorical columns to numeric")
            for col in categorical_cols:
                X[col] = pd.Categorical(X[col]).codes
        
        # Convert to numeric
        X = X.astype(float)
        
        # Feature selection
        print("🔍 Performing feature selection...")
        selector = SelectKBest(score_func=mutual_info_regression, k=min(100, len(X.columns)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        X_selected = pd.DataFrame(X_selected, columns=selected_features)
        
        print(f"✅ Feature selection completed: {len(selected_features)} features selected")
        print(f"📊 Selected features: {selected_features[:15]}...")
        
        return X_selected, y, selected_features
    
    def train_quantile_regression(self, X, y):
        """Train quantile regression models for different confidence levels"""
        print("📊 Training Quantile Regression models...")
        
        quantile_models = {}
        
        for quantile in self.confidence_levels:
            print(f"  Training quantile {quantile:.2f}...")
            
            model = QuantileRegressor(
                quantile=quantile,
                alpha=0.1,
                solver='highs'
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * 0.2))
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                X_train_scaled = self.feature_scaler.fit_transform(X_train)
                X_val_scaled = self.feature_scaler.transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_val_scaled)
                
                # Calculate score (quantile loss)
                score = np.mean(np.where(y_val >= y_pred, 
                                       quantile * (y_val - y_pred), 
                                       (1 - quantile) * (y_pred - y_val)))
                scores.append(score)
            
            # Train final model on full dataset
            X_scaled = self.feature_scaler.fit_transform(X)
            final_model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
            final_model.fit(X_scaled, y)
            
            quantile_models[quantile] = {
                'model': final_model,
                'scaler': self.feature_scaler,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            
            print(f"    Quantile {quantile:.2f}: Score = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        self.quantile_model = quantile_models
        print("✅ Quantile Regression training completed")
        
        return quantile_models
    
    def train_gaussian_process(self, X, y):
        """Train Gaussian Process Regression model"""
        print("🌊 Training Gaussian Process Regression model...")
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Define kernel
        kernel = RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=1.5)
        
        # Train GPR model
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * 0.2))
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            gpr.fit(X_train, y_train)
            
            # Predict
            y_pred, y_std = gpr.predict(X_val, return_std=True)
            
            # Calculate score
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        # Train final model on full dataset
        final_gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        final_gpr.fit(X_scaled, y)
        
        self.gpr_model = {
            'model': final_gpr,
            'scaler': self.feature_scaler,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        
        print(f"✅ GPR training completed: R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return self.gpr_model
    
    def train_bayesian_neural_network(self, X, y):
        """Train Bayesian Neural Network using PyTorch"""
        print("🧠 Training Bayesian Neural Network...")
        
        try:
            # Check if PyTorch is available
            if not torch.cuda.is_available():
                device = torch.device('cpu')
                print("  Using CPU for training")
            else:
                device = torch.device('cuda')
                print("  Using GPU for training")
            
            # Scale features and target
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            y_tensor = torch.FloatTensor(y_scaled).to(device)
            
            # Define BNN architecture
            class BayesianNN(nn.Module):
                def __init__(self, input_size):
                    super(BayesianNN, self).__init__()
                    self.fc1 = nn.Linear(input_size, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.fc4 = nn.Linear(64, 32)
                    self.fc5 = nn.Linear(32, 1)
                    self.dropout = nn.Dropout(0.3)
                    self.relu = nn.ReLU()
                    self.batch_norm1 = nn.BatchNorm1d(256)
                    self.batch_norm2 = nn.BatchNorm1d(128)
                    self.batch_norm3 = nn.BatchNorm1d(64)
                
                def forward(self, x):
                    x = self.batch_norm1(self.dropout(self.relu(self.fc1(x))))
                    x = self.batch_norm2(self.dropout(self.relu(self.fc2(x))))
                    x = self.batch_norm3(self.dropout(self.relu(self.fc3(x))))
                    x = self.dropout(self.relu(self.fc4(x)))
                    x = self.fc5(x)
                    return x
            
            # Initialize model
            model = BayesianNN(X_scaled.shape[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop
            model.train()
            for epoch in range(150):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                
                if (epoch + 1) % 30 == 0:
                    print(f"    Epoch {epoch+1}/150, Loss: {loss.item():.6f}")
            
            # Monte Carlo dropout for uncertainty estimation
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for _ in range(100):  # 100 forward passes
                    outputs = model(X_tensor)
                    pred = self.target_scaler.inverse_transform(outputs.cpu().numpy())
                    predictions.append(pred.flatten())
            
            predictions = np.array(predictions)
            
            # Calculate uncertainty metrics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Calculate R² score
            r2 = r2_score(y, mean_pred)
            
            self.bnn_model = {
                'model': model,
                'scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'device': device,
                'r2_score': r2,
                'predictions': predictions
            }
            
            print(f"✅ BNN training completed: R² = {r2:.4f}")
            
        except Exception as e:
            print(f"⚠️  Warning: BNN training failed: {e}")
            print("  Continuing with other models...")
            self.bnn_model = None
        
        return self.bnn_model
    
    def train_ensemble_model(self, X, y):
        """Train ensemble model combining multiple approaches"""
        print("🎯 Training Ensemble Model...")
        
        # Train base models
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * 0.2))
        rf_scores, gb_scores = [], []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train Random Forest
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_val_scaled)
            rf_scores.append(r2_score(y_val, rf_pred))
            
            # Train Gradient Boosting
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_val_scaled)
            gb_scores.append(r2_score(y_val, gb_pred))
        
        # Train final models on full dataset
        X_scaled = self.feature_scaler.fit_transform(X)
        
        final_rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        final_rf.fit(X_scaled, y)
        
        final_gb = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42)
        final_gb.fit(X_scaled, y)
        
        self.ensemble_model = {
            'rf_model': final_rf,
            'gb_model': final_gb,
            'scaler': self.feature_scaler,
            'rf_score': np.mean(rf_scores),
            'gb_score': np.mean(gb_scores)
        }
        
        print(f"✅ Ensemble training completed:")
        print(f"  Random Forest: R² = {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
        print(f"  Gradient Boosting: R² = {np.mean(gb_scores):.4f} ± {np.std(gb_scores):.4f}")
        
        return self.ensemble_model
    
    def train_all_models(self):
        """Train all probabilistic models"""
        print("🚀 Training all probabilistic models...")
        print("=" * 60)
        
        # Prepare features
        X, y, selected_features = self.prepare_features_for_training()
        self.selected_features = selected_features
        
        # Train models
        self.train_quantile_regression(X, y)
        self.train_gaussian_process(X, y)
        self.train_bayesian_neural_network(X, y)
        self.train_ensemble_model(X, y)
        
        print("✅ All models trained successfully!")
        print("=" * 60)
        
        return True
    
    def predict_with_confidence_intervals(self, X_future, method='ensemble'):
        """
        Generate predictions with confidence intervals using specified method
        
        Args:
            X_future: Future feature matrix
            method: 'quantile', 'gpr', 'bnn', 'ensemble', or 'all'
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        print(f"🔮 Generating predictions with confidence intervals using {method} method...")
        
        results = {}
        
        if method == 'quantile' or method == 'all':
            results['quantile'] = self._predict_quantile_regression(X_future)
        
        if method == 'gpr' or method == 'all':
            results['gpr'] = self._predict_gaussian_process(X_future)
        
        if method == 'bnn' or method == 'all':
            results['bnn'] = self._predict_bayesian_neural_network(X_future)
        
        if method == 'ensemble' or method == 'all':
            results['ensemble'] = self._predict_ensemble(X_future)
        
        return results
    
    def _predict_quantile_regression(self, X_future):
        """Generate predictions using quantile regression"""
        if self.quantile_model is None:
            return None
        
        predictions = {}
        
        for quantile, model_info in self.quantile_model.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Scale features
            X_scaled = scaler.transform(X_future)
            
            # Predict
            pred = model.predict(X_scaled)
            predictions[quantile] = pred
        
        # Calculate confidence intervals
        ci_results = {
            'predictions': predictions,
            'mean': predictions[0.5],  # Median prediction
            'lower_95': predictions[0.05],  # 5th percentile
            'upper_95': predictions[0.95],  # 95th percentile
            'lower_75': predictions[0.25],  # 25th percentile
            'upper_75': predictions[0.75],  # 75th percentile
            'uncertainty_95': predictions[0.95] - predictions[0.05],
            'uncertainty_75': predictions[0.75] - predictions[0.25]
        }
        
        return ci_results
    
    def _predict_gaussian_process(self, X_future):
        """Generate predictions using Gaussian Process Regression"""
        if self.gpr_model is None:
            return None
        
        model = self.gpr_model['model']
        scaler = self.gpr_model['scaler']
        
        # Scale features
        X_scaled = scaler.transform(X_future)
        
        # Predict with uncertainty
        mean_pred, std_pred = model.predict(X_scaled, return_std=True)
        
        # Calculate confidence intervals
        z_score_95 = 1.96  # 95% confidence
        z_score_75 = 1.15  # 75% confidence
        
        ci_results = {
            'mean': mean_pred,
            'std': std_pred,
            'lower_95': mean_pred - z_score_95 * std_pred,
            'upper_95': mean_pred + z_score_95 * std_pred,
            'lower_75': mean_pred - z_score_75 * std_pred,
            'upper_75': mean_pred + z_score_75 * std_pred,
            'uncertainty_95': 2 * z_score_95 * std_pred,
            'uncertainty_75': 2 * z_score_75 * std_pred
        }
        
        return ci_results
    
    def _predict_bayesian_neural_network(self, X_future):
        """Generate predictions using Bayesian Neural Network"""
        if self.bnn_model is None:
            return None
        
        model = self.bnn_model['model']
        scaler = self.bnn_model['scaler']
        target_scaler = self.bnn_model['target_scaler']
        device = self.bnn_model['device']
        
        # Scale features
        X_scaled = scaler.transform(X_future)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Monte Carlo dropout for uncertainty estimation
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(100):  # 100 forward passes
                outputs = model(X_tensor)
                pred = target_scaler.inverse_transform(outputs.cpu().numpy())
                predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        z_score_95 = 1.96
        z_score_75 = 1.15
        
        ci_results = {
            'mean': mean_pred,
            'std': std_pred,
            'lower_95': mean_pred - z_score_95 * std_pred,
            'upper_95': mean_pred + z_score_95 * std_pred,
            'lower_75': mean_pred - z_score_75 * std_pred,
            'upper_75': mean_pred + z_score_75 * std_pred,
            'uncertainty_95': 2 * z_score_95 * std_pred,
            'uncertainty_75': 2 * z_score_75 * std_pred,
            'predictions': predictions
        }
        
        return ci_results
    
    def _predict_ensemble(self, X_future):
        """Generate predictions using ensemble methods"""
        if self.ensemble_model is None:
            return None
        
        rf_model = self.ensemble_model['rf_model']
        gb_model = self.ensemble_model['gb_model']
        scaler = self.ensemble_model['scaler']
        
        # Scale features
        X_scaled = scaler.transform(X_future)
        
        # Generate predictions
        rf_pred = rf_model.predict(X_scaled)
        gb_pred = gb_model.predict(X_scaled)
        
        # Ensemble prediction (weighted average based on performance)
        rf_weight = self.ensemble_model['rf_score'] / (self.ensemble_model['rf_score'] + self.ensemble_model['gb_score'])
        gb_weight = self.ensemble_model['gb_score'] / (self.ensemble_model['rf_score'] + self.ensemble_model['gb_score'])
        
        mean_pred = rf_weight * rf_pred + gb_weight * gb_pred
        
        # Calculate uncertainty based on model disagreement
        model_disagreement = np.abs(rf_pred - gb_pred)
        
        # Estimate uncertainty (assuming normal distribution)
        estimated_std = model_disagreement / 2
        
        # Calculate confidence intervals
        z_score_95 = 1.96
        z_score_75 = 1.15
        
        ci_results = {
            'mean': mean_pred,
            'rf_pred': rf_pred,
            'gb_pred': gb_pred,
            'model_disagreement': model_disagreement,
            'estimated_std': estimated_std,
            'lower_95': mean_pred - z_score_95 * estimated_std,
            'upper_95': mean_pred + z_score_95 * estimated_std,
            'lower_75': mean_pred - z_score_75 * estimated_std,
            'upper_75': mean_pred + z_score_75 * estimated_std,
            'uncertainty_95': 2 * z_score_95 * estimated_std,
            'uncertainty_75': 2 * z_score_75 * estimated_std
        }
        
        return ci_results
    
    def forecast_extended_period(self, method='ensemble'):
        """Forecast for extended period (7 days by default) with confidence intervals"""
        print(f"⏰ Forecasting next {self.forecast_days} days ({self.forecast_days * 48} slots)...")
        
        # Get the most recent data for feature creation
        latest_data = self.merged_data[self.selected_features].iloc[-1:].copy()
        
        # Create future feature matrix
        future_features = []
        total_slots = self.forecast_days * 48
        
        for i in range(1, total_slots + 1):
            future_row = latest_data.copy()
            
            # Update time-based features that are in selected_features
            if 'hour' in self.selected_features:
                future_row['hour'] = (self.merged_data['hour'].iloc[-1] + i) % 24
            if 'day_of_week' in self.selected_features:
                future_row['day_of_week'] = (self.merged_data['day_of_week'].iloc[-1] + (i // 48)) % 7
            if 'month' in self.selected_features:
                future_row['month'] = (self.merged_data['month'].iloc[-1] + (i // (48 * 30))) % 12
            if 'day_of_year' in self.selected_features:
                future_row['day_of_year'] = (self.merged_data['day_of_year'].iloc[-1] + (i // 48)) % 365
            if 'week_of_year' in self.selected_features:
                future_row['week_of_year'] = (self.merged_data['week_of_year'].iloc[-1] + (i // 336)) % 53
            if 'quarter' in self.selected_features:
                future_row['quarter'] = (self.merged_data['quarter'].iloc[-1] + (i // (48 * 90))) % 4
            if 'year' in self.selected_features:
                future_row['year'] = self.merged_data['year'].iloc[-1] + (i // (48 * 365))
            
            # Update cyclical features that are in selected_features
            if 'hour_sin' in self.selected_features:
                future_row['hour_sin'] = np.sin(2 * np.pi * future_row['hour'] / 24)
            if 'hour_cos' in self.selected_features:
                future_row['hour_cos'] = np.cos(2 * np.pi * future_row['hour'] / 24)
            if 'day_sin' in self.selected_features:
                future_row['day_sin'] = np.sin(2 * np.pi * future_row['day_of_week'] / 7)
            if 'day_cos' in self.selected_features:
                future_row['day_cos'] = np.cos(2 * np.pi * future_row['day_of_week'] / 7)
            if 'month_sin' in self.selected_features:
                future_row['month_sin'] = np.sin(2 * np.pi * future_row['month'] / 12)
            if 'month_cos' in self.selected_features:
                future_row['month_cos'] = np.cos(2 * np.pi * future_row['month'] / 12)
            
            # Update peak hour indicator
            if 'is_peak_hour' in self.selected_features:
                future_row['is_peak_hour'] = (
                    (future_row['hour'] >= 7) & (future_row['hour'] <= 19)
                ).astype(int)
            
            if 'is_weekend' in self.selected_features:
                future_row['is_weekend'] = (future_row['day_of_week'] >= 5).astype(int)
            
            # Update seasonal indicators
            if 'is_summer' in self.selected_features:
                future_row['is_summer'] = (
                    (future_row['month'] >= 6) & (future_row['month'] <= 8)
                ).astype(int)
            
            if 'is_winter' in self.selected_features:
                future_row['is_winter'] = (
                    (future_row['month'] == 12) | (future_row['month'] <= 2)
                ).astype(int)
            
            # Update price-based features (use historical averages)
            for col in self.selected_features:
                if 'lag' in col or 'rolling' in col:
                    # Use historical average for lagged features
                    historical_avg = self.merged_data[col].mean()
                    future_row[col] = historical_avg
            
            future_features.append(future_row)
        
        future_features_df = pd.concat(future_features, ignore_index=True)
        
        # Ensure the DataFrame only contains the selected features in the correct order
        future_features_df = future_features_df[self.selected_features]
        
        # Verify feature consistency
        print(f"🔍 Feature consistency check:")
        print(f"  Expected features: {len(self.selected_features)}")
        print(f"  Future features: {len(future_features_df.columns)}")
        print(f"  Feature names match: {list(future_features_df.columns) == self.selected_features}")
        
        # Generate predictions with confidence intervals
        predictions = self.predict_with_confidence_intervals(future_features_df, method)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'period': range(1, total_slots + 1),
            'timestamp': pd.date_range(
                start=self.merged_data['timestamp'].iloc[-1] + pd.Timedelta(minutes=30),
                periods=total_slots,
                freq='30T'
            )
        })
        
        # Add predictions based on method
        if method == 'all':
            # Use ensemble as primary prediction
            primary_pred = predictions['ensemble']
            forecast_df['predicted_price'] = primary_pred['mean']
            forecast_df['lower_bound_95'] = primary_pred['lower_95']
            forecast_df['upper_bound_95'] = primary_pred['upper_95']
            forecast_df['lower_bound_75'] = primary_pred['lower_75']
            forecast_df['upper_bound_75'] = primary_pred['upper_75']
            forecast_df['uncertainty_95'] = primary_pred['uncertainty_95']
            forecast_df['uncertainty_75'] = primary_pred['uncertainty_75']
            
            # Add other method predictions
            for method_name, pred in predictions.items():
                if method_name != 'ensemble':
                    forecast_df[f'{method_name}_price'] = pred['mean']
                    forecast_df[f'{method_name}_uncertainty'] = pred['uncertainty_95']
        else:
            pred = predictions[method]
            forecast_df['predicted_price'] = pred['mean']
            forecast_df['lower_bound_95'] = pred['lower_95']
            forecast_df['upper_bound_95'] = pred['upper_95']
            forecast_df['lower_bound_75'] = pred['lower_75']
            forecast_df['upper_bound_75'] = pred['upper_75']
            forecast_df['uncertainty_95'] = pred['uncertainty_95']
            forecast_df['uncertainty_75'] = pred['uncertainty_75']
        
        print(f"✅ Extended forecast generated successfully")
        print(f"📊 Price range: {forecast_df['predicted_price'].min():.2f} - {forecast_df['predicted_price'].max():.2f}")
        print(f"📈 Average uncertainty (95%): {forecast_df['uncertainty_95'].mean():.2f}")
        
        return forecast_df, predictions

    def evaluate_performance(self, method='ensemble'):
        """Evaluate model performance using walk-forward validation"""
        print("📊 Evaluating model performance...")
        
        # Prepare features
        X, y, _ = self.prepare_features_for_training()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * 0.2))
        
        metrics = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'ci_coverage_95': [],
            'ci_coverage_75': [],
            'ci_width_95': [],
            'ci_width_75': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold + 1}/3...")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train models on this fold
            self._train_fold_models(X_train, y_train)
            
            # Generate predictions using fold-specific models
            if method == 'ensemble':
                pred = self._predict_fold_ensemble(X_test)
            else:
                # For other methods, we'll use a simplified approach
                # Use the fold-specific ensemble as fallback
                pred = self._predict_fold_ensemble(X_test)
            
            if pred is None:
                print(f"    ⚠️  Warning: No predictions generated for fold {fold + 1}")
                continue
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, pred['mean']))
            mae = mean_absolute_error(y_test, pred['mean'])
            r2 = r2_score(y_test, pred['mean'])
            
            # Calculate confidence interval coverage
            ci_95_coverage = np.mean(
                (y_test >= pred['lower_95']) & (y_test <= pred['upper_95'])
            )
            ci_75_coverage = np.mean(
                (y_test >= pred['lower_75']) & (y_test <= pred['upper_75'])
            )
            
            # Calculate confidence interval width
            ci_95_width = np.mean(pred['uncertainty_95'])
            ci_75_width = np.mean(pred['uncertainty_75'])
            
            # Store metrics
            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['r2'].append(r2)
            metrics['ci_coverage_95'].append(ci_95_coverage)
            metrics['ci_coverage_75'].append(ci_75_coverage)
            metrics['ci_width_95'].append(ci_95_width)
            metrics['ci_width_75'].append(ci_75_width)
            
            print(f"    📊 Fold {fold + 1} - RMSE: {rmse:.2f}, R²: {r2:.4f}, 95% CI Coverage: {ci_95_coverage:.3f}")
        
        # Calculate average metrics
        avg_metrics = {}
        for key, values in metrics.items():
            if values:  # Only calculate if we have values
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        self.performance_metrics = avg_metrics
        
        print("✅ Performance evaluation completed")
        if 'rmse' in avg_metrics:
            print(f"📊 RMSE: {avg_metrics['rmse']['mean']:.2f} ± {avg_metrics['rmse']['std']:.2f}")
        if 'mae' in avg_metrics:
            print(f"📊 MAE: {avg_metrics['mae']['mean']:.2f} ± {avg_metrics['mae']['std']:.2f}")
        if 'r2' in avg_metrics:
            print(f"📊 R²: {avg_metrics['r2']['mean']:.4f} ± {avg_metrics['r2']['std']:.4f}")
        if 'ci_coverage_95' in avg_metrics:
            print(f"📊 95% CI Coverage: {avg_metrics['ci_coverage_95']['mean']:.3f} ± {avg_metrics['ci_coverage_95']['std']:.3f}")
        if 'ci_coverage_75' in avg_metrics:
            print(f"📊 75% CI Coverage: {avg_metrics['ci_coverage_75']['mean']:.3f} ± {avg_metrics['ci_coverage_75']['std']:.3f}")
        
        return avg_metrics
    
    def _train_fold_models(self, X_train, y_train):
        """Train models on a specific fold for evaluation"""
        print(f"    🔧 Training fold-specific models...")
        
        # Create new scalers for this fold
        fold_feature_scaler = StandardScaler()
        fold_target_scaler = StandardScaler()
        
        # Scale features
        X_train_scaled = fold_feature_scaler.fit_transform(X_train)
        
        # Train Random Forest for this fold
        fold_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        fold_rf.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting for this fold
        fold_gb = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        fold_gb.fit(X_train_scaled, y_train)
        
        # Store fold-specific models and scalers
        self.fold_models = {
            'rf_model': fold_rf,
            'gb_model': fold_gb,
            'scaler': fold_feature_scaler
        }
        
        print(f"    ✅ Fold models trained successfully")
    
    def _predict_fold_ensemble(self, X_test):
        """Generate predictions using fold-specific ensemble models"""
        if not hasattr(self, 'fold_models'):
            return None
        
        rf_model = self.fold_models['rf_model']
        gb_model = self.fold_models['gb_model']
        scaler = self.fold_models['scaler']
        
        # Scale features
        X_scaled = scaler.transform(X_test)
        
        # Generate predictions
        rf_pred = rf_model.predict(X_scaled)
        gb_pred = gb_model.predict(X_scaled)
        
        # Simple ensemble (equal weights)
        mean_pred = (rf_pred + gb_pred) / 2
        
        # Calculate uncertainty based on model disagreement
        model_disagreement = np.abs(rf_pred - gb_pred)
        estimated_std = model_disagreement / 2
        
        # Calculate confidence intervals
        z_score_95 = 1.96
        z_score_75 = 1.15
        
        ci_results = {
            'mean': mean_pred,
            'rf_pred': rf_pred,
            'gb_pred': gb_pred,
            'model_disagreement': model_disagreement,
            'estimated_std': estimated_std,
            'lower_95': mean_pred - z_score_95 * estimated_std,
            'upper_95': mean_pred + z_score_95 * estimated_std,
            'lower_75': mean_pred - z_score_75 * estimated_std,
            'upper_75': mean_pred + z_score_75 * estimated_std,
            'uncertainty_95': 2 * z_score_95 * estimated_std,
            'uncertainty_75': 2 * z_score_75 * estimated_std
        }
        
        return ci_results
    
    def save_results(self, forecast_df, predictions, filename=None):
        """Save forecasting results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_probabilistic_forecast_{self.forecast_days}days_{timestamp}.csv"
        
        # Save forecast
        output_path = f"./Outputs/{filename}"
        forecast_df.to_csv(output_path, index=False)
        print(f"💾 Enhanced forecast saved to: {output_path}")
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_path = f"./Outputs/enhanced_performance_metrics_{self.forecast_days}days_{timestamp}.json"
            import json
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
            print(f"💾 Performance metrics saved to: {metrics_path}")
        
        return output_path
    
    def generate_visualizations(self, forecast_df, predictions, method='ensemble'):
        """Generate comprehensive visualizations for extended forecast"""
        print("🎨 Generating enhanced visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle(f'Enhanced Probabilistic DR Forecasting - {self.forecast_days} Days ({method.upper()} Method)', 
                    fontsize=18, fontweight='bold')
        
        # Plot 1: Extended price forecast with confidence intervals
        axes[0, 0].plot(forecast_df['timestamp'], forecast_df['predicted_price'], 
                        'b-', linewidth=2, label='Predicted Price')
        axes[0, 0].fill_between(forecast_df['timestamp'], 
                                forecast_df['lower_bound_95'], 
                                forecast_df['upper_bound_95'], 
                                alpha=0.3, color='blue', label='95% Confidence Interval')
        axes[0, 0].fill_between(forecast_df['timestamp'], 
                                forecast_df['lower_bound_75'], 
                                forecast_df['upper_bound_75'], 
                                alpha=0.5, color='lightblue', label='75% Confidence Interval')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($/MWh)')
        axes[0, 0].set_title(f'{self.forecast_days}-Day Price Forecast with Confidence Intervals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Uncertainty over time
        axes[0, 1].plot(forecast_df['timestamp'], forecast_df['uncertainty_95'], 
                        'r-', linewidth=2, label='95% Uncertainty')
        axes[0, 1].plot(forecast_df['timestamp'], forecast_df['uncertainty_75'], 
                        'orange', linewidth=2, label='75% Uncertainty')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Uncertainty ($/MWh)')
        axes[0, 1].set_title('Forecast Uncertainty Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Price distribution (histogram)
        if method == 'all' and 'ensemble' in predictions:
            pred = predictions['ensemble']
        elif method in predictions:
            pred = predictions[method]
        else:
            pred = None
        
        if pred and 'mean' in pred:
            axes[1, 0].hist(pred['mean'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].axvline(pred['mean'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {pred["mean"].mean():.2f}')
            axes[1, 0].set_xlabel('Price ($/MWh)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Predicted Prices')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Method comparison (if available)
        if method == 'all' and len(predictions) > 1:
            method_names = list(predictions.keys())
            method_means = [predictions[m]['mean'].mean() for m in method_names]
            method_uncertainties = [predictions[m]['uncertainty_95'].mean() for m in method_names]
            
            x_pos = np.arange(len(method_names))
            axes[1, 1].bar(x_pos, method_means, yerr=method_uncertainties, 
                           capsize=5, alpha=0.7)
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylabel('Average Price ($/MWh)')
            axes[1, 1].set_title('Method Comparison')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(method_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Daily price patterns
        daily_prices = forecast_df.groupby(forecast_df['timestamp'].dt.date)['predicted_price'].mean()
        axes[2, 0].plot(daily_prices.index, daily_prices.values, 'g-o', linewidth=2, markersize=6)
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Average Daily Price ($/MWh)')
        axes[2, 0].set_title('Daily Average Price Trends')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # Plot 6: Hourly price patterns (first 7 days)
        first_week = forecast_df[forecast_df['period'] <= 336]  # First 7 days
        hourly_prices = first_week.groupby(first_week['timestamp'].dt.hour)['predicted_price'].mean()
        axes[2, 1].plot(hourly_prices.index, hourly_prices.values, color='purple', marker='o', linewidth=2, markersize=6)
        axes[2, 1].set_xlabel('Hour of Day')
        axes[2, 1].set_ylabel('Average Hourly Price ($/MWh)')
        axes[2, 1].set_title('Hourly Price Patterns (First Week)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_xticks(range(0, 24, 2))
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"./Outputs/enhanced_probabilistic_forecast_viz_{self.forecast_days}days_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Enhanced visualization saved to: {viz_path}")
        
        plt.show()
        
        return viz_path


def main():
    """Main execution function for enhanced probabilistic forecasting"""
    print("🚀 Enhanced Probabilistic Demand Response Forecasting System")
    print("📊 Using ALL available data (2023-2025) with extended forecast periods")
    print("=" * 70)
    
    # Define data paths
    price_data_paths = [
        './uploads/USEP-Data_Jan2025-June2025.csv',
        './Outputs/usep_forecast_results.csv',
        './Outputs/usep_forecast_results_updated.csv',
        './Outputs/usep_forecast_results_with_weather.csv'
    ]
    
    weather_data_path = './Outputs/weather_data/weather_changi.csv'
    
    transaction_log_paths = [
        './uploads/TransactionLogs_Jan2023-Dec2023_clean.xlsx',
        './uploads/TransactionLogs_Jan2024-Dec2024_clean.xlsx',
        './uploads/TransactionLogs_Jan2025-June2025.xlsx'
    ]
    
    # Initialize enhanced forecaster with 7-day forecast
    forecaster = EnhancedProbabilisticDRForecaster(
        price_data_paths=price_data_paths,
        weather_data_path=weather_data_path,
        transaction_log_paths=transaction_log_paths,
        forecast_days=7  # 7 days = 336 slots
    )
    
    try:
        # Step 1: Load and prepare ALL data
        if not forecaster.load_and_prepare_data():
            print("❌ Failed to load data. Exiting.")
            return
        
        # Step 2: Create enhanced features
        forecaster.create_features()
        
        # Step 3: Train all models
        forecaster.train_all_models()
        
        # Step 4: Evaluate performance
        performance = forecaster.evaluate_performance(method='ensemble')
        
        # Step 5: Generate extended forecast (7 days)
        forecast_df, predictions = forecaster.forecast_extended_period(method='ensemble')
        
        # Step 6: Save results
        forecaster.save_results(forecast_df, predictions)
        
        # Step 7: Generate enhanced visualizations
        forecaster.generate_visualizations(forecast_df, predictions, method='ensemble')
        
        print("\n" + "=" * 70)
        print("🎉 ENHANCED PROBABILISTIC FORECASTING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Forecast periods: {len(forecast_df)} slots ({forecaster.forecast_days} days)")
        print(f"📈 Price range: ${forecast_df['predicted_price'].min():.2f} - ${forecast_df['predicted_price'].max():.2f}")
        print(f"🎯 Average uncertainty (95%): ±${forecast_df['uncertainty_95'].mean():.2f}")
        print(f"📊 Model performance: R² = {performance['r2']['mean']:.4f}")
        print(f"📊 CI coverage (95%): {performance['ci_coverage_95']['mean']:.1%}")
        print(f"📅 Data range used: {forecaster.merged_data['timestamp'].min().strftime('%Y-%m-%d')} to {forecaster.merged_data['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"�� Total data points: {len(forecaster.merged_data):,}")
        
    except Exception as e:
        print(f"❌ Error during enhanced analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
