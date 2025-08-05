# V2G EDA: Vehicle-to-Grid Exploratory Data Analysis

Comprehensive exploratory data analysis and demand response modeling for Vehicle-to-Grid (V2G) optimizations, focusing on EV charging patterns, user behavior analysis, and grid integration strategies.

## 📋 Project Overview

This project analyzes EV charging transaction data from January-June 2025 to understand charging patterns, identify optimization opportunities, and develop demand response strategies for Vehicle-to-Grid systems. The analysis covers 196,580+ charging transactions across multiple zones and charging sites.

### Key Features

- **Comprehensive EDA**: Time-series analysis, temporal patterns, and usage trends
- **Customer Segmentation**: User behavior clustering and persona identification
- **Demand Response Strategy**: Peak-hour throttling candidate identification
- **Grid Integration Analysis**: Site utilization, zone-based load distribution
- **Revenue Modeling**: V2G discharge potential and dynamic pricing analysis
- **Retention Analysis**: Customer loyalty and repeat usage patterns

### Key Findings

- **Temporal Patterns**: Peak demand occurs on weekday evenings (18:00-22:00) with weekend demand more evenly distributed
- **Geographic Distribution**: West and Central zones dominate both session counts and energy delivery
- **User Behavior**: High session duration variability with median ~2 hours but long tail extending to multi-day charging
- **Seasonal Trends**: Increasing demand trend from January through June, particularly strong in May-June
- **Customer Retention**: Only ~8% of customers used charging services all 6 months, suggesting need for improved loyalty programs

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

Place your data files in the `uploads/` directory:

- `TransactionLogs_Jan2025-June2025.xlsx` - Main transaction data
- `transaction_data_tagged.csv` - Tagged transaction data
- `USEP data/` - Directory containing monthly USEP pricing data (optional)

## 🚀 Usage

### Running the Analysis

1. **Open and run the main analysis**:

   - `EDA.ipynb` - Complete exploratory data analysis
   - `Demand Response Modelling.ipynb` - Demand response strategies (in development)

### Key Notebooks

- **`EDA.ipynb`**: Main analysis notebook containing:

  - Data cleaning and preprocessing
  - Temporal pattern analysis (daily, weekly, monthly trends)
  - Customer behavior clustering and segmentation
  - Site utilization and zone analysis
  - Peak-hour throttling candidate identification
  - Revenue potential modeling
- **`Demand Response Modelling.ipynb`**: Specialized notebook for:

  - Demand response strategy development
  - Grid integration modeling
  - Dynamic pricing correlation analysis

### Output Files

Generated visualizations are saved to `Outputs/`:

- `customer_clustering.png` - Customer segmentation analysis
- `customer_correlation_matrix.png` - Feature correlation heatmap
- `evse_utilization_top10.png` - Top charger utilization rates
- `zone_hourly_heatmap.png` - Hourly energy consumption by zone
- `peak_user_*.png` - Peak hour usage trend analysis
- `repeat_customer_retention.png` - Customer retention analysis

## 📊 Analysis Components

### 1. Data Preprocessing

- Cleaning 196,580+ transaction records
- Feature engineering (temporal features, duration calculations)
- Missing data handling and outlier detection

### 2. Temporal Analysis

- Daily/weekly/monthly energy consumption trends
- Hour-of-day and day-of-week usage patterns
- Seasonal trend identification

### 3. Customer Segmentation

- K-means clustering based on usage patterns
- Customer loyalty ratio calculation
- Repeat usage and retention analysis

### 4. Infrastructure Analysis

- EVSE (charger gun) utilization rates
- Site-level performance metrics
- Zone-based load distribution

### 5. Demand Response Strategy

- Peak-hour throttling candidate identification
- Revenue potential from V2G discharge
- Dynamic pricing correlation analysis

## 🎯 Key Insights for V2G Optimization

1. **Load Shifting Opportunities**: Peak demand concentration in evening hours presents clear opportunities for demand response programs
2. **Customer Segmentation**: Distinct user patterns enable targeted incentive programs for different customer personas
3. **Infrastructure Focus**: West and Central zones show highest utilization, suggesting priority areas for V2G infrastructure deployment
4. **Retention Strategy**: Low customer retention rates indicate need for improved loyalty programs to ensure V2G participation

## 📈 Future Development

- Enhanced demand response modeling
- Real-time pricing integration
- Predictive modeling for load forecasting
- V2G revenue optimization algorithms
