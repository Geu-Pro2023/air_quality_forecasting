# **Beijing Air Quality Forecasting with LSTM**

## **Table of Contents**
- [Project Overview](#-project-overview)
- [Project Objectives](#-Project-Objectives)
- [Dataset](#-key-Dataset)
- [Repository Structure](#-repository-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Kaggle Submission](#-kaggle-submission)
- [Contributing](#-contributing)
- [License](#-license)
- [References](#-references)

## **Project Overview**
This project applies Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models to forecast PM2.5 air pollution levels in Beijing. Accurate predictions of these harmful particulates enable governments to issue health warnings and implement emission control strategies, ultimately protecting public health.

## **Project Objectives**
- **Validation RMSE**: 67.98 
- **Training Time**: 2.1 hours (on GPU)
- **Best Model**: 3-layer Bidirectional LSTM
- **Preprocess and analyze time-series air quality data (meteorological and pollution measurements)**
- **Design and train LSTM/RNN models to predict PM2.5 concentrations**
- **Optimize performance through hyperparameter tuning and feature engineering**
- **Compete on Kaggle to achieve a top leaderboard ranking (target RMSE: <4000)**

## **Dataset**
The dataset contains hourly air quality measurements from Beijing between 2010-2013, including:
- Meteorological features: DEWP (dew point), TEMP (temperature), PRES (pressure), IWS (wind speed)
- Pollution measurements: PM2.5 concentrations
- Categorical features: Wind direction (cbwd_NW, cbwd_SE, cbwd_cv)
Training Data: 30,676 records
Test Data: 13,148 records

## **Repository Structure**
```
Air-quality-forecasting
│── /Data
│├── train.csv
│└── test.csv
│── /notebooks
│└── air_quality_forecasting_starter_code.ipynb
│└── Beijing_Air_Quality_Forecasting.ipynb
│── README.md
│── submission.csv

```
## **Setup Instructions**
1. **Clone the Repository:**
```
git clone https://github.com/your-username/air_quality_forecasting.git
cd air_quality_forecasting
```
2. **Install Dependencies:**
```
pip install -r requirements.txt  # Includes TensorFlow, Pandas, NumPy, etc.
```
3. **Run the Jupyter Notebook:**
```
jupyter notebook air_quality_forecasting.ipynb
```
5. **Kaggle Submission:**
Format predictions as per sample_submission.csv and submit to the Kaggle competition

## **Usage Instructions**
### **Running the Pipeline**
1. **Data preprocessing:**
```python
from src.data_processing import clean_dataset
df = clean_dataset('data/raw/train.csv')
```python

2. **Feature engineering:**
```python
from src.features import create_features
df = create_features(df)
```python

3. **Model training:**
```python
from src.models import LSTMForecaster
model = LSTMForecaster()
model.train(X_train, y_train)
```python
```

## **Model Architecture**
```python
# Example model definition
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), 
                 input_shape=(24, 21)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
```python
```

## **Key Findings**
1. **Data Preprocessing:** Handling missing values and normalizing features significantly improved model performance.
2. **Model Selection:** LSTMs outperformed vanilla RNNs due to their ability to capture long-term dependencies.
3. **Regularization:** Dropout and L2 regularization mitigated overfitting.

## **Future Work**
- **Incorporate weather forecasts for better predictions**
- **Test Transformer-based models (e.g., Temporal Fusion Transformers)**
- **Implement attention mechanisms to focus on important timesteps**
- **Build an ensemble of different model architectures**
  
## **Contributors**
**Name:** Geu Aguto Garang Bior **GitHub:** Geu-Pro2023

## **References**
1. Kaggle Inc., "Assignment 1 - Time Series Forecasting May 2025," Kaggle Competition, 2025.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". Neural Computation.
3. Beijing Municipal Ecological Environment Bureau, "Beijing Air Quality Report 2010-2013," 2014.
4. World Health Organization, "WHO Global Air Quality Guidelines," 2021.

## **License**
This project is licensed under the MIT License. See LICENSE for details.
