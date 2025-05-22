# Beijing Air Quality Forecasting

## Project Overview  
This project is part of the Machine Learning Techniques I course and focuses on forecasting PM2.5 air pollution levels in Beijing using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models. Accurate predictions of PM2.5 concentrations enable policymakers and communities to take timely action to mitigate the adverse effects of air pollution.  

### Objectives  
- Preprocess sequential air quality and meteorological data  
- Design and train RNN/LSTM models to forecast PM2.5 levels  
- Fine-tune models through systematic experimentation  
- Achieve RMSE < 4000 on Kaggle leaderboard  

## Best Performing Model  
**Architecture**:  
```python
model = Sequential([
    LSTM(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(n_steps, n_features)),
    Dropout(0.2),
    LSTM(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])

## 
