# Beijing Air Quality Forecasting

## Project Overview  
This project is part of the Machine Learning Techniques I course and focuses on forecasting PM2.5 air pollution levels in Beijing using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models. Accurate predictions of PM2.5 concentrations enable policymakers and communities to take timely action to mitigate the adverse effects of air pollution.  

### Objectives  
- Preprocess sequential air quality and meteorological data  
- Design and train RNN/LSTM models to forecast PM2.5 levels  
- Fine-tune models through systematic experimentation  
- Achieve RMSE < 4000 on Kaggle leaderboard  

## Best Performing Model
After extensive experimentation with various architectures, the following LSTM model demonstrated the best performance in terms of RMSE while maintaining generalization:
### **Architecture**:
**1. LSTM Layers:**
```python
model = Sequential([
    LSTM(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(n_steps, n_features)),
    Dropout(0.2),
    LSTM(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])
```
**2. Dense Layers:**
```
- Fully connected layer with 32 units, ReLU activation, and L2 regularization.
- Output layer with 1 unit for PM2.5 prediction.
```

### **Training Configuration **
Optimizer: Adam (learning rate = 0.01).

Loss Function: Mean Squared Error (MSE).

Evaluation Metric: Root Mean Squared Error (RMSE).

Performance: Achieved an RMSE of 70.44 on the validation set.

Training: 20 epochs with a batch size of 32.

## **Repository Structure**
```
├── data/                  # Dataset files (train.csv, test.csv, sample_submission.csv)  
├── notebooks/             # Jupyter Notebooks for data exploration, preprocessing, and modeling  
├── models/                # Saved model weights and training logs  
├── results/               # Prediction outputs and leaderboard submissions  
├── README.md              # Project documentation  
└── report/                # Final report (PDF or Markdown)  
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
- Execute notebooks/air_quality_forecasting.ipynb to preprocess data, train the model, and generate predictions.
4. **Kaggle Submission:**
Format predictions as per sample_submission.csv and submit to the Kaggle competition

## **Key Findings**
i. **Data Preprocessing:** Handling missing values and normalizing features significantly improved model performance.

ii. **Model Selection:** LSTMs outperformed vanilla RNNs due to their ability to capture long-term dependencies.

iii. **Regularization:** Dropout and L2 regularization mitigated overfitting.
  
## **Contributors**
Name: Geu Aguto Garang Bior
GitHub: Geu-Pro2023
Open to collaborations! Fork the repository and submit pull requests.

## **License**
This project is licensed under the MIT License. See LICENSE for details.
