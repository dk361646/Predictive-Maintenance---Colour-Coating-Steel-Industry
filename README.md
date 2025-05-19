
# Predictive Maintenance Using LSTM

This project develops a deep learning-based Predictive Maintenance (PdM) model using LSTM (Long Short-Term Memory) networks. It predicts whether a motor will fail in the next 24 hours (classification) and estimates its Remaining Useful Life (RUL) in hours (regression), based on historical sensor and process data.

---

## Project Overview

- **Objective**: Predict machine failure and estimate RUL for motors in a color coating line.
- **Approach**: Time-series modeling using LSTM with dual outputs.
- **Outputs**:
  - **Failure Probability** (Binary Classification)
  - **Remaining Useful Life** (Regression)

---

## Dataset Details

The dataset is synthetically generated and structured as follows:

- **Timestamp** (hourly intervals for 10,000 hours per motor)
- **Motor ID** (5 motors)
- **Temperature**
- **Vibration**
- **Current**
- **Line Speed** (47–150 m/min)
- **Coil Thickness** (0.28–2.0 mm)
- **Failure** (binary label, at least 30 per motor)
- **RUL** (Remaining Useful Life in hours)

---

## Model Architecture

- Input shape: `(24, 5)` — sequences of 24 timesteps with 5 features
- Shared LSTM layers
- Dual output heads:
  - **Failure head**: Dense(1, activation='sigmoid')
  - **RUL head**: Dense(1, activation='linear')
- Losses:
  - Binary Crossentropy for failure prediction
  - Mean Squared Error (MSE) for RUL
- Optimizer: Adam

---

## Workflow

1. **Data Generation**: Create realistic time-series data with degradation patterns and noise.
2. **Preprocessing**:
   - Per-motor data splitting (65% train / 35% test)
   - Feature scaling using `MinMaxScaler`
   - Sequence generation for LSTM input
3. **Model Training**:
   - Multi-output LSTM model
   - Sample weighting to handle class imbalance
4. **Evaluation**:
   - Classification metrics: Accuracy, Precision, Recall, Confusion Matrix
   - Regression metrics: MAE, RMSE
5. **User Inference**:
   - Query by motor ID to return:
     - Will it fail in next 24 hours?
     - Its predicted RUL
     - 
---

## Evaluation Example

```text
Accuracy: 0.9737
Confusion Matrix:
[[84950   775]
 [ 1525   130]]
Precision: 0.1436
Recall: 0.0785
F1-Score: 0.1016
```
---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn
- Matplotlib / Seaborn (for visualization)

---

## 📄 License

This project is for educational and research purposes only.
