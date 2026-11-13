# Rainfall Prediction Using Machine Learning

## Project Overview

This project focuses on predicting rainfall occurrence using historical meteorological data and Machine Learning techniques. A Random Forest Classifier is used to classify whether rainfall will occur based on weather-related features.

The project includes data preprocessing, feature engineering, model training, performance evaluation, and model serialization.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask (if implemented)

## Key Features

- Data cleaning and preprocessing.
- Missing value handling using median and mode imputation.
- Categorical feature encoding using Label Encoding.
- Numerical feature scaling using StandardScaler.
- Rainfall classification using Random Forest.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
- Feature importance analysis.
- Model and preprocessing artifact serialization using Joblib.

## Machine Learning Model

**Algorithm:** Random Forest Classifier

| Hyperparameter | Value |
|---|---:|
| Number of Estimators | 100 |
| Maximum Depth | 12 |
| Minimum Samples per Leaf | 5 |
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Class Weight | Balanced |

## Model Performance

The model was evaluated on 142,557 unseen test records.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | **83.09%** |
| Rainfall Precision | 81% |
| Rainfall Recall | 85% |
| Rainfall F1-Score | 0.83 |
| No Rainfall Precision | 85% |
| No Rainfall Recall | 81% |
| No Rainfall F1-Score | 0.83 |

### Confusion Matrix

| Actual / Predicted | No Rainfall | Rainfall |
|---|---:|---:|
| No Rainfall | 59,274 | 13,974 |
| Rainfall | 10,132 | 59,177 |

## Feature Importance

The model identified the following features as important contributors to rainfall prediction:

| Feature | Importance |
|---|---:|
| Season | 20.6% |
| Air Pressure | 18.2% |
| Maximum Temperature | 14.6% |

## Project Structure

```text
Rainfall-Prediction/
│
├── data/
│   └── rainfall_dataset.csv
│
├── models/
│   ├── rainfall_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── metadata.json
│
├── scripts/
│   └── train_model.py
│
├── preprocess.py
├── app.py
└── README.md
```

*Note: Update the directory structure according to the actual project files.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Rainfall-Prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib flask
```

### 3. Train the Model

```bash
python scripts/train_model.py
```

### 4. Run the Application

If the Flask application is implemented:

```bash
python app.py
```

## Saved Model Artifacts

The training script generates the following files inside the `models/` directory:

- `rainfall_model.pkl` – Trained Random Forest classification model.
- `encoders.pkl` – Saved categorical encoders.
- `scaler.pkl` – Fitted numerical feature scaler.
- `metadata.json` – Feature information, numerical ranges, categorical options, and model accuracy.

## Project Objective

To develop a machine learning-based rainfall classification system that processes meteorological data and predicts rainfall occurrence while evaluating model performance through standard classification metrics.

## Author

**NAVYA KUNWAR**

- GitHub: https://github.com/navyas2006
