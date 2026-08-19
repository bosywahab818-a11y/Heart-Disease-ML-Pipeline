# Heart Disease Risk Prediction

## Project Overview

Complete Machine Learning pipeline for heart disease prediction using the UCI Heart Disease dataset.

## Dataset

- Samples: 303
- Features: 13
- Target: target

## Pipeline

1. Data preprocessing and cleaning
2. Exploratory Data Analysis
3. PCA dimensionality reduction
4. Feature selection
5. Supervised learning
6. Unsupervised learning
7. Hyperparameter tuning
8. Model evaluation
9. Model export
10. Streamlit deployment
11. Ngrok deployment

## Preprocessing

The preprocessing pipeline handles missing values, categorical features, numerical features, and feature scaling.

## PCA

PCA was applied to the processed feature space. The original processed feature space contained 28 features.

## Feature Selection

Feature selection techniques used:
- Random Forest Feature Importance
- Recursive Feature Elimination (RFE)
- Chi-Square Test

## Supervised Learning

Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Unsupervised Learning

Clustering techniques:
- K-Means
- Hierarchical Clustering

## Hyperparameter Tuning

GridSearchCV and RandomizedSearchCV were used to optimize model hyperparameters.

## Final Model

The final trained model is saved as:

models/final_model.pkl

The best-performing model in the final evaluation was Logistic Regression.

## Streamlit Application

The Streamlit application is located at:

ui/app.py

The application provides:
- Patient information input
- Heart disease prediction
- Prediction probabilities
- Input data preview
- Age distribution
- Disease distribution
- Cholesterol vs Age visualization

## Running the Application

Install dependencies:

pip install -r requirements.txt

Run Streamlit:

streamlit run ui/app.py

The application runs on port 8501.

## Ngrok

Ngrok was used to expose the Streamlit application publicly.

Example:

ngrok http 8501

## Project Structure

Heart_Disease_Project/
├── data/
│   └── heart_disease.csv
├── models/
│   └── final_model.pkl
├── ui/
│   └── app.py
├── results/
│   ├── final_evaluation.csv
│   ├── evaluation_metrics.txt
│   └── figures/
│       └── final_model_comparison.png
├── README.md
├── requirements.txt
└── .gitignore

## Results

Final evaluation results are available in results/final_evaluation.csv.

Evaluation summary is available in results/evaluation_metrics.txt.

The final model comparison visualization is available in results/figures/final_model_comparison.png.

## Limitations

- The dataset contains only 303 samples.
- The model may not generalize to other populations.
- Predictions should not be interpreted as medical diagnoses.
- The application is intended for educational purposes.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Ngrok

## Disclaimer

This project is for educational and machine learning demonstration purposes only and is not intended to replace professional medical advice, diagnosis, or treatment.