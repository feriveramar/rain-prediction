import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump
import pathlib

# Load the dataset
df = pd.read_csv(pathlib.Path('data/usa_rain_prediction_dataset_2024_2025.csv'))

# Features and target variable
y = df.pop('Rain Tomorrow')  # Target variable
X = df.drop(columns=['Date', 'Location'])  # Features, excluding non-numeric columns

# Create a pipeline with imputer and model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
dump(pipeline, pathlib.Path('model/rain-prediction-v1.joblib'))
print("Model training complete and saved.")