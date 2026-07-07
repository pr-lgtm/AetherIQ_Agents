import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib

def train_predictive_models(data_path='data/engineered_telemetry.csv', model_dir='models'):
    print("Initiating Machine Learning Pipeline...")
    
    # Create directory to save trained models
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the engineered dataset
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Define features (X) and drop targets and non-numeric identifiers
    features_to_drop = ['Timestamp', 'Zone_Name', 'Carbon_Emission_Tons', 'Grid_Overload_Risk']
    X = df.drop(columns=features_to_drop)
    
    # Define Targets (Y)
    y_regression = df['Carbon_Emission_Tons']
    
    # Convert probability risk into a binary classification target (1 if Risk > 0.75, else 0)
    y_classification = (df['Grid_Overload_Risk'] > 0.75).astype(int)
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_regression, y_classification, test_size=0.2, random_state=42
    )
    
    print("\nTraining Model 1: Carbon Emission Regressor (Random Forest)")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_reg_train)
    reg_predictions = regressor.predict(X_test)
    
    mse = mean_squared_error(y_reg_test, reg_predictions)
    r2 = r2_score(y_reg_test, reg_predictions)
    print(f"Regression Performance - Mean Squared Error: {mse:.4f}")
    print(f"Regression Performance - R2 Score: {r2:.4f}")
    
    print("\nTraining Model 2: Grid Overload Classifier (Random Forest)")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_clf_train)
    clf_predictions = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_clf_test, clf_predictions)
    print(f"Classification Performance - Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_clf_test, clf_predictions))
    
    # Export models to disk for the app to use
    joblib.dump(regressor, os.path.join(model_dir, 'rf_regressor.pkl'))
    joblib.dump(classifier, os.path.join(model_dir, 'rf_classifier.pkl'))
    joblib.dump(list(X.columns), os.path.join(model_dir, 'model_features.pkl'))
    
    print(f"\nModels successfully trained and saved to the '{model_dir}' directory.")

if __name__ == "__main__":
    train_predictive_models()