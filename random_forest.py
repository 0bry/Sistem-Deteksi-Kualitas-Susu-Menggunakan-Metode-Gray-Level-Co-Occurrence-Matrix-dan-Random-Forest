import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_random_forest(): 
    data_path = "features/glcm_features_distance_4_angle_135.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight=None,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    os.makedirs('training', exist_ok=True)
    
    model_path = 'training/model.pkl'
    joblib.dump(rf_model, model_path)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    cm_path = 'training/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    summary_path = 'training/training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write(f"\n\nAccuracy: {accuracy:.4f}\n")
    
    return rf_model, accuracy, cm

if __name__ == "__main__":
    model, accuracy, confusion_matrix = train_random_forest()
    print(f"\nAccuracy: {accuracy:.4f}")
