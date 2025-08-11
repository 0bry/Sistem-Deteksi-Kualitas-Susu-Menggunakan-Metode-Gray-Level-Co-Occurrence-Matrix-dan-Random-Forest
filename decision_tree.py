import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

df = pd.read_csv("features/glcm_features_distance_1_angle_0.csv")
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy: {accuracy:.4f}")

os.makedirs('training', exist_ok=True)
joblib.dump(dt_model, 'training/decision_tree_model.pkl')

with open('training/decision_tree_structure.txt', 'w') as f:
    f.write(export_text(dt_model, feature_names=list(X.columns)))

print("\nModel saved to: training/decision_tree_model.pkl")
print("Tree structure saved to: training/decision_tree_structure.txt")
