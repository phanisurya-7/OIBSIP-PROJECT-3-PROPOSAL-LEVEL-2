import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Show the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Feature and target variables
X = data.drop(['Class'], axis=1)  # Drop 'Class' column to keep features
y = data['Class']  # 'Class' column is the target

# Standardize the feature data (important for models like logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Resampled data shape: {X_train_res.shape}, {y_train_res.shape}")

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model with the resampled data
model.fit(X_train_res, y_train_res)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ROC AUC score for evaluating classifier performance
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Score: {roc_auc}")

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
