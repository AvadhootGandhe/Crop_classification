import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
import joblib

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Check for missing values
missing_values_count = df.isnull().sum()
print("Missing values per column:")
print(missing_values_count)

# Check for duplicate rows
duplicate_rows_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows_count}")
df = df.drop_duplicates()

# Check data types
print("\nData types of columns:")
print(df.dtypes)

# Get list of numerical columns
column_list = df.columns.tolist()
column_list.remove('label')
print("\nNumerical columns:", column_list)

# Calculate Z-scores for outlier detection
for column in column_list:
    df[f'Z score of {column}'] = zscore(df[column])

# Calculate absolute Z-scores
for column in column_list:
    df[f'Z score of {column}'] = df[f'Z score of {column}'].abs()

# Identify rows with outliers (Z-score > 3)
outlier_mask = pd.Series(False, index=df.index)
for column in column_list:
    outlier_mask = outlier_mask | (df[f'Z score of {column}'] > 3)

# Remove outlier rows
df = df[~outlier_mask]
print("\nShape of DataFrame after outlier removal:", df.shape)

# Prepare the data for training
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create and train the model
model = XGBClassifier(
    objective='multi:softmax',
    num_class=22,
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Print model performance
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save the model and label encoder
joblib.dump(model, "model.joblib")
joblib.dump(le, "label_encoder.joblib")

print("\nModel and label encoder have been saved successfully!")

# Print the valid ranges for each feature
print("\nValid ranges for each feature:")
for column in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    min_val = df[column].min()
    max_val = df[column].max()
    print(f"{column}: {min_val:.2f} to {max_val:.2f}") 