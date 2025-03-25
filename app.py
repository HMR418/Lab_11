# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks

# -------------------------------
# 1. Data Exploration and Preprocessing
# -------------------------------

# Load the dataset from the Excel file
file_path = r"C:\Users\hayde\OneDrive\Computer Applications\lab_11_bridge_data.xlsx"
data = pd.read_excel(file_path)

# Explore the data
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values
# (For demonstration, we fill numerical missing values with the mean and categorical with the mode)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Assuming 'Max_Load_Tons' is the target variable, separate features and target
target = 'Max_Load_Tons'
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical features
cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(exclude=['object']).columns.tolist()
print("\nCategorical Features:", cat_features)
print("Numerical Features:", num_features)

# Create preprocessing pipelines for numeric and categorical features
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Combine the pipelines into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# -------------------------------
# 2. Splitting the Data
# -------------------------------
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. Model Development: Build the ANN
# -------------------------------

def build_model(input_shape):
    model = tf.keras.Sequential([
        # First hidden layer with L2 regularization and dropout
        layers.Dense(64, activation='relu', input_shape=(input_shape,),
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        # Second hidden layer
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        # Output layer for regression
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

input_shape = X_train.shape[1]
model = build_model(input_shape)
model.summary()

# -------------------------------
# 4. Training and Evaluation
# -------------------------------

# Define early stopping callback: stop training when the validation loss does not improve for 10 epochs
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model (using a validation split from the training data)
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stop])

# Plot training and validation loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# -------------------------------
# 5. Save the Model for Deployment
# -------------------------------
model.save("tf_bridge_model.h5")
print("Model saved as tf_bridge_model.h5")
