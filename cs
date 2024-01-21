import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate synthetic credit score data for illustration purposes
np.random.seed(42)
data_size = 1000

# Features
age = np.random.randint(18, 70, data_size)
income = np.random.uniform(20000, 100000, data_size)
loan_amount = np.random.uniform(5000, 50000, data_size)
previous_delinquency = np.random.choice([0, 1], size=data_size, p=[0.9, 0.1])

# Target variable (1: Approved, 0: Denied)
approval_status = np.random.choice([1, 0], size=data_size, p=[0.7, 0.3])

# Create DataFrame
credit_data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'LoanAmount': loan_amount,
    'PreviousDelinquency': previous_delinquency,
    'ApprovalStatus': approval_status
})

# Display the first few rows of the dataset
print(credit_data.head())

# Separate features and target variable
X = credit_data.drop('ApprovalStatus', axis=1)  # Features
y = credit_data['ApprovalStatus']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Keras model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Display evaluation metrics for Keras model
accuracy = accuracy_score(y_test, y_pred)
print("Keras Model Accuracy:", accuracy)
print("\nKeras Model Classification Report:\n", classification_report(y_test, y_pred))
print("\nKeras Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
