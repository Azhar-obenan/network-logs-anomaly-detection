import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import os
import joblib  # For saving and loading the model
import gdown  # For downloading Google Sheets as CSV
import re  # For extracting file IDs from URLs
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Function to extract file ID from a Google Drive link
def extract_drive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def download_from_gdrive(file_id, output_path):
    if file_id:
        gdrive_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
        gdown.download(gdrive_url, output_path, quiet=False)

# Define PyTorch Isolation Forest Model
class IsolationForestTorch(nn.Module):
    def __init__(self, n_estimators=100, contamination=0.1):
        super(IsolationForestTorch, self).__init__()
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)

    def forward(self, x):
        return torch.tensor(self.model.decision_function(x), dtype=torch.float32)

    def fit(self, x):
        self.model.fit(x.numpy())

    def predict(self, x):
        return torch.tensor(self.model.predict(x), dtype=torch.int8)

# Function to test a batch of inputs from a CSV file
def test_from_csv(test_file, model, label_encoders):
    test_df = pd.read_csv(test_file)
    
    if test_df.empty:
        st.error("Test file is empty.")
        return None

    test_df['label'] = 0  # Initialize label column

    for index, row in test_df.iterrows():
        single_input = [row['host.ip'], row['host.name'], row['log.level'], row['log.level_num']]
        result = test_single_input(single_input, model, label_encoders)
        
        # Assign label based on anomaly detection
        test_df.at[index, 'label'] = -1 if result['anomaly_label'] == 'Anomaly' else 1

    # Save the updated test file with labels
    output_file = "test_data_with_labels.csv"
    test_df.to_csv(output_file, index=False)
    return output_file

# Function to test a single input against the trained model
def test_single_input(single_input, model, label_encoders):
    new_df = pd.DataFrame([single_input], columns=['host.ip', 'host.name', 'log.level', 'log.level_num'])

    # Handle critical log levels separately
    if new_df['log.level_num'][0] <= 3 or new_df['log.level'][0] in ['error', 'critical']:
        return {'anomaly_score': None, 'anomaly_label': 'Anomaly'}

    unseen_flag = False  

    # Encode categorical features
    for col in ['host.ip', 'host.name', 'log.level']:
        if col in label_encoders:
            try:
                new_df[col] = label_encoders[col].transform([new_df[col][0]])
            except ValueError:
                unseen_flag = True

    if unseen_flag:
        return {'anomaly_score': None, 'anomaly_label': 'Anomaly'}

    input_tensor = torch.tensor(new_df.values, dtype=torch.float32)

    # Get anomaly score and label
    anomaly_score = model.forward(input_tensor)
    anomaly_label = model.predict(input_tensor)

    return {
        'anomaly_score': anomaly_score[0].item(),
        'anomaly_label': 'Anomaly' if anomaly_label[0].item() == -1 else 'Normal'
    }

# Streamlit UI
st.title("🚀 Anomaly Detection with Isolation Forest")
st.write("Upload Google Drive links for Train & Test CSV files")

# User input for Google Sheets links
train_link = st.text_input("Enter the Google Drive link for the TRAIN file:", "")
test_link = st.text_input("Enter the Google Drive link for the TEST file:", "")

if st.button("Run Model"):
    if not train_link or not test_link:
        st.error("Please enter both train and test file links.")
    else:
        train_file_id = extract_drive_id(train_link)
        test_file_id = extract_drive_id(test_link)

        if not train_file_id or not test_file_id:
            st.error("Invalid Google Drive link. Please check the format.")
        else:
            st.info("Downloading files...")

            # Download training data
            train_csv_path = "logs.csv"
            download_from_gdrive(train_file_id, train_csv_path)

            # Load dataset
            df = pd.read_csv(train_csv_path).dropna()

            # Encode categorical variables
            label_encoders = {}
            categorical_features = ['host.ip', 'host.name', 'log.level']

            for col in categorical_features:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le  # Save encoder for future decoding

            # Define features for anomaly detection
            anomaly_inputs = ['host.ip', 'host.name', 'log.level', 'log.level_num']

            # Convert data to PyTorch tensor
            data_tensor = torch.tensor(df[anomaly_inputs].values, dtype=torch.float32)

            # Check if the model weights file exists
            model_path = "isolation_forest_model.pth"
            if os.path.exists(model_path):
                st.success("Loading existing model weights...")
                model_IF = IsolationForestTorch()
                model_IF.model = joblib.load(model_path)
            else:
                st.success("Training new model and saving weights...")
                model_IF = IsolationForestTorch()
                model_IF.fit(data_tensor)
                joblib.dump(model_IF.model, model_path)

            # Apply anomaly detection
            df['anomaly_scores'] = model_IF.forward(data_tensor).numpy()
            df['anomaly'] = model_IF.predict(data_tensor).numpy()

            # Download test data
            test_csv_path = "test_logs.csv"
            download_from_gdrive(test_file_id, test_csv_path)

            # Run the test
            result_file = test_from_csv(test_csv_path, model_IF, label_encoders)

            if result_file:
                st.success("Processing complete! Click below to download the results.")

                # Provide download link for result file
                with open(result_file, "rb") as file:
                    st.download_button(
                        label="📥 Download Test Results",
                        data=file,
                        file_name="test_data_with_labels.csv",
                        mime="text/csv"
                    )
