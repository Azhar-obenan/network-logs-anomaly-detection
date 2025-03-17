import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Load dataset
# df = pd.read_csv("/Users/mac/Documents/BD-Anomaly-detection/logs.csv")
url = "https://docs.google.com/spreadsheets/d/1LO_OvvWHuiAN3Moy2bhbiZOP9PBB0-TIF8XrNcGdCvU/export?format=csv"
df = pd.read_csv(url)
# Drop missing values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_features = ['host.ip', 'host.name', 'log.level']

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for future decoding if needed

# Define features for anomaly detection
anomaly_inputs = ['host.ip', 'host.name', 'log.level', 'log.level_num']

# Convert data to PyTorch tensor
data_tensor = torch.tensor(df[anomaly_inputs].values, dtype=torch.float32)

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

# Train Isolation Forest Model
model_IF = IsolationForestTorch()
model_IF.fit(data_tensor)

df['anomaly_scores'] = model_IF.forward(data_tensor).numpy()
df['anomaly'] = model_IF.predict(data_tensor).numpy()

# Function to test a single input
def test_single_input(single_input, model, label_encoders):
    new_df = pd.DataFrame([single_input], columns=['host.ip', 'host.name', 'log.level', 'log.level_num'])

    # Manual anomaly detection rule
    if new_df['log.level_num'][0] <= 3 or new_df['log.level'][0] in ['error', 'critical']:
        return {
            'anomaly_score': None, 
            'anomaly_label': 'Anomaly'
        }

    unseen_flag = False  

    # Encode categorical values while handling unseen values
    for col in ['host.ip', 'host.name', 'log.level']:
        if col in label_encoders:
            try:
                new_df[col] = label_encoders[col].transform([new_df[col][0]])
            except ValueError:
                unseen_flag = True

    if unseen_flag:
        return {
            'anomaly_score': None,
            'anomaly_label': 'Anomaly'
        }

    # Convert to tensor for model prediction
    input_tensor = torch.tensor(new_df.values, dtype=torch.float32)

    anomaly_score = model.forward(input_tensor)
    anomaly_label = model.predict(input_tensor)

    return {
        'anomaly_score': anomaly_score[0].item(),
        'anomaly_label': 'Anomaly' if anomaly_label[0].item() == -1 else 'Normal'
    }

# ---------- STREAMLIT UI ----------

st.title("🚀 Anomaly Detection with Isolation Forest")

# st.selectbox("Select Log Level", ["informational", "error", "notice", "critical", "Other"])
# User Inputs
host_ip = st.text_input("Enter Host IP:", "172.31.9.215")
host_name = st.selectbox("Select Router", ["Router01", "Router02", "Router03", "Other"])
# host_name = st.text_input("Enter Host Name:", "Router03")
log_level = st.selectbox("Select Log Level:", ['informational', 'error', 'notice', 'critical', 'Other'])
log_level_num = st.number_input("Enter Log Level Num:", min_value=0, max_value=10, value=3)

# Detect Anomaly Button
if st.button("Detect Anomaly"):
    input_data = [host_ip, host_name, log_level, log_level_num]
    result = test_single_input(input_data, model_IF, label_encoders)
    
    st.subheader("🔍 Detection Result")
    if result['anomaly_label'] == "Anomaly":
        st.error(f"⚠️ **Anomaly Detected!**")
    else:
        st.success(f"✅ **Normal Data**")
    
    if result['anomaly_score'] is not None:
        st.write(f"**Anomaly Score:** {result['anomaly_score']:.4f}")
