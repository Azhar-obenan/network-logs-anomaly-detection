import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv("/Users/mac/Documents/BD-Anomaly-detection/network-logs-anomaly-detection/logs.csv")
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_features = ['host.ip', 'host.name', 'log.level']

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical data
scaler = MinMaxScaler()
df[['log.level_num']] = scaler.fit_transform(df[['log.level_num']])

# Define features for anomaly detection
anomaly_inputs = ['host.ip', 'host.name', 'log.level', 'log.level_num']
X_train = torch.tensor(df[anomaly_inputs].values, dtype=torch.float32)

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model and save the best weights
best_loss = float('inf')
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "best_model.pth")

# Load best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Compute reconstruction error for anomaly detection
with torch.no_grad():
    reconstructions = model(X_train)
    reconstruction_errors = torch.mean((X_train - reconstructions) ** 2, dim=1)

# Set threshold for anomalies
threshold = torch.quantile(reconstruction_errors, 0.90).item()

# Function to test single input
def test_single_input(single_input, model, label_encoders, scaler, threshold):
    new_df = pd.DataFrame([single_input], columns=['host.ip', 'host.name', 'log.level', 'log.level_num'])

    unseen_flag = False

    # Encode categorical values
    for col in ['host.ip', 'host.name', 'log.level']:
        if col in label_encoders:
            try:
                new_df[col] = label_encoders[col].transform([new_df[col][0]])
            except ValueError:
                unseen_flag = True

    if unseen_flag:
        return {'anomaly_score': None, 'anomaly_label': 'Anomaly'}

    # Normalize numerical data
    new_df[['log.level_num']] = scaler.transform(new_df[['log.level_num']])
    new_tensor = torch.tensor(new_df.values, dtype=torch.float32)

    # Compute anomaly score
    with torch.no_grad():
        reconstructed = model(new_tensor)
        reconstruction_error = torch.mean((new_tensor - reconstructed) ** 2, dim=1).item()

    anomaly_label = "Anomaly" if reconstruction_error > threshold else "Normal"
    return {'anomaly_score': reconstruction_error, 'anomaly_label': anomaly_label}

# -------------------- Streamlit UI --------------------

st.title("🔍 Anomaly Detection System")

# Dropdowns for input selection
host_ip = st.selectbox("Select Host IP", ["172.31.9.215", "Other"])
router = st.selectbox("Select Router", ["Router01", "Router02", "Router03", "Other"])
log_level = st.selectbox("Select Log Level", ["informational", "error", "notice", "critical", "Other"])
log_number = st.selectbox("Select Log Number", [1, 2, 3, 4, 5, 6])

# Button to test input
if st.button("🚀 Detect Anomaly"):
    input_data = [host_ip, router, log_level, log_number]
    result = test_single_input(input_data, model, label_encoders, scaler, threshold)

    st.subheader("📊 Prediction Result")
    st.write(f"✅ **Anomaly Score:** {result['anomaly_score']}")
    st.write(f"🚨 **Anomaly Label:** {result['anomaly_label']}")
