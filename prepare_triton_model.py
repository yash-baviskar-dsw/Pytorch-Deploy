import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import boto3
import shutil

# ============================
# 1. Configuration & Credentials
# ============================
MINIO_ENDPOINT = "http://localhost:9005"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"
BUCKET_NAME = "mlflow-artifacts"

# Triton Specific Configs
MODEL_NAME = "iris-model"
TRITON_ROOT = "triton-store"  # The folder inside your bucket

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# ============================
# 2. Data & Model Definition
# ============================
# Mocking data creation if csv is missing, or load yours
if not os.path.exists("iris.csv"):
    print("⚠️ 'iris.csv' not found, downloading default...")
    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    df = pd.read_csv(url)
else:
    df = pd.read_csv("iris.csv")

X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
y = LabelEncoder().fit_transform(df["variety"].values)

x_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)

class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ============================
# 3. Training
# ============================
print("🚀 Starting Training...")
model = IrisNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
print("✅ Training Complete.")

# ============================
# 4. Convert to TorchScript (Trace) - CRITICAL FOR TRITON
# ============================
print("\n🔄 Converting to TorchScript (Tracing)...")
model.eval()
# Create dummy input: 1 sample, 4 features
dummy_input = torch.randn(1, 4)
traced_model = torch.jit.trace(model, dummy_input)

# Create a temporary local folder structure for Triton
local_repo = "temp_model_repo"
version_dir = f"{local_repo}/{MODEL_NAME}/1"
os.makedirs(version_dir, exist_ok=True)

# Save the TRACED model as 'model.pt' (Standard Triton Name)
local_model_path = f"{version_dir}/model.pt"
traced_model.save(local_model_path)
print(f"   Saved traced model to: {local_model_path}")

# ============================
# 5. Generate config.pbtxt - CRITICAL FOR TRITON
# ============================
print("\n📝 Generating config.pbtxt...")
config_content = f"""name: "{MODEL_NAME}"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 4 ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }}
]
"""

local_config_path = f"{local_repo}/{MODEL_NAME}/config.pbtxt"
with open(local_config_path, "w") as f:
    f.write(config_content)
print(f"   Saved config to: {local_config_path}")

# ============================
# 6. Upload to MinIO (S3)
# ============================
print("\n☁️  Uploading to MinIO...")

# 1. Upload model.pt to .../iris-model/1/model.pt
s3_model_key = f"{TRITON_ROOT}/{MODEL_NAME}/1/model.pt"
s3_client.upload_file(local_model_path, BUCKET_NAME, s3_model_key)
print(f"   Uploaded Model: s3://{BUCKET_NAME}/{s3_model_key}")

# 2. Upload config.pbtxt to .../iris-model/config.pbtxt
s3_config_key = f"{TRITON_ROOT}/{MODEL_NAME}/config.pbtxt"
s3_client.upload_file(local_config_path, BUCKET_NAME, s3_config_key)
print(f"   Uploaded Config: s3://{BUCKET_NAME}/{s3_config_key}")

# Cleanup local temp files
shutil.rmtree(local_repo)

print("\n🎉 DONE! Deployment Details:")
print("------------------------------------------------")
print(f"Implementation: TRITON_SERVER")
print(f"Model URI:      s3://{BUCKET_NAME}/{TRITON_ROOT}/{MODEL_NAME}")
print("------------------------------------------------")
