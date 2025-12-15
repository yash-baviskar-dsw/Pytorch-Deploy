

# **Documentation: PyTorch Deployment with Seldon Core & Triton Server**

## **1. Overview**

This project establishes a high-performance inference pipeline on Kubernetes. Instead of wrapping Python scripts in custom Docker containers, we utilize the **NVIDIA Triton Inference Server**, a pre-packaged C++ backend optimized for GPU/CPU inference. Seldon Core orchestrates the deployment, while MinIO serves as the artifact store.

**Tech Stack:**

  * **Model Framework:** PyTorch (TorchScript)
  * **Inference Server:** NVIDIA Triton Inference Server (v23.08)
  * **Orchestrator:** Seldon Core v1 (Kubernetes Operator)
  * **Protocol:** KServe V2 Open Inference Protocol
  * **Storage:** MinIO (S3 Compatible)

-----

## **2. Architecture**

1.  **Model Training:** The Data Scientist trains a model and converts it to a serialized graph (**TorchScript**).
2.  **Artifact Storage:** The model and a configuration file (`config.pbtxt`) are uploaded to MinIO in a strict **Model Repository** structure.
3.  **Deployment:** Seldon Core pulls the artifacts from MinIO into a Pod running the Triton container.
4.  **Inference:** A client sends a standard **KServe V2** JSON request. Seldon routes this to Triton, which executes the graph and returns predictions.

-----

## **3. Phase 1: Model Preparation**

Triton is a C++ engine. It cannot directly load a Python `state_dict` or execute Python classes (`class Net(nn.Module)`). The model must be **Traced** into an intermediate representation called **TorchScript**.

### **Step 3.1: Training & Tracing Script**

This script trains a simple Iris classifier and exports it for Triton.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Define Model Architecture
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3) # 3 Output Classes

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ... [Training Logic Omitted for Brevity] ...

# 2. Conversion to TorchScript (CRITICAL STEP)
model.eval()
dummy_input = torch.randn(1, 4) # Match input shape (Batch=1, Features=4)

# Trace: Records the mathematical operations into a graph
traced_model = torch.jit.trace(model, dummy_input)

# Save: Must be named 'model.pt' for Triton
traced_model.save("model.pt")
print("✅ Model converted to TorchScript")
```

-----

## **4. Phase 2: The Model Repository (Storage)**

Triton does not load single files. It loads **Repositories**. You must adhere to a strict directory hierarchy, or the server will crash on startup.

### **Step 4.1: Directory Structure**

On your local machine (before uploading) or in your S3 bucket, the structure **must** be:

```text
triton-store/              <-- Root Folder (The URI points here)
└── iris-model/            <-- Model Name (Must match config.pbtxt)
    ├── config.pbtxt       <-- Schema Definition
    └── 1/                 <-- Version Folder (Must be an integer)
        └── model.pt       <-- The Traced Model File
```

### **Step 4.2: The Configuration (`config.pbtxt`)**

Triton needs to know the input/output shapes and types explicitly.

```protobuf
name: "iris-model"         # Must match the folder name
platform: "pytorch_libtorch"
max_batch_size: 8

input [
  {
    name: "input__0"       # Standard PyTorch input name
    data_type: TYPE_FP32
    dims: [ 4 ]            # 4 Features (Sepal L/W, Petal L/W)
  }
]
output [
  {
    name: "output__0"      # Standard PyTorch output name
    data_type: TYPE_FP32
    dims: [ 3 ]            # 3 Classes (Probabilities/Logits)
  }
]
```

-----

## **5. Phase 3: Kubernetes Deployment**

### **Step 5.1: Secrets (MinIO Access)**

Seldon needs credentials to download the model.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: seldon-rclone-secret
type: Opaque
stringData:
  RCLONE_CONFIG_S3_TYPE: s3
  RCLONE_CONFIG_S3_PROVIDER: Minio
  RCLONE_CONFIG_S3_ENV_AUTH: "false"
  RCLONE_CONFIG_S3_ACCESS_KEY_ID: admin        # Your Key
  RCLONE_CONFIG_S3_SECRET_ACCESS_KEY: admin123 # Your Secret
  RCLONE_CONFIG_S3_ENDPOINT: http://minio.minio.svc.cluster.local:9000
```

### **Step 5.2: The Seldon Deployment YAML**

This instructs Kubernetes to spin up the Triton server.

**Key Configurations:**

  * **`implementation: TRITON_SERVER`**: Tells Seldon to use the V2 Protocol proxy.
  * **`modelUri`**: Must point to the **PARENT** folder (`triton-store`), not the model folder itself.
  * **`image`**: Used `nvcr.io/nvidia/tritonserver:23.08-py3` to support newer PyTorch versions.

<!-- end list -->

```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: iris-triton
spec:
  predictors:
  - name: default
    replicas: 1
    componentSpecs:
    - spec:
        containers:
        - name: iris-model
          image: nvcr.io/nvidia/tritonserver:23.08-py3
          resources:
            requests:
              cpu: 100m
              memory: 1Gi
            limits:
              cpu: 1000m
              memory: 4Gi
    graph:
      name: iris-model
      implementation: TRITON_SERVER
      modelUri: s3://mlflow-artifacts/triton-store  # Points to folder containing 'iris-model'
      envSecretRefName: seldon-rclone-secret
```

-----

## **6. Phase 4: Inference (KServe V2 Protocol)**

We interact with the model using the **KServe V2 Open Inference Protocol**. This is a standard JSON format accepted by Triton, TensorFlow Serving, and ONNX Runtime.

### **Step 6.1: The Request Format**

**POST** `http://<SELDON_URL>/v2/models/<MODEL_NAME>/infer`

```json
{
  "inputs": [
    {
      "name": "input__0",
      "shape": [1, 4],
      "datatype": "FP32",
      "data": [[5.1, 3.5, 1.4, 0.2]]
    }
  ]
}
```

### **Step 6.2: Python Client Code**

```python
import requests

URL = "http://localhost:8000/v2/models/iris-model/infer"
payload = {
    "inputs": [{
        "name": "input__0",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [[5.1, 3.5, 1.4, 0.2]]
    }]
}

response = requests.post(URL, json=payload)
print(response.json())
```

-----

## **7. Troubleshooting & Pitfalls**

| Error | Cause | Fix |
| :--- | :--- | :--- |
| **CrashLoopBackOff (Exit Code 1)** | Directory Mismatch. Triton cannot find the model folder inside the mount. | Ensure `modelUri` points to the *parent* folder, not the model folder itself. |
| **"Invalid argument: specific inputs are required"** | `config.pbtxt` Mismatch. The JSON input name does not match the config name. | Check `config.pbtxt` for input names (usually `input__0`) and update your JSON. |
| **Protocol Error (400 Bad Request)** | Using Seldon V1 JSON format on a V2 endpoint. | Ensure you use the `"inputs": [...]` structure, not `{"data": {"ndarray": ...}}`. |
| **Model Load Failed (PyTorch Version)** | The `model.pt` was trained on PyTorch 2.x but the Server image is old (PyTorch 1.x). | Update the Docker image in YAML to `nvcr.io/nvidia/tritonserver:23.08-py3` or later. |

-----

