# 💎 Diamond Price Prediction API

> **Project Goal:** Implement polynomial regression + ridge regularization to predict diamond prices **from scratch** without using Scikit-learn estimators.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.3.5-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.125.0-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4051B5)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-Dev%20%7C%20Prod-2496ED)

</div>

---

## 🛠 Tech Stack

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Language** | 🐍 Python 3.12 | Core programming language |
| **ML Core** | 🔢 NumPy | Pretrained polynomial ridge model (No Scikit-learn) |
| **API** | ⚡ FastAPI, Uvicorn | High-performance ASGI framework |
| **Validation** | ✅ Pydantic v2 | Data validation and settings management |
| **DevOps** | 🐳 Docker | Multi-stage build for optimized images |

---

## 🧮 Model: Polynomial Ridge Regression (From Scratch)

This project implements a **Polynomial Ridge Regression model built entirely from scratch using NumPy**.

### Why this approach?

| Feature | Reason |
| :--- | :--- |
| **Polynomial Expansion** | Diamond prices are **highly non-linear**. Expansion captures interactions (e.g., `carat × cut`, `carat² × color`). |
| **Ridge Regularization** | Polynomial expansion causes high dimensionality. L2 penalty stabilizes weights, improves generalization, and reduces variance. |

### Model Formulation

After polynomial feature expansion, the model is defined as a linear regression:

$$\hat{y} = Xw$$

The training objective combines **Mean Squared Error (MSE)** with **Ridge (L2) regularization**:

$$L(w) = ||Xw - y||^2 + \lambda ||w||^2$$

**Where:**
* $\mathbf{X}$: Polynomial feature matrix
* $\mathbf{w}$: Model weights
* $\mathbf{y}$: Log-transformed target values
* $\mathbf{\lambda}$: Regularization strength

Although Ridge Regression admits a closed-form solution, this implementation optimizes the loss using **Gradient-Based Optimization**. The gradient is:

$$\frac{\partial L}{\partial w} = 2X^T(Xw - y) + 2\lambda w$$

Weights are updated iteratively using **Batch Gradient Descent**:

$$w_{t+1} = w_t - \alpha \cdot \frac{\partial L}{\partial w}$$

*(Where $\alpha$ is the learning rate)*

### Target Transformation & Inference

1.  **Training:** The model learns on log-transformed targets: $y_{train} = \log(1 + \text{price})$
2.  **Inference:** Predictions are restored to original scale: $\text{price} = \exp(y_{pred}) - 1$
3.  **Process:** No training occurs at API runtime. Only **pretrained weights** and artifacts are loaded.

---

## 📂 Project Structure
```text
├── artifacts/
│   └── models/
│       └── polynomial_ridge/  # Trained model & preprocessing artifacts
│           ├── features_schema.yaml
│           ├── features.yaml  # Features config (copied at training time)             
│           ├── metrics.yaml
│           ├── model_config.yaml
│           ├── scaler.pkl
│           └── weights.npy
├── configs/                   # Training-time feature configuration
├── data/                      # Raw & Cleaned data
│   ├── processed/             # Production dependencies
│   └── raw/
├── docker/                    # Docker files
│   ├── Dockerfile
│   ├── docker-compose.dev.yml
│   └── docker-compose.prod.yml
├── notebook/                  # Experiment notebook
├── requirements/
│   ├── base.txt               # Core dependencies
│   ├── prod.txt               # Production dependencies
│   └── dev.txt                # Development dependencies
├── scripts/
│   ├── evaluate.py            # Model evaluation cli
│   ├── predict.py             # Model prediction cli
│   └── train.py               # Model training cli
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   └── predict.py     # API routes 
│   │   ├── schema/
│   │   │   └── predict.py     # Pydantic request/response schemas      
│   │   └── app.py             # FastAPI entrypoint
│   ├── data/                  # Data loading & preprocessing logic
│   ├── features/              # Features engineering logic
│   ├── inference/             # Model loading & inference service
│   │    ├── artifacts_loader.py          
│   │    └── service.py
│   └── models/
│       └── service.py         # Polynomial Ridge model logic
├── .dockerignore
└── README.md
```

---
## 🧠 Technical Details
### 1. Input Schema Validation (Pydantic)
The API enforces strict validation rules for incoming requests:  
- Numerical constraints:
    - carat, depth, table, x, y, z must be greater than 0  
- Categorical constraints:  
    - cut, color, clarity must belong to predefined allowed sets

### 2. Inference Pipeline
- Pretrained model and preprocessing artifacts are loaded from /artifacts  
- No training logic is exposed in the API  
- The API focuses purely on low-latency prediction

### 3. Docker (Multi-stage Build)
- Base stage: installs shared dependencies
- Dev stage: includes development-only tools
- Prod stage: minimal runtime image for inference
- Benefits:
    - Smaller image size
    - Faster startup
    - Clear separation between dev & production environments
---
## 🚀 Quick Start
### Option 1: Docker (Recommended)
``` bash
# Development
docker compose -f docker/docker-compose.dev.yml up --build

# Production
docker build -t diamond-price-api .
docker run -p 8000:8000 diamond-price-api

```
Swagger UI: http://localhost:8000/docs

### Option 2: Local Development
``` bash
# Install dependencies
pip install -r requirements/dev.txt

# Start API server
uvicorn src.api.app:app --reload

```
---
## 🔌 API Usage
### Endpoint
POST /predict
``` json
Request Body
{
  "carat": 0.8,
  "cut": "Ideal",
  "color": "E",
  "clarity": "VS1",
  "depth": 61.5,
  "table": 55.0,
  "x": 5.95,
  "y": 5.98,
  "z": 3.65
}
```
### Response
``` json
{
  "price_prediction": 4114.459782224803
}
```
### Curl Command
``` bash
curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "records": [
    {
    "carat": 0.8,
    "cut": "Ideal",
    "color": "E",
    "clarity": "VS1",
    "depth": 61.5,
    "table": 55.0,
    "x": 5.95,
    "y": 5.98,
    "z": 3.65
    }
  ]
}'
```
---
## 👤 Author
**Huy Quach**

GitHub: [@QuachGHuy](https://github.com/QuachGHuy)  
LinkedIn: [Gia Huy Quach](https://www.linkedin.com/in/gia-huy-quach)