"""
FastAPI Backend — Multimodal Healthcare Prediction API
--------------------------------------------------------
Exposes REST endpoints for individual branch predictions and
multimodal fusion inference. Loads pretrained model checkpoints
on startup.

Endpoints:
    GET  /health           — Health check + model status
    POST /predict/xray     — Chest X-ray pneumonia classification
    POST /predict/tabular  — Diabetes risk from tabular features
    POST /predict/timeseries — ICU mortality risk from vitals
    POST /predict/fusion   — Full multimodal fused prediction
"""

import os
import sys
import io
import yaml
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_branch import CNNBranch
from src.models.tabular_branch import TabularBranch
from src.models.rnn_branch import RNNBranch
from src.models.fusion_model import FusionModel


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model globals (populated on startup)
cnn_model: Optional[CNNBranch] = None
tabular_model: Optional[TabularBranch] = None
rnn_model: Optional[RNNBranch] = None
fusion_model: Optional[FusionModel] = None

# Scaler/Normalization stats loaded from checkpoints
tabular_scaler_mean: Optional[List[float]] = None
tabular_scaler_scale: Optional[List[float]] = None
rnn_stats: Optional[dict] = None

# Image transforms for CNN
IMAGE_SIZE = CONFIG['data']['xray']['image_size']
XRAY_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #

class TabularInput(BaseModel):
    Pregnancies: float = Field(..., description="Number of pregnancies")
    Glucose: float = Field(..., description="Plasma glucose concentration")
    BloodPressure: float = Field(..., description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., description="Body mass index")
    DiabetesPedigree: float = Field(..., alias="DiabetesPedigreeFunction",
                                    description="Diabetes pedigree function")
    Age: float = Field(..., description="Age in years")

    class Config:
        populate_by_name = True


class HospitalIntakeInput(BaseModel):
    PatientName: str = Field(..., description="Patient Full Name")
    Gender: str = Field(..., description="Biological Gender")
    Age: float = Field(..., description="Age in years")
    Height: float = Field(..., description="Height in centimeters")
    Weight: float = Field(..., description="Weight in kilograms")
    BMI: Optional[float] = Field(None, description="Body Mass Index (optional/auto-calculated)")
    BloodGlucose: float = Field(..., description="Blood Glucose Level (mg/dL)")
    BloodPressure: float = Field(..., description="Blood Pressure (mmHg)")
    FamilyHistory: str = Field(..., description="Family History of Diabetes")
    PhysicalActivity: str = Field(..., description="Physical Activity Level")
    SmokingStatus: str = Field(..., description="Smoking Status")
    MedicalConditions: str = Field(..., description="Existing Medical Conditions")
    Symptoms: List[str] = Field(default=[], description="Active Symptoms Checklist")
    HbA1c: float = Field(..., description="HbA1c Level (%)")
    InsulinLevel: float = Field(..., description="Insulin Level (mu U/ml)")

    # Optional Advanced Parameters
    Pregnancies: Optional[float] = Field(None, description="Number of pregnancies")
    SkinThickness: Optional[float] = Field(None, description="Triceps skin fold thickness (mm)")
    DiabetesPedigreeFunction: Optional[float] = Field(None, description="Diabetes pedigree function")
    CholesterolLevel: Optional[float] = Field(None, description="Cholesterol Level (mg/dL)")
    SleepDuration: Optional[float] = Field(None, description="Sleep Duration (hours)")
    StressLevel: Optional[float] = Field(None, description="Stress Level (1-10)")
    DietaryPattern: Optional[str] = Field(None, description="Dietary Pattern")
    AlcoholConsumption: Optional[str] = Field(None, description="Alcohol Consumption Pattern")
    HeartRate: Optional[float] = Field(None, description="Heart Rate (bpm)")
    OxygenSaturation: Optional[float] = Field(None, description="Oxygen Saturation (%)")
    WaistCircumference: Optional[float] = Field(None, description="Waist Circumference (cm)")


class HospitalIntakePrediction(BaseModel):
    prediction: str
    confidence: float
    clinical_interpretation: str


class TabularPrediction(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict



class XrayPrediction(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict


class TimeSeriesInput(BaseModel):
    sequences: List[List[List[float]]] = Field(
        ..., description="List of sequences, each (48, 12) — batch of ICU vital sequences"
    )


class TimeSeriesPrediction(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict


class FusionInput(BaseModel):
    tabular: TabularInput
    timeseries: List[List[float]] = Field(
        ..., description="Single ICU sequence (48, 12)"
    )


class FusionPrediction(BaseModel):
    binary_risk: str
    severity: str
    binary_confidence: float
    severity_probabilities: dict
    modality_confidence: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: list
    device: str


# --------------------------------------------------------------------------- #
# Helper Functions for Feature Scaling
# --------------------------------------------------------------------------- #

def scale_tabular_features(features: List[float]) -> List[float]:
    """Scale tabular features using the mean/scale parameters loaded from checkpoint."""
    if tabular_scaler_mean and tabular_scaler_scale:
        return [
            (f - m) / s
            for f, m, s in zip(features, tabular_scaler_mean, tabular_scaler_scale)
        ]
    return features


def scale_timeseries_features(sequence: np.ndarray) -> np.ndarray:
    """Scale a 48x12 vital sequence using the dataset mean/std parameters from checkpoint."""
    if rnn_stats and 'means' in rnn_stats and 'stds' in rnn_stats:
        means = np.array(rnn_stats['means'], dtype=np.float32)
        stds = np.array(rnn_stats['stds'], dtype=np.float32)
        return (sequence - means) / stds
    return sequence


# --------------------------------------------------------------------------- #
# Model Loading
# --------------------------------------------------------------------------- #

def load_models():
    """Load all pretrained model checkpoints and their scaling statistics."""
    global cnn_model, tabular_model, rnn_model, fusion_model
    global tabular_scaler_mean, tabular_scaler_scale, rnn_stats

    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', CONFIG['outputs']['checkpoints']
    )

    # --- CNN ---
    cnn_cfg = CONFIG['models']['cnn']
    cnn_model = CNNBranch(
        num_classes=CONFIG['data']['xray']['num_classes'],
        embedding_dim=cnn_cfg['embedding_dim'],
        dropout=cnn_cfg['dropout'],
        pretrained=False,
    ).to(DEVICE)
    cnn_ckpt = os.path.join(ckpt_dir, 'cnn_best.pth')
    if os.path.exists(cnn_ckpt):
        ckpt = torch.load(cnn_ckpt, map_location=DEVICE, weights_only=False)
        cnn_model.load_state_dict(ckpt['model_state_dict'])
        print(f"[API] CNN model loaded from {cnn_ckpt}")
    else:
        print(f"[API] WARNING: CNN checkpoint not found. Using untrained model.")
    cnn_model.eval()

    # --- Tabular ---
    tab_cfg = CONFIG['models']['tabular']
    tabular_model = TabularBranch(
        num_features=CONFIG['data']['tabular']['num_features'],
        num_classes=CONFIG['data']['tabular']['num_classes'],
        hidden_dims=tab_cfg['hidden_dims'],
        embedding_dim=tab_cfg['embedding_dim'],
        dropout=tab_cfg['dropout'],
    ).to(DEVICE)
    tab_ckpt = os.path.join(ckpt_dir, 'tabular_best.pth')
    if os.path.exists(tab_ckpt):
        ckpt = torch.load(tab_ckpt, map_location=DEVICE, weights_only=False)
        tabular_model.load_state_dict(ckpt['model_state_dict'])
        tabular_scaler_mean = ckpt.get('scaler_mean')
        tabular_scaler_scale = ckpt.get('scaler_scale')
        print(f"[API] Tabular model loaded from {tab_ckpt}")
        if tabular_scaler_mean and tabular_scaler_scale:
            print(f"[API] Tabular scaler loaded from checkpoint")
    else:
        print(f"[API] WARNING: Tabular checkpoint not found. Using untrained model.")
    tabular_model.eval()

    # --- RNN ---
    rnn_cfg = CONFIG['models']['rnn']
    ts_cfg = CONFIG['data']['timeseries']
    rnn_model = RNNBranch(
        input_size=ts_cfg['num_vitals'],
        hidden_size=rnn_cfg['hidden_size'],
        num_layers=rnn_cfg['num_layers'],
        num_classes=ts_cfg['num_classes'],
        embedding_dim=rnn_cfg['embedding_dim'],
        dropout=rnn_cfg['dropout'],
    ).to(DEVICE)
    rnn_ckpt = os.path.join(ckpt_dir, 'rnn_best.pth')
    if os.path.exists(rnn_ckpt):
        ckpt = torch.load(rnn_ckpt, map_location=DEVICE, weights_only=False)
        rnn_model.load_state_dict(ckpt['model_state_dict'])
        rnn_stats = ckpt.get('stats')
        print(f"[API] RNN model loaded from {rnn_ckpt}")
        if rnn_stats:
            print(f"[API] RNN timeseries stats loaded from checkpoint")
    else:
        print(f"[API] WARNING: RNN checkpoint not found. Using untrained model.")
    rnn_model.eval()

    # --- Fusion ---
    fusion_cfg = CONFIG['models']['fusion']
    fusion_model = FusionModel(
        cnn_branch=cnn_model,
        tabular_branch=tabular_model,
        rnn_branch=rnn_model,
        freeze_branches=True,
        hidden_dims=fusion_cfg['hidden_dims'],
        dropout=fusion_cfg['dropout'],
        num_severity_classes=fusion_cfg['num_severity_classes'],
    ).to(DEVICE)
    fusion_ckpt = os.path.join(ckpt_dir, 'fusion_best.pth')
    if os.path.exists(fusion_ckpt):
        ckpt = torch.load(fusion_ckpt, map_location=DEVICE, weights_only=False)
        fusion_model.load_state_dict(ckpt['fusion_state_dict'])
        print(f"[API] Fusion model loaded from {fusion_ckpt}")
    else:
        print(f"[API] WARNING: Fusion checkpoint not found. Using untrained model.")
    fusion_model.eval()


# --------------------------------------------------------------------------- #
# FastAPI App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Multimodal Healthcare Prediction API",
    description=(
        "Academic prototype API for multimodal health risk prediction. "
        "Combines chest X-ray (CNN), diabetes tabular data (MLP), and "
        "ICU time-series vitals (LSTM) via a fusion model."
    ),
    version="1.0.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — returns model loading status."""
    models_loaded = []
    if cnn_model is not None:
        models_loaded.append("cnn")
    if tabular_model is not None:
        models_loaded.append("tabular")
    if rnn_model is not None:
        models_loaded.append("rnn")
    if fusion_model is not None:
        models_loaded.append("fusion")

    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        device=str(DEVICE),
    )


@app.post("/predict/xray", response_model=XrayPrediction)
async def predict_xray(file: UploadFile = File(...)):
    """
    Classify a chest X-ray image as NORMAL or PNEUMONIA.
    Accepts PNG, JPEG, or BMP image uploads.
    """
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")

    # Validate file type
    if file.content_type not in ("image/png", "image/jpeg", "image/bmp"):
        raise HTTPException(status_code=400,
                            detail="Unsupported image format. Use PNG, JPEG, or BMP.")

    # Load and preprocess
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = XRAY_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = cnn_model(tensor)
        proba = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(proba))
    class_names = CONFIG['data']['xray']['class_names']

    return XrayPrediction(
        prediction=class_names[pred_class],
        confidence=float(proba[pred_class]),
        class_probabilities={
            class_names[i]: round(float(proba[i]), 4)
            for i in range(len(class_names))
        },
    )


def generate_clinical_interpretation(data: HospitalIntakeInput, risk_level: str, p_diabetes: float, bmi: float, pedigree: float) -> str:
    symptoms_str = ", ".join(data.Symptoms) if data.Symptoms else "None"
    interpretation = f"Patient intake analysis for {data.PatientName} ({int(data.Age)}-year-old {data.Gender}) indicates a {risk_level} of diabetes (estimated probability: {p_diabetes*100:.1f}%). "
    
    findings = []
    if data.BloodGlucose > 125:
        findings.append(f"hyperglycemia ({data.BloodGlucose} mg/dL)")
    elif data.BloodGlucose > 100:
        findings.append(f"impaired fasting glucose ({data.BloodGlucose} mg/dL)")
        
    if data.HbA1c >= 6.5:
        findings.append(f"HbA1c level of {data.HbA1c}% indicating diabetic range")
    elif data.HbA1c >= 5.7:
        findings.append(f"HbA1c level of {data.HbA1c}% indicating prediabetic range")
        
    if bmi >= 30:
        findings.append(f"obese BMI classification ({bmi:.1f} kg/m²)")
    elif bmi >= 25:
        findings.append(f"overweight BMI classification ({bmi:.1f} kg/m²)")
        
    if data.BloodPressure >= 130:
        findings.append(f"elevated blood pressure / hypertension ({data.BloodPressure} mmHg)")
        
    if data.FamilyHistory in ["First-Degree Relative", "Both Parents", "Yes"]:
        findings.append("strong family history of diabetes")
        
    if findings:
        interpretation += "Key clinical contributors include: " + ", ".join(findings) + ". "
    else:
        interpretation += "No major metabolic or cardiovascular risk indicators were flagged. "
        
    if data.Symptoms:
        interpretation += f"Active symptoms reported: {symptoms_str}. "
        
    if risk_level == "HIGH RISK":
        interpretation += "Immediate clinical follow-up is recommended, including oral glucose tolerance testing and lifestyle/dietary intervention."
    elif risk_level == "MODERATE RISK":
        interpretation += "Routine screening and metabolic monitoring are advised, alongside proactive lifestyle counseling."
    else:
        interpretation += "Continue regular preventive care and healthy lifestyle practices."
        
    return interpretation


@app.post("/predict/tabular", response_model=TabularPrediction)
async def predict_tabular(data: TabularInput):
    """
    Predict diabetes risk from 8 tabular clinical features.
    """
    if tabular_model is None:
        raise HTTPException(status_code=503, detail="Tabular model not loaded")

    # Extract features in correct order and scale them
    features = [
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigree, data.Age
    ]
    scaled_features = scale_tabular_features(features)
    tensor = torch.tensor([scaled_features], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = tabular_model(tensor)
        proba = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(proba))
    class_names = CONFIG['data']['tabular']['class_names']

    return TabularPrediction(
        prediction=class_names[pred_class],
        confidence=float(proba[pred_class]),
        class_probabilities={
            class_names[i]: round(float(proba[i]), 4)
            for i in range(len(class_names))
        },
    )


@app.post("/api/predict/tabular", response_model=HospitalIntakePrediction)
async def predict_hospital_intake(data: HospitalIntakeInput):
    """
    Predict diabetes risk from the expanded hospital patient intake form,
    mapping values to the underlying tabular MLP model's 8 features, and
    generating a clinical interpretation.
    """
    if tabular_model is None:
        raise HTTPException(status_code=503, detail="Tabular model not loaded")

    # Auto-calculate BMI if not provided
    calculated_bmi = data.BMI
    if calculated_bmi is None or calculated_bmi <= 0:
        if data.Height > 0:
            calculated_bmi = data.Weight / ((data.Height / 100) ** 2)
        else:
            calculated_bmi = 22.0 # Fallback

    # Map Family History of Diabetes to DiabetesPedigreeFunction score
    # Yes -> 0.65, No -> 0.15, First-Degree -> 0.65, Second-Degree -> 0.35, Both Parents -> 0.85
    pedigree = 0.25
    if data.DiabetesPedigreeFunction is not None:
        pedigree = data.DiabetesPedigreeFunction
    else:
        fh = data.FamilyHistory.lower()
        if "both" in fh:
            pedigree = 0.85
        elif "first" in fh or "parent" in fh or "sibling" in fh or fh == "yes":
            pedigree = 0.65
        elif "second" in fh or "grandparent" in fh or "uncle" in fh or "aunt" in fh:
            pedigree = 0.35
        elif "no" in fh or "none" in fh:
            pedigree = 0.15

    # Map Pregnancies
    pregnancies = 0.0
    if data.Pregnancies is not None:
        pregnancies = data.Pregnancies
    elif data.Gender.lower() == "female":
        pregnancies = 1.0 if data.Age > 25 else 0.0

    # Map SkinThickness
    skin_thickness = 20.0
    if data.SkinThickness is not None:
        skin_thickness = data.SkinThickness

    # Extract Glucose and Insulin Level
    glucose = data.BloodGlucose
    insulin = data.InsulinLevel

    # Assemble PIMA 8 features in correct order:
    # 1. Pregnancies, 2. Glucose, 3. BloodPressure, 4. SkinThickness, 5. Insulin, 6. BMI, 7. DiabetesPedigreeFunction, 8. Age
    features = [
        float(pregnancies),
        float(glucose),
        float(data.BloodPressure),
        float(skin_thickness),
        float(insulin),
        float(calculated_bmi),
        float(pedigree),
        float(data.Age)
    ]

    scaled_features = scale_tabular_features(features)
    tensor = torch.tensor([scaled_features], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = tabular_model(tensor)
        proba = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Probability of diabetes is proba[1] (class index 1 is "Diabetes")
    p_diabetes = float(proba[1])

    # Stratify risk level
    if p_diabetes < 0.35:
        risk_level = "LOW RISK"
    elif p_diabetes <= 0.65:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "HIGH RISK"

    # Confidence calculation: max(p, 1-p) * 100
    confidence = float(max(proba[0], proba[1]) * 100)

    # Generate Clinical Interpretation
    clinical_interpretation = generate_clinical_interpretation(data, risk_level, p_diabetes, calculated_bmi, pedigree)

    return HospitalIntakePrediction(
        prediction=risk_level,
        confidence=round(confidence, 1),
        clinical_interpretation=clinical_interpretation
    )


@app.post("/predict/timeseries", response_model=TimeSeriesPrediction)
async def predict_timeseries(data: TimeSeriesInput):
    """
    Predict ICU mortality risk from time-series vital signs.
    Input: batch of sequences, each shape (48, 12).
    """
    if rnn_model is None:
        raise HTTPException(status_code=503, detail="RNN model not loaded")

    sequences = np.array(data.sequences, dtype=np.float32)
    if sequences.ndim != 3 or sequences.shape[1:] != (48, 12):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (batch, 48, 12), got {sequences.shape}"
        )

    # Scale the sequence batch
    scaled_sequences = np.array([scale_timeseries_features(seq) for seq in sequences], dtype=np.float32)
    tensor = torch.tensor(scaled_sequences, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = rnn_model(tensor)
        proba = F.softmax(logits, dim=1)  # (batch, 2)

    # Return prediction for first sample
    p = proba[0].cpu().numpy()
    pred_class = int(np.argmax(p))
    class_names = CONFIG['data']['timeseries']['class_names']

    return TimeSeriesPrediction(
        prediction=class_names[pred_class],
        confidence=float(p[pred_class]),
        class_probabilities={
            class_names[i]: round(float(p[i]), 4)
            for i in range(len(class_names))
        },
    )


@app.post("/predict/fusion", response_model=FusionPrediction)
async def predict_fusion(
    data: str = Form(..., description="JSON string of FusionInput"),
    xray_file: Optional[UploadFile] = File(None)
):
    """
    Multimodal fused prediction combining chest X-ray, tabular features,
    and ICU time-series vitals. Returns binary risk, severity classification,
    and per-modality confidence scores.

    Note: If no X-ray is provided, a random noise tensor is used (placeholder).
    """
    if fusion_model is None:
        raise HTTPException(status_code=503, detail="Fusion model not loaded")

    # Parse JSON from Form data
    try:
        data_dict = json.loads(data)
        input_data = FusionInput(**data_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in data field: {e}")

    # --- X-ray input ---
    if xray_file is not None:
        contents = await xray_file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        xray_tensor = XRAY_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    else:
        # Placeholder: random noise (academic prototype)
        xray_tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # --- Tabular input ---
    tab_features = [
        input_data.tabular.Pregnancies, input_data.tabular.Glucose, input_data.tabular.BloodPressure,
        input_data.tabular.SkinThickness, input_data.tabular.Insulin, input_data.tabular.BMI,
        input_data.tabular.DiabetesPedigree, input_data.tabular.Age
    ]
    scaled_tab = scale_tabular_features(tab_features)
    tab_tensor = torch.tensor([scaled_tab], dtype=torch.float32).to(DEVICE)

    # --- Time-series input ---
    ts_array = np.array(input_data.timeseries, dtype=np.float32)
    if ts_array.shape != (48, 12):
        raise HTTPException(
            status_code=400,
            detail=f"Expected timeseries shape (48, 12), got {ts_array.shape}"
        )
    scaled_ts = scale_timeseries_features(ts_array)
    ts_tensor = torch.tensor(scaled_ts, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # --- Fusion inference ---
    with torch.no_grad():
        output = fusion_model(xray_tensor, tab_tensor, ts_tensor)

    binary_logits = output['binary_logits']
    severity_logits = output['severity_logits']
    mod_conf = output['modality_confidence']

    binary_proba = F.softmax(binary_logits, dim=1)[0].cpu().numpy()
    severity_proba = F.softmax(severity_logits, dim=1)[0].cpu().numpy()

    binary_pred = int(np.argmax(binary_proba))
    severity_pred = int(np.argmax(severity_proba))

    binary_names = ["Low Risk", "High Risk"]
    severity_names = CONFIG['models']['fusion']['severity_names']

    return FusionPrediction(
        binary_risk=binary_names[binary_pred],
        severity=severity_names[severity_pred],
        binary_confidence=float(binary_proba[binary_pred]),
        severity_probabilities={
            severity_names[i]: round(float(severity_proba[i]), 4)
            for i in range(len(severity_names))
        },
        modality_confidence={
            'cnn': {
                CONFIG['data']['xray']['class_names'][i]: round(float(v), 4)
                for i, v in enumerate(mod_conf['cnn'][0].cpu().numpy())
            },
            'tabular': {
                CONFIG['data']['tabular']['class_names'][i]: round(float(v), 4)
                for i, v in enumerate(mod_conf['tabular'][0].cpu().numpy())
            },
            'rnn': {
                CONFIG['data']['timeseries']['class_names'][i]: round(float(v), 4)
                for i, v in enumerate(mod_conf['rnn'][0].cpu().numpy())
            },
        },
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
