from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
import uuid
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import pickle

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Global variables for models
rf_model = None
lr_model = None
scaler = None
model_metrics = {}

# Pydantic Models
class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float
    merchant: str
    merchant_category: str
    location: str
    user_id: str
    card_type: str
    transaction_time: datetime
    ip_address: str
    device_type: str
    risk_score_lr: float = 0.0
    risk_score_rf: float = 0.0
    is_fraud: bool = False
    status: Literal["safe", "suspicious", "fraud"] = "safe"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TransactionCreate(BaseModel):
    amount: float
    merchant: str
    merchant_category: str
    location: str
    user_id: str
    card_type: str
    transaction_time: Optional[datetime] = None
    ip_address: str
    device_type: str

class Alert(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str
    severity: Literal["low", "medium", "high"]
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ModelPerformance(BaseModel):
    logistic_regression: dict
    random_forest: dict

class SimulateRequest(BaseModel):
    count: int = 10

# Helper Functions
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic transaction data for training"""
    np.random.seed(42)
    
    data = {
        'amount': np.random.exponential(100, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'distance_from_home': np.random.exponential(50, n_samples),
        'frequency_24h': np.random.poisson(3, n_samples),
        'merchant_risk': np.random.uniform(0, 1, n_samples),
        'is_international': np.random.binomial(1, 0.1, n_samples),
    }
    
    # Create fraud labels based on features
    fraud_prob = (
        (data['amount'] > 500) * 0.3 +
        (data['hour'] < 6) * 0.2 +
        (data['distance_from_home'] > 100) * 0.3 +
        (data['merchant_risk'] > 0.7) * 0.4 +
        data['is_international'] * 0.3
    )
    data['is_fraud'] = (fraud_prob > 0.6).astype(int)
    
    return pd.DataFrame(data)

def train_models():
    """Train both ML models"""
    global rf_model, lr_model, scaler, model_metrics
    
    # Generate training data
    df = generate_synthetic_data(2000)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    model_metrics = {
        'logistic_regression': {
            'accuracy': float(accuracy_score(y_test, lr_pred)),
            'precision': float(precision_score(y_test, lr_pred, zero_division=0)),
            'recall': float(recall_score(y_test, lr_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, lr_pred, zero_division=0))
        },
        'random_forest': {
            'accuracy': float(accuracy_score(y_test, rf_pred)),
            'precision': float(precision_score(y_test, rf_pred, zero_division=0)),
            'recall': float(recall_score(y_test, rf_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, rf_pred, zero_division=0))
        }
    }

def extract_features(transaction_data: dict) -> pd.DataFrame:
    """Extract features from transaction for prediction"""
    trans_time = transaction_data.get('transaction_time', datetime.now(timezone.utc))
    if isinstance(trans_time, str):
        trans_time = datetime.fromisoformat(trans_time.replace('Z', '+00:00'))
    
    features = {
        'amount': transaction_data['amount'],
        'hour': trans_time.hour,
        'distance_from_home': random.uniform(0, 200),
        'frequency_24h': random.randint(1, 10),
        'merchant_risk': hash(transaction_data['merchant']) % 100 / 100,
        'is_international': 1 if 'international' in transaction_data['location'].lower() else 0,
    }
    return pd.DataFrame([features])

def predict_fraud(transaction_data: dict) -> tuple:
    """Predict fraud probability using both models"""
    features = extract_features(transaction_data)
    features_scaled = scaler.transform(features)
    
    lr_prob = lr_model.predict_proba(features_scaled)[0][1]
    rf_prob = rf_model.predict_proba(features_scaled)[0][1]
    
    return float(lr_prob * 100), float(rf_prob * 100)

def determine_status(lr_score: float, rf_score: float) -> str:
    """Determine transaction status based on risk scores"""
    avg_score = (lr_score + rf_score) / 2
    if avg_score > 70:
        return "fraud"
    elif avg_score > 40:
        return "suspicious"
    return "safe"

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Fraud Detection API", "status": "active"}

@api_router.post("/transactions/analyze", response_model=Transaction)
async def analyze_transaction(transaction: TransactionCreate):
    """Analyze a single transaction for fraud"""
    try:
        # Convert to dict
        trans_dict = transaction.model_dump()
        if trans_dict['transaction_time'] is None:
            trans_dict['transaction_time'] = datetime.now(timezone.utc)
        
        # Predict fraud
        lr_score, rf_score = predict_fraud(trans_dict)
        
        # Create transaction object
        trans_obj = Transaction(
            **trans_dict,
            risk_score_lr=lr_score,
            risk_score_rf=rf_score,
            is_fraud=(lr_score > 70 or rf_score > 70),
            status=determine_status(lr_score, rf_score)
        )
        
        # Save to database
        doc = trans_obj.model_dump()
        doc['transaction_time'] = doc['transaction_time'].isoformat()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.transactions.insert_one(doc)
        
        # Create alert if high risk
        if trans_obj.status in ["suspicious", "fraud"]:
            severity = "high" if trans_obj.status == "fraud" else "medium"
            alert = Alert(
                transaction_id=trans_obj.id,
                severity=severity,
                message=f"{severity.capitalize()} risk transaction detected: ${trans_obj.amount:.2f} at {trans_obj.merchant}"
            )
            alert_doc = alert.model_dump()
            alert_doc['created_at'] = alert_doc['created_at'].isoformat()
            await db.alerts.insert_one(alert_doc)
        
        return trans_obj
    except Exception as e:
        logging.error(f"Error analyzing transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/transactions", response_model=List[Transaction])
async def get_transactions(limit: int = 50, status: Optional[str] = None):
    """Get all transactions with optional filters"""
    try:
        query = {}
        if status:
            query['status'] = status
        
        transactions = await db.transactions.find(query, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
        
        for trans in transactions:
            if isinstance(trans['transaction_time'], str):
                trans['transaction_time'] = datetime.fromisoformat(trans['transaction_time'])
            if isinstance(trans['created_at'], str):
                trans['created_at'] = datetime.fromisoformat(trans['created_at'])
        
        return transactions
    except Exception as e:
        logging.error(f"Error fetching transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str):
    """Get a single transaction by ID"""
    try:
        trans = await db.transactions.find_one({"id": transaction_id}, {"_id": 0})
        if not trans:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        if isinstance(trans['transaction_time'], str):
            trans['transaction_time'] = datetime.fromisoformat(trans['transaction_time'])
        if isinstance(trans['created_at'], str):
            trans['created_at'] = datetime.fromisoformat(trans['created_at'])
        
        return trans
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/alerts", response_model=List[Alert])
async def get_alerts(limit: int = 20):
    """Get recent alerts"""
    try:
        alerts = await db.alerts.find({}, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(limit)
        
        for alert in alerts:
            if isinstance(alert['created_at'], str):
                alert['created_at'] = datetime.fromisoformat(alert['created_at'])
        
        return alerts
    except Exception as e:
        logging.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/models/performance", response_model=ModelPerformance)
async def get_model_performance():
    """Get model performance metrics"""
    return ModelPerformance(**model_metrics)

@api_router.post("/simulate", response_model=List[Transaction])
async def simulate_transactions(request: SimulateRequest):
    """Simulate random transactions"""
    try:
        merchants = ["Amazon", "Walmart", "Target", "Best Buy", "Apple Store", "Gas Station", "Restaurant", "Hotel", "Airline", "Online Casino"]
        categories = ["retail", "grocery", "electronics", "travel", "entertainment", "fuel", "food"]
        locations = ["New York, US", "Los Angeles, US", "London, UK", "Tokyo, Japan", "Dubai, UAE", "Miami, US"]
        card_types = ["visa", "mastercard", "amex"]
        devices = ["mobile", "desktop", "tablet"]
        
        transactions = []
        for _ in range(request.count):
            trans_create = TransactionCreate(
                amount=round(random.uniform(5, 2000), 2),
                merchant=random.choice(merchants),
                merchant_category=random.choice(categories),
                location=random.choice(locations),
                user_id=f"user_{random.randint(1000, 9999)}",
                card_type=random.choice(card_types),
                transaction_time=datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 1440)),
                ip_address=f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                device_type=random.choice(devices)
            )
            
            trans_result = await analyze_transaction(trans_create)
            transactions.append(trans_result)
        
        return transactions
    except Exception as e:
        logging.error(f"Error simulating transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stats")
async def get_stats():
    """Get dashboard statistics"""
    try:
        total = await db.transactions.count_documents({})
        fraud = await db.transactions.count_documents({"status": "fraud"})
        suspicious = await db.transactions.count_documents({"status": "suspicious"})
        safe = await db.transactions.count_documents({"status": "safe"})
        alerts_count = await db.alerts.count_documents({})
        
        # Calculate total amount
        pipeline = [{"$group": {"_id": None, "total": {"$sum": "$amount"}}}]
        result = await db.transactions.aggregate(pipeline).to_list(1)
        total_amount = result[0]['total'] if result else 0
        
        return {
            "total_transactions": total,
            "fraud_count": fraud,
            "suspicious_count": suspicious,
            "safe_count": safe,
            "alerts_count": alerts_count,
            "total_amount": round(total_amount, 2),
            "fraud_rate": round((fraud / total * 100) if total > 0 else 0, 2)
        }
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@api_router.on_event("startup")
async def startup_event():
    train_models()
    logging.info("Models trained successfully")

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()