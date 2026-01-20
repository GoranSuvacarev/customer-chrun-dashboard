"""
Pydantic Models for Customer Churn Dashboard
Data validation and serialization schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import datetime
from bson import ObjectId


# Custom ObjectId type for MongoDB
class PyObjectId(ObjectId):
    """Custom ObjectId type that works with Pydantic"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


# ============================================================
# SERVICES MODEL (Nested Object)
# ============================================================

class Services(BaseModel):
    """
    Customer service subscriptions
    Represents all telco services the customer has
    """

    # Phone Services
    phone_service: Literal["Yes", "No"] = Field(
        ...,
        alias="PhoneService",
        description="Whether customer has phone service"
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ...,
        alias="MultipleLines",
        description="Whether customer has multiple phone lines"
    )

    # Internet Services
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        alias="InternetService",
        description="Type of internet service (DSL, Fiber optic, or No)"
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        alias="OnlineSecurity",
        description="Whether customer has online security add-on"
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        alias="OnlineBackup",
        description="Whether customer has online backup service"
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        alias="DeviceProtection",
        description="Whether customer has device protection plan"
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        alias="TechSupport",
        description="Whether customer has tech support service"
    )

    # Streaming Services
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        alias="StreamingTV",
        description="Whether customer has streaming TV service"
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        alias="StreamingMovies",
        description="Whether customer has streaming movies service"
    )

    class Config:
        populate_by_name = True  # Allow both alias and field name
        json_schema_extra = {
            "example": {
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes"
            }
        }


# ============================================================
# PREDICTION MODEL (Nested Object)
# ============================================================

class Prediction(BaseModel):
    """
    Churn prediction results from ML model
    Stores prediction probability and classification
    """

    will_churn: bool = Field(
        ...,
        description="Predicted churn classification (True = will churn, False = will not churn)"
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of customer churning (0.0 to 1.0)"
    )
    confidence: Literal["Low", "Medium", "High"] = Field(
        ...,
        description="Confidence level of prediction"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="List of factors contributing to churn risk"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Version of ML model used for prediction"
    )
    predicted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when prediction was made"
    )

    @validator('confidence', pre=True, always=True)
    def determine_confidence(cls, v, values):
        """Automatically determine confidence based on probability"""
        if 'churn_probability' in values:
            prob = values['churn_probability']
            if prob < 0.3 or prob > 0.7:
                return "High"
            elif prob < 0.4 or prob > 0.6:
                return "Medium"
            else:
                return "Low"
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "will_churn": True,
                "churn_probability": 0.78,
                "confidence": "High",
                "risk_factors": ["Month-to-month contract", "High monthly charges", "No online security"],
                "model_version": "1.0.0",
                "predicted_at": "2024-01-20T10:30:00"
            }
        }


# ============================================================
# CUSTOMER MODEL (Main Model)
# ============================================================

class Customer(BaseModel):
    """
    Customer information and account details
    Main model representing a telco customer
    """

    # Customer Identification
    customer_id: str = Field(
        ...,
        alias="customerID",
        description="Unique customer identifier",
        min_length=1
    )

    # Demographics
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender"
    )
    senior_citizen: int = Field(
        ...,
        alias="SeniorCitizen",
        ge=0,
        le=1,
        description="Whether customer is a senior citizen (0 = No, 1 = Yes)"
    )
    partner: Literal["Yes", "No"] = Field(
        ...,
        alias="Partner",
        description="Whether customer has a partner"
    )
    dependents: Literal["Yes", "No"] = Field(
        ...,
        alias="Dependents",
        description="Whether customer has dependents"
    )

    # Account Information
    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months customer has been with company"
    )

    # Services (nested object)
    services: Services = Field(
        ...,
        description="All customer service subscriptions"
    )

    # Contract Details
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        alias="Contract",
        description="Type of contract"
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ...,
        alias="PaperlessBilling",
        description="Whether customer uses paperless billing"
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ] = Field(
        ...,
        alias="PaymentMethod",
        description="Payment method used by customer"
    )

    # Billing
    monthly_charges: float = Field(
        ...,
        alias="MonthlyCharges",
        ge=0,
        description="Monthly charges in dollars"
    )
    total_charges: float = Field(
        ...,
        alias="TotalCharges",
        ge=0,
        description="Total charges to date in dollars"
    )

    # Churn Status (actual)
    churn: Literal["Yes", "No"] = Field(
        ...,
        alias="Churn",
        description="Whether customer has churned (actual value)"
    )

    # Prediction (optional - added after ML prediction)
    prediction: Optional[Prediction] = Field(
        None,
        description="Churn prediction from ML model (if available)"
    )

    @validator('total_charges', pre=True)
    def parse_total_charges(cls, v):
        """Handle empty string or space in TotalCharges field"""
        if isinstance(v, str):
            v = v.strip()
            if v == '' or v == ' ':
                return 0.0
            return float(v)
        return v

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "services": {
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No"
                },
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
                "Churn": "No"
            }
        }


# ============================================================
# CUSTOMER IN DATABASE MODEL (with MongoDB fields)
# ============================================================

class CustomerInDB(Customer):
    """
    Customer model as stored in MongoDB
    Includes MongoDB _id and timestamps
    """

    id: Optional[PyObjectId] = Field(
        None,
        alias="_id",
        description="MongoDB document ID"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when record was last updated"
    )

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "services": {
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No"
                },
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
                "Churn": "No",
                "created_at": "2024-01-20T10:00:00",
                "updated_at": "2024-01-20T10:00:00"
            }
        }


# ============================================================
# RESPONSE MODELS
# ============================================================

class CustomerResponse(BaseModel):
    """Standard response for customer operations"""
    success: bool
    message: str
    data: Optional[CustomerInDB] = None


class CustomersListResponse(BaseModel):
    """Response for listing multiple customers"""
    success: bool
    message: str
    total: int
    data: list[CustomerInDB]


class PredictionResponse(BaseModel):
    """Response for prediction requests"""
    success: bool
    message: str
    customer_id: str
    prediction: Prediction