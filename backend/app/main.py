"""
Customer Churn Dashboard - FastAPI Application
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.database import connect_to_mongo, close_mongo_connection
from app.routes.upload import router as upload_router
from app.routes.customers import router as customers_router

# Load environment variables
load_dotenv()

# Get configuration from environment
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Handles MongoDB connection lifecycle
    """
    # Startup: Connect to MongoDB
    print("\n" + "=" * 60)
    print("🚀 STARTING CUSTOMER CHURN DASHBOARD API")
    print("=" * 60)
    await connect_to_mongo()
    print("=" * 60)
    print(f"✅ API Ready! Environment: {ENVIRONMENT}")
    print("=" * 60 + "\n")

    yield  # Application runs here

    # Shutdown: Close MongoDB connection
    print("\n" + "=" * 60)
    print("🛑 SHUTTING DOWN API")
    print("=" * 60)
    await close_mongo_connection()
    print("=" * 60)
    print("✅ Shutdown complete")
    print("=" * 60 + "\n")


# Create FastAPI application
app = FastAPI(
    title="Customer Churn Dashboard API",
    description="FastAPI backend for customer churn prediction and analytics dashboard",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    debug=DEBUG
)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Include routers
app.include_router(upload_router, prefix="/api/upload", tags=["Upload"])
app.include_router(customers_router, prefix="/api/customers", tags=["Customers"])  # Add this line

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - Welcome message

    Returns:
        dict: Welcome message with API information
    """
    return {
        "message": "Welcome to Customer Churn Dashboard API",
        "version": "1.0.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "docs": "/docs",
        "redoc": "/redoc"
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Used to verify the API is running and MongoDB is connected

    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "api": "running",
        "database": "connected"
    }


# API Information endpoint
@app.get("/api/info", tags=["Info"])
async def api_info():
    """
    Get API information and available endpoints

    Returns:
        dict: API information
    """
    return {
        "name": "Customer Churn Dashboard API",
        "version": "1.0.0",
        "description": "Backend API for customer churn prediction and analytics",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "api_info": "/api/info",
            "upload_csv": "/api/upload/csv",
            "upload_status": "/api/upload/status"
        },
        "features": [
            "Customer data management",
            "CSV file upload",
            "Churn prediction",
            "Analytics and insights",
            "MongoDB integration"
        ]
    }