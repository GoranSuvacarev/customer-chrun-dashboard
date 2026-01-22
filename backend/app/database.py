"""
MongoDB Database Connection Module

This module handles:
- Async MongoDB connection using Motor
- Database initialization and shutdown
- Collection access helpers
- Connection health monitoring
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "churn_dashboard")

# Global variables for database connection
client: AsyncIOMotorClient = None
database = None


async def connect_to_mongo():
    """
    Connect to MongoDB database
    Called when the application starts
    """
    global client, database

    try:
        print(f"🔄 Connecting to MongoDB at {MONGODB_URL}...")

        # Create async MongoDB client
        client = AsyncIOMotorClient(MONGODB_URL)

        # Get database instance
        database = client[DATABASE_NAME]

        # Test the connection by pinging the server
        await client.admin.command('ping')

        print(f"✅ Successfully connected to MongoDB!")
        print(f"📊 Database: {DATABASE_NAME}")

        # List existing collections
        collections = await database.list_collection_names()
        if collections:
            print(f"📂 Existing collections: {', '.join(collections)}")
        else:
            print(f"📂 No collections yet (will be created when data is inserted)")

    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        raise e


async def close_mongo_connection():
    """
    Close MongoDB connection
    Called when the application shuts down
    """
    global client

    try:
        if client:
            print("🔄 Closing MongoDB connection...")
            client.close()
            print("✅ MongoDB connection closed successfully")
    except Exception as e:
        print(f"❌ Error closing MongoDB connection: {e}")
        raise e


def get_database():
    """
    Get the database instance
    Use this in your routes/services to access the database

    Returns:
        database: AsyncIOMotorDatabase instance
    """
    if database is None:
        raise Exception("Database not initialized. Call connect_to_mongo() first.")
    return database


def get_collection(collection_name: str):
    """
    Get a specific collection from the database

    Args:
        collection_name: Name of the collection to access

    Returns:
        collection: AsyncIOMotorCollection instance
    """
    db = get_database()
    return db[collection_name]


# Collection names (define them here for consistency)
class Collections:
    """Collection names used in the application"""
    CUSTOMERS = "customers"
    MODEL_METADATA = "model_metadata"
    PREDICTIONS = "predictions"
    MODELS = "models"
    ANALYTICS = "analytics"