# backend/app/routes/analytics.py

from fastapi import APIRouter, HTTPException
from app.database import get_database
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# GET /api/analytics/overview - Dashboard Overview Statistics
# ============================================================================

@router.get("/overview")
async def get_overview() -> Dict[str, Any]:
    """
    Get comprehensive overview statistics for dashboard.

    Returns:
        - total_customers: Total count of customers in database
        - customers_with_predictions: Customers that have churn predictions
        - high_risk_count: Customers with "High" risk level
        - medium_risk_count: Customers with "Medium" risk level
        - low_risk_count: Customers with "Low" risk level
        - at_risk_count: High + Medium risk customers
        - churn_rate: Historical churn rate from actual 'Churn' field
        - model_accuracy: Latest model's accuracy from metadata
    """
    try:
        db = get_database()
        customers_collection = db["customers"]
        model_metadata_collection = db["model_metadata"]

        # --------------------------------------------------------------------
        # 1. Total customer count
        # --------------------------------------------------------------------
        total_customers = await customers_collection.count_documents({})

        # --------------------------------------------------------------------
        # 2. Customers with predictions (have churn_risk field)
        # --------------------------------------------------------------------
        customers_with_predictions = await customers_collection.count_documents({
            "prediction.risk_level": {"$exists": True, "$ne": None}
        })

        # --------------------------------------------------------------------
        # 3. Risk level distribution
        # MongoDB Query: Count documents for each risk level
        # --------------------------------------------------------------------
        high_risk_count = await customers_collection.count_documents({
            "prediction.risk_level": "High"
        })

        medium_risk_count = await customers_collection.count_documents({
            "prediction.risk_level": "Medium"
        })

        low_risk_count = await customers_collection.count_documents({
            "prediction.risk_level": "Low"
        })

        at_risk_count = high_risk_count + medium_risk_count

        # --------------------------------------------------------------------
        # 4. Historical churn rate (from actual 'Churn' field in data)
        # MongoDB Query: Calculate percentage of customers who churned
        # --------------------------------------------------------------------
        churned_customers = await customers_collection.count_documents({
            "Churn": "Yes"
        })

        churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0.0

        # --------------------------------------------------------------------
        # 5. Latest model accuracy from model_metadata collection
        # MongoDB Query: Find most recent model by created_at, sorted descending
        # --------------------------------------------------------------------
        latest_model = await model_metadata_collection.find_one(
            {},
            sort=[("created_at", -1)]  # -1 = descending order (newest first)
        )

        model_accuracy = None
        if latest_model and "metrics" in latest_model:
            model_accuracy = latest_model["metrics"].get("accuracy")

        # --------------------------------------------------------------------
        # Build response
        # --------------------------------------------------------------------
        return {
            "success": True,
            "data": {
                "total_customers": total_customers,
                "customers_with_predictions": customers_with_predictions,
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "low_risk_count": low_risk_count,
                "at_risk_count": at_risk_count,
                "churn_rate": round(churn_rate, 2),
                "model_accuracy": round(model_accuracy, 2) if model_accuracy else None,
            }
        }

    except Exception as e:
        logger.error(f"Error getting overview analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GET /api/analytics/risk-distribution - Risk Level Distribution
# ============================================================================

@router.get("/risk-distribution")
async def get_risk_distribution() -> Dict[str, Any]:
    """
    Get distribution of customers by risk level.
    Returns array of objects with risk_level and count.
    Useful for pie charts and donut charts.

    Returns:
        Array of {risk_level: "High", count: 1867}
    """
    try:
        db = get_database()
        customers_collection = db["customers"]

        # --------------------------------------------------------------------
        # MongoDB Aggregation Pipeline
        # This is more efficient than multiple count queries
        # --------------------------------------------------------------------
        # Explanation:
        # 1. $match: Filter only customers with predictions
        # 2. $group: Group by churn_risk field and count each group
        # 3. $sort: Sort by count descending (highest first)

        pipeline = [
            # Stage 1: Only include customers with risk predictions
            {
                "$match": {
                    "prediction.risk_level": {"$exists": True, "$ne": None}
                }
            },
            # Stage 2: Group by risk level and count
            {
                "$group": {
                    "_id": "$prediction.risk_level",  # Group by this field
                    "count": {"$sum": 1}  # Count documents in each group
                }
            },
            # Stage 3: Sort by count (highest first)
            {
                "$sort": {"count": -1}
            }
        ]

        cursor = customers_collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)

        # Format results for frontend
        distribution = [
            {
                "risk_level": item["_id"],
                "count": item["count"]
            }
            for item in results
        ]

        return {
            "success": True,
            "data": distribution
        }

    except Exception as e:
        logger.error(f"Error getting risk distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GET /api/analytics/feature-importance - Top Features from Model
# ============================================================================

@router.get("/feature-importance")
async def get_feature_importance() -> Dict[str, Any]:
    """
    Get feature importance from latest trained model.
    Returns top 10 features that most influence churn predictions.

    Returns:
        Array of {feature: "Contract", importance: 0.2324}
    """
    try:
        db = get_database()
        model_metadata_collection = db["model_metadata"]

        # Get latest model metadata
        latest_model = await model_metadata_collection.find_one(
            {},
            sort=[("created_at", -1)]
        )

        if not latest_model:
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Please train a model first."
            )

        # Extract feature importance from model metadata
        feature_importance = latest_model.get("top_features", {})

        if not feature_importance:
            raise HTTPException(
                status_code=404,
                detail="Feature importance not found in model metadata"
            )

        # Convert dict to sorted list of tuples, get top 10
        # Example: {"Contract": 0.23, "tenure": 0.14}
        # becomes: [{"feature": "Contract", "importance": 0.23}, ...]
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Format for frontend
        features = [
            {
                "feature": feature_name,
                "importance": round(importance, 4)
            }
            for feature_name, importance in sorted_features
        ]

        return {
            "success": True,
            "data": features,
            "model_id": latest_model.get("model_id")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GET /api/analytics/churn-by-contract - Churn Rate by Contract Type
# ============================================================================

@router.get("/churn-by-contract")
async def get_churn_by_contract() -> Dict[str, Any]:
    """
    Analyze churn rate by contract type using MongoDB aggregation.
    Groups customers by contract type and calculates:
    - Total customers per contract type
    - Number who churned
    - Churn rate percentage

    Returns:
        Array of {contract_type, total_customers, churned_customers, churn_rate}
    """
    try:
        db = get_database()
        customers_collection = db["customers"]

        # --------------------------------------------------------------------
        # MongoDB Aggregation Pipeline - Advanced
        # --------------------------------------------------------------------
        # This pipeline performs complex calculations in the database
        # which is much faster than doing it in Python

        pipeline = [
            # Stage 1: Group by Contract type
            {
                "$group": {
                    "_id": "$Contract",  # Group by Contract field
                    "total_customers": {"$sum": 1},  # Count all customers
                    "churned_customers": {  # Count only churned
                        "$sum": {
                            "$cond": [  # Conditional sum
                                {"$eq": ["$Churn", "Yes"]},  # If Churn == "Yes"
                                1,  # Add 1
                                0  # Otherwise add 0
                            ]
                        }
                    }
                }
            },
            # Stage 2: Add calculated churn_rate field
            {
                "$addFields": {
                    "churn_rate": {
                        "$multiply": [  # Multiply by 100 for percentage
                            {
                                "$divide": [  # Divide churned by total
                                    "$churned_customers",
                                    "$total_customers"
                                ]
                            },
                            100
                        ]
                    }
                }
            },
            # Stage 3: Sort by churn rate (highest first)
            {
                "$sort": {"churn_rate": -1}
            }
        ]

        cursor = customers_collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)

        # Format results for frontend
        contract_analysis = [
            {
                "contract_type": item["_id"],
                "total_customers": item["total_customers"],
                "churned_customers": item["churned_customers"],
                "churn_rate": round(item["churn_rate"], 2)
            }
            for item in results
        ]

        return {
            "success": True,
            "data": contract_analysis
        }

    except Exception as e:
        logger.error(f"Error getting churn by contract: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BONUS: GET /api/analytics/churn-by-tenure - Churn by Customer Tenure
# ============================================================================

@router.get("/churn-by-tenure")
async def get_churn_by_tenure() -> Dict[str, Any]:
    """
    Analyze churn rate by tenure brackets.
    Groups customers into tenure ranges (0-12, 13-24, 25-36, etc. months)

    Returns:
        Array of {tenure_range, total_customers, churned_customers, churn_rate}
    """
    try:
        db = get_database()
        customers_collection = db["customers"]

        # MongoDB Aggregation Pipeline with tenure buckets
        pipeline = [
            # Stage 1: Create tenure brackets
            {
                "$addFields": {
                    "tenure_bracket": {
                        "$switch": {
                            "branches": [
                                {"case": {"$lte": ["$tenure", 12]}, "then": "0-12 months"},
                                {"case": {"$lte": ["$tenure", 24]}, "then": "13-24 months"},
                                {"case": {"$lte": ["$tenure", 36]}, "then": "25-36 months"},
                                {"case": {"$lte": ["$tenure", 48]}, "then": "37-48 months"},
                                {"case": {"$lte": ["$tenure", 60]}, "then": "49-60 months"},
                            ],
                            "default": "60+ months"
                        }
                    }
                }
            },
            # Stage 2: Group by tenure bracket
            {
                "$group": {
                    "_id": "$tenure_bracket",
                    "total_customers": {"$sum": 1},
                    "churned_customers": {
                        "$sum": {
                            "$cond": [{"$eq": ["$Churn", "Yes"]}, 1, 0]
                        }
                    }
                }
            },
            # Stage 3: Calculate churn rate
            {
                "$addFields": {
                    "churn_rate": {
                        "$multiply": [
                            {"$divide": ["$churned_customers", "$total_customers"]},
                            100
                        ]
                    }
                }
            },
            # Stage 4: Sort by tenure (you'd need custom sorting logic for proper order)
            {
                "$sort": {"churn_rate": -1}
            }
        ]

        cursor = customers_collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)

        tenure_analysis = [
            {
                "tenure_range": item["_id"],
                "total_customers": item["total_customers"],
                "churned_customers": item["churned_customers"],
                "churn_rate": round(item["churn_rate"], 2)
            }
            for item in results
        ]

        return {
            "success": True,
            "data": tenure_analysis
        }

    except Exception as e:
        logger.error(f"Error getting churn by tenure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))