"""
Customers Route
Handles retrieving and managing customer data
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional, List
from bson import ObjectId

from app.database import get_collection, Collections
from app.models import CustomerInDB

# Create router
router = APIRouter()


def convert_object_id(document):
    """
    Convert MongoDB ObjectId to string for JSON serialization

    Args:
        document: MongoDB document with _id field

    Returns:
        Document with _id converted to string
    """
    if document and "_id" in document:
        document["_id"] = str(document["_id"])
    return document


@router.get("/")
async def get_customers(
        page: int = Query(1, ge=1, description="Page number (starts at 1)"),
        limit: int = Query(20, ge=1, le=100, description="Number of customers per page (max 100)"),
        search: Optional[str] = Query(None, description="Search by customer ID (case insensitive)"),
        churn: Optional[str] = Query(None, description="Filter by churn status (Yes/No)"),
        contract: Optional[str] = Query(None, description="Filter by contract type"),
        internet_service: Optional[str] = Query(None, description="Filter by internet service type")
):
    """
    Get paginated list of customers with optional filtering

    Query Parameters:
        - page: Page number (default: 1)
        - limit: Items per page (default: 20, max: 100)
        - search: Search by customer ID (case insensitive)
        - churn: Filter by churn status (Yes/No)
        - contract: Filter by contract type (Month-to-month, One year, Two year)
        - internet_service: Filter by internet service (DSL, Fiber optic, No)

    Returns:
        Paginated customer list with metadata
    """

    try:
        # Get customers collection
        customers_collection = get_collection(Collections.CUSTOMERS)

        # ========================================
        # Build Query Filter
        # ========================================

        query_filter = {}

        # Search by customer ID (case insensitive)
        if search:
            query_filter["customerID"] = {"$regex": search, "$options": "i"}

        # Filter by churn status
        if churn:
            if churn not in ["Yes", "No"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Churn filter must be 'Yes' or 'No'"
                )
            query_filter["Churn"] = churn

        # Filter by contract type
        if contract:
            valid_contracts = ["Month-to-month", "One year", "Two year"]
            if contract not in valid_contracts:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Contract must be one of: {', '.join(valid_contracts)}"
                )
            query_filter["Contract"] = contract

        # Filter by internet service
        if internet_service:
            valid_services = ["DSL", "Fiber optic", "No"]
            if internet_service not in valid_services:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Internet service must be one of: {', '.join(valid_services)}"
                )
            query_filter["InternetService"] = internet_service

        # ========================================
        # Calculate Pagination
        # ========================================

        # Calculate skip value (how many documents to skip)
        skip = (page - 1) * limit

        # Get total count of documents matching filter
        total_count = await customers_collection.count_documents(query_filter)

        # Calculate total pages
        total_pages = (total_count + limit - 1) // limit  # Ceiling division

        # Check if requested page exists
        if page > total_pages and total_count > 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page {page} does not exist. Total pages: {total_pages}"
            )

        # ========================================
        # Fetch Customers from Database
        # ========================================

        # Query database with pagination
        cursor = customers_collection.find(query_filter).skip(skip).limit(limit)

        # Convert cursor to list and process documents
        customers = []
        async for document in cursor:
            # Convert ObjectId to string
            document = convert_object_id(document)
            customers.append(document)

        # ========================================
        # Build Response
        # ========================================

        return {
            "success": True,
            "message": f"Retrieved {len(customers)} customers",
            "data": customers,
            "pagination": {
                "current_page": page,
                "page_size": limit,
                "total_items": total_count,
                "total_pages": total_pages,
                "has_previous": page > 1,
                "has_next": page < total_pages
            },
            "filters": {
                "search": search,
                "churn": churn,
                "contract": contract,
                "internet_service": internet_service
            }
        }

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving customers: {str(e)}"
        )


@router.get("/{customer_id}")
async def get_customer_by_id(customer_id: str):
    """
    Get a single customer by their customer ID

    Args:
        customer_id: Customer ID (e.g., '7590-VHVEG')

    Returns:
        Customer details
    """

    try:
        # Get customers collection
        customers_collection = get_collection(Collections.CUSTOMERS)

        # Find customer by customerID field
        customer = await customers_collection.find_one({"customerID": customer_id})

        # Check if customer exists
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with ID '{customer_id}' not found"
            )

        # Convert ObjectId to string
        customer = convert_object_id(customer)

        return {
            "success": True,
            "message": f"Customer {customer_id} retrieved successfully",
            "data": customer
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving customer: {str(e)}"
        )


@router.get("/stats/overview")
async def get_customers_overview():
    """
    Get overview statistics about customers

    Returns:
        Customer statistics and insights
    """

    try:
        customers_collection = get_collection(Collections.CUSTOMERS)

        # Total customers
        total = await customers_collection.count_documents({})

        # Churn statistics
        churned = await customers_collection.count_documents({"Churn": "Yes"})
        active = await customers_collection.count_documents({"Churn": "No"})
        churn_rate = (churned / total * 100) if total > 0 else 0

        # Contract distribution
        month_to_month = await customers_collection.count_documents({"Contract": "Month-to-month"})
        one_year = await customers_collection.count_documents({"Contract": "One year"})
        two_year = await customers_collection.count_documents({"Contract": "Two year"})

        # Internet service distribution
        dsl = await customers_collection.count_documents({"InternetService": "DSL"})
        fiber = await customers_collection.count_documents({"InternetService": "Fiber optic"})
        no_internet = await customers_collection.count_documents({"InternetService": "No"})

        # Senior citizens
        seniors = await customers_collection.count_documents({"SeniorCitizen": 1})

        # Calculate average monthly charges
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_monthly_charges": {"$avg": "$MonthlyCharges"},
                    "avg_total_charges": {"$avg": "$TotalCharges"},
                    "avg_tenure": {"$avg": "$tenure"}
                }
            }
        ]

        avg_cursor = customers_collection.aggregate(pipeline)
        averages = await avg_cursor.to_list(length=1)
        avg_data = averages[0] if averages else {}

        return {
            "success": True,
            "data": {
                "total_customers": total,
                "churn_statistics": {
                    "churned": churned,
                    "active": active,
                    "churn_rate_percent": round(churn_rate, 2)
                },
                "contract_distribution": {
                    "month_to_month": month_to_month,
                    "one_year": one_year,
                    "two_year": two_year
                },
                "internet_service_distribution": {
                    "dsl": dsl,
                    "fiber_optic": fiber,
                    "no_internet": no_internet
                },
                "demographics": {
                    "senior_citizens": seniors,
                    "non_seniors": total - seniors
                },
                "averages": {
                    "monthly_charges": round(avg_data.get("avg_monthly_charges", 0), 2),
                    "total_charges": round(avg_data.get("avg_total_charges", 0), 2),
                    "tenure_months": round(avg_data.get("avg_tenure", 0), 1)
                }
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving customer statistics: {str(e)}"
        )