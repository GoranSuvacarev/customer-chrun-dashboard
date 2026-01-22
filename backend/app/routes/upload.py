"""
CSV Upload Route
Handles uploading customer data from CSV files
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Dict, List, Any
from datetime import datetime, timezone

from app.database import get_collection, Collections
from app.models import Customer, Services, CustomerInDB

# Create router
router = APIRouter()

# Required CSV columns
REQUIRED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]


@router.post("/csv")
async def upload_csv(
        file: UploadFile = File(...),
        clear_existing: bool = False
):
    """
    Upload customer data from CSV file

    Args:
        file: CSV file containing customer data
        clear_existing: If True, clears existing data before inserting new data

    Returns:
        JSONResponse with upload results
    """

    # ========================================
    # STEP 1: Validate File
    # ========================================

    # Check file is provided
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Check file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Expected CSV, got {file.filename}"
        )

    # Check file is not empty
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )

    # ========================================
    # STEP 2: Read CSV into Pandas
    # ========================================

    try:
        # Create file-like object from bytes
        csv_buffer = io.BytesIO(contents)

        # Read CSV with pandas
        df = pd.read_csv(csv_buffer)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file is empty or malformed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading CSV file: {str(e)}"
        )

    # ========================================
    # STEP 3: Validate CSV Columns
    # ========================================

    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required columns: {', '.join(missing_columns)}"
        )

    # Check if CSV has data
    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file contains no data rows"
        )

    # ========================================
    # STEP 4: Get MongoDB Collection
    # ========================================

    try:
        customers_collection = get_collection(Collections.CUSTOMERS)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )

    # ========================================
    # STEP 5: Clear Existing Data (Optional)
    # ========================================

    deleted_count = 0
    if clear_existing:
        try:
            delete_result = await customers_collection.delete_many({})
            deleted_count = delete_result.deleted_count
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error clearing existing data: {str(e)}"
            )

    # ========================================
    # STEP 6: Transform and Validate Data
    # ========================================

    # Convert DataFrame to list of dictionaries
    customers_data = df.to_dict('records')

    # Lists to track results
    valid_documents = []
    errors = []

    for idx, customer_dict in enumerate(customers_data, start=1):
        try:
            # Create Services object from service-related columns
            services = Services(
                PhoneService=customer_dict['PhoneService'],
                MultipleLines=customer_dict['MultipleLines'],
                InternetService=customer_dict['InternetService'],
                OnlineSecurity=customer_dict['OnlineSecurity'],
                OnlineBackup=customer_dict['OnlineBackup'],
                DeviceProtection=customer_dict['DeviceProtection'],
                TechSupport=customer_dict['TechSupport'],
                StreamingTV=customer_dict['StreamingTV'],
                StreamingMovies=customer_dict['StreamingMovies']
            )

            # Add services object to customer data
            customer_dict['services'] = services

            # Get current timestamp
            current_time = datetime.now(timezone.utc)

            # Create CustomerInDB object (Pydantic validates automatically)
            customer_in_db = CustomerInDB(
                **customer_dict,
                created_at=current_time,
                updated_at=current_time
            )

            # Convert to MongoDB document (dict)
            customer_doc = customer_in_db.model_dump(by_alias=True, exclude={'id'})

            # Add to valid documents list
            valid_documents.append(customer_doc)

        except Exception as e:
            # Track validation errors
            customer_id = customer_dict.get('customerID', f'Row {idx}')
            errors.append({
                'row': idx,
                'customer_id': customer_id,
                'error': str(e)
            })

    # ========================================
    # STEP 7: Insert/Update into MongoDB (Upsert)
    # ========================================

    inserted_count = 0
    updated_count = 0

    if valid_documents:
        try:
            # Use upsert to update existing or insert new customers
            for document in valid_documents:
                document.pop("created_at", None)
                document.pop("updated_at", None)

                result = await customers_collection.update_one(
                    {"customerID": document["customerID"]},
                    {
                        "$set": document,  # Update all fields (no timestamps)
                        "$setOnInsert": {
                            "created_at": datetime.now(timezone.utc)  # Only on insert
                        },
                        "$currentDate": {
                            "updated_at": True  # Always update to current time
                        }
                    },
                    upsert=True
                )

                if result.upserted_id:
                    inserted_count += 1
                elif result.modified_count > 0:
                    updated_count += 1

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error inserting/updating data in database: {str(e)}"
            )

    # ========================================
    # STEP 8: Calculate Statistics
    # ========================================

    # Get total count and churn statistics
    try:
        total_in_db = await customers_collection.count_documents({})
        churn_yes = await customers_collection.count_documents({"Churn": "Yes"})
        churn_no = await customers_collection.count_documents({"Churn": "No"})
        churn_rate = (churn_yes / total_in_db * 100) if total_in_db > 0 else 0

    except Exception as e:
        # If stats fail, just set defaults
        total_in_db = inserted_count
        churn_yes = 0
        churn_no = 0
        churn_rate = 0.0

    # ========================================
    # STEP 9: Build Response
    # ========================================

    response_data = {
        "success": True,
        "message": f"Successfully processed CSV upload",
        "file_name": file.filename,
        "total_rows_in_csv": len(customers_data),
        "customers_created": inserted_count,
        "customers_updated": updated_count,
        "failed_validations": len(errors),
        "deleted_existing": deleted_count if clear_existing else 0,
        "database_statistics": {
            "total_customers": total_in_db,
            "churned_customers": churn_yes,
            "active_customers": churn_no,
            "churn_rate_percent": round(churn_rate, 2)
        }
    }

    # Add errors if any
    if errors:
        response_data["errors"] = errors[:20]  # Show first 20 errors
        if len(errors) > 20:
            response_data["additional_errors"] = len(errors) - 20

    # Determine HTTP status code
    if inserted_count == 0:
        status_code = status.HTTP_400_BAD_REQUEST
        response_data["success"] = False
        response_data["message"] = "No valid records to insert. All rows failed validation."
    elif errors:
        status_code = status.HTTP_207_MULTI_STATUS  # Partial success
        response_data["message"] = f"Partial success: {inserted_count} inserted, {len(errors)} failed"
    else:
        status_code = status.HTTP_201_CREATED  # Complete success

    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


@router.get("/status")
async def get_upload_status():
    """
    Get current database status

    Returns:
        Database statistics
    """
    try:
        customers_collection = get_collection(Collections.CUSTOMERS)

        total = await customers_collection.count_documents({})
        churned = await customers_collection.count_documents({"Churn": "Yes"})
        active = await customers_collection.count_documents({"Churn": "No"})
        churn_rate = (churned / total * 100) if total > 0 else 0

        return {
            "success": True,
            "database_statistics": {
                "total_customers": total,
                "churned_customers": churned,
                "active_customers": active,
                "churn_rate_percent": round(churn_rate, 2)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving database status: {str(e)}"
        )