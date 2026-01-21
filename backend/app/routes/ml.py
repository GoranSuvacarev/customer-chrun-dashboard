"""
ML Routes
Handles machine learning training and prediction endpoints
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any

from app.database import get_collection, Collections
from app.services.ml_service import ChurnPreprocessor, ChurnModelTrainer

# Create router
router = APIRouter()


@router.post("/train")
async def train_model():
    """
    Train a new churn prediction model on all available customer data.

    This endpoint:
    1. Fetches all customers from MongoDB
    2. Validates sufficient data exists
    3. Preprocesses the data
    4. Trains a Random Forest model
    5. Saves model metadata to MongoDB
    6. Returns training metrics and results

    Returns:
        JSONResponse with training results including:
        - metrics (accuracy, precision, recall, F1, ROC-AUC)
        - feature_importance (top features)
        - confusion_matrix
        - model_path
        - training_samples

    Raises:
        HTTPException: For various failure scenarios with detailed error messages
    """

    try:
        print("\n" + "="*70)
        print("🚀 STARTING MODEL TRAINING ENDPOINT")
        print("="*70)

        # ========================================
        # STEP 1: Connect to MongoDB
        # ========================================
        print("\n📡 Step 1: Connecting to MongoDB...")

        try:
            customers_collection = get_collection(Collections.CUSTOMERS)
            metadata_collection = get_collection(Collections.MODEL_METADATA)
            print("   ✅ Connected to MongoDB successfully")
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to database: {str(e)}"
            )

        # ========================================
        # STEP 2: Fetch All Customers
        # ========================================
        print("\n📦 Step 2: Fetching customers from database...")

        try:
            # Fetch all customers from the collection
            cursor = customers_collection.find()
            customers = await cursor.to_list(length=None)

            customer_count = len(customers)
            print(f"   ✅ Fetched {customer_count} customers from database")

        except Exception as e:
            print(f"   ❌ Failed to fetch customers: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch customer data: {str(e)}"
            )

        # ========================================
        # STEP 3: Validate Sufficient Data
        # ========================================
        print("\n🔍 Step 3: Validating data requirements...")

        # Check 1: Do we have any customers?
        if customer_count == 0:
            print("   ❌ No customers found in database")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No customer data available. Please upload customer data first."
            )

        # Check 2: Do we have enough customers for training?
        MIN_CUSTOMERS = 100
        if customer_count < MIN_CUSTOMERS:
            print(f"   ❌ Insufficient data: {customer_count} customers (need {MIN_CUSTOMERS})")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient data for training. Need at least {MIN_CUSTOMERS} customers, but only have {customer_count}. Please upload more data."
            )

        print(f"   ✅ Sufficient data: {customer_count} customers (>= {MIN_CUSTOMERS})")

        # ========================================
        # STEP 4: Check for Historical Churn Data
        # ========================================
        print("\n🎯 Step 4: Validating historical churn data...")

        # Check if customers have the 'Churn' field (historical data)
        # We need historical data to train a supervised model

        churn_field_name = None

        # Check which field name is used (could be 'Churn' or 'churned')
        if 'Churn' in customers[0]:
            churn_field_name = 'Churn'
        elif 'churned' in customers[0]:
            churn_field_name = 'churned'

        if churn_field_name is None:
            print("   ❌ No churn field found in customer data")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Customer data is missing 'Churn' field. Historical churn data is required for training. Please upload a dataset with known churn outcomes."
            )

        print(f"   ✅ Found churn field: '{churn_field_name}'")

        # Check churn distribution
        churn_values = [c.get(churn_field_name) for c in customers]
        churn_count = sum(1 for v in churn_values if v in ['Yes', True, 1, '1'])
        stay_count = sum(1 for v in churn_values if v in ['No', False, 0, '0'])

        print(f"   Distribution:")
        print(f"   - Churned: {churn_count} ({churn_count/customer_count*100:.1f}%)")
        print(f"   - Stayed: {stay_count} ({stay_count/customer_count*100:.1f}%)")

        # Validate we have examples of both classes
        if churn_count == 0:
            print("   ❌ No churned customers in dataset")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No churned customers found in dataset. Need examples of both churned and retained customers for training."
            )

        if stay_count == 0:
            print("   ❌ No retained customers in dataset")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No retained customers found in dataset. Need examples of both churned and retained customers for training."
            )

        # Warn if severely imbalanced (less than 5% of either class)
        if churn_count / customer_count < 0.05:
            print(f"   ⚠️  Warning: Very few churned customers ({churn_count/customer_count*100:.1f}%)")
        if stay_count / customer_count < 0.05:
            print(f"   ⚠️  Warning: Very few retained customers ({stay_count/customer_count*100:.1f}%)")

        print(f"   ✅ Data has both churned and retained customers")

        # ========================================
        # STEP 5: Convert to pandas DataFrame
        # ========================================
        print("\n📊 Step 5: Converting to pandas DataFrame...")

        try:
            df = pd.DataFrame(customers)
            print(f"   ✅ Created DataFrame: {df.shape} (rows, columns)")
            print(f"   Columns: {list(df.columns)[:10]}...")

        except Exception as e:
            print(f"   ❌ Failed to create DataFrame: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to convert data to DataFrame: {str(e)}"
            )

        # ========================================
        # STEP 6: Preprocess Data
        # ========================================
        print("\n🔧 Step 6: Preprocessing data...")

        try:
            # Create preprocessor
            preprocessor = ChurnPreprocessor()

            # Fit and transform the data
            X, y = preprocessor.fit_transform(df)

            print(f"   ✅ Preprocessing complete")
            print(f"   X shape: {X.shape}")
            print(f"   y shape: {y.shape}")
            print(f"   Features used: {len(preprocessor.feature_names)}")

        except ValueError as e:
            print(f"   ❌ Preprocessing validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Data validation error during preprocessing: {str(e)}"
            )
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to preprocess data: {str(e)}"
            )

        # ========================================
        # STEP 7: Train Model
        # ========================================
        print("\n🌲 Step 7: Training Random Forest model...")

        try:
            # Create trainer
            trainer = ChurnModelTrainer(n_estimators=100, random_state=42)

            # Train the model
            training_results = trainer.train(X, y, preprocessor.feature_names)

            print(f"   ✅ Training complete!")
            print(f"   Accuracy: {training_results['metrics']['accuracy']:.1%}")
            print(f"   F1-Score: {training_results['metrics']['f1_score']:.1%}")
            print(f"   ROC-AUC: {training_results['metrics']['roc_auc']:.2f}")

        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model training failed: {str(e)}"
            )

        # ========================================
        # STEP 8: Save Model Metadata to MongoDB
        # ========================================
        print("\n💾 Step 8: Saving model metadata to MongoDB...")

        try:
            # Create metadata document
            metadata_doc = {
                'model_id': training_results['timestamp'],
                'model_filename': training_results['model_filename'],
                'model_path': training_results['model_path'],
                'created_at': datetime.now(timezone.utc),
                'is_active': True,  # Mark as the active model

                # Training info
                'training_samples': training_results['training_samples'],
                'test_samples': training_results['test_samples'],
                'training_time_seconds': training_results['training_time_seconds'],

                # Model configuration
                'n_estimators': training_results['n_estimators'],
                'max_depth': training_results['max_depth'],
                'feature_count': training_results['feature_count'],

                # Performance metrics
                'metrics': {
                    'accuracy': training_results['metrics']['accuracy'],
                    'precision': training_results['metrics']['precision'],
                    'recall': training_results['metrics']['recall'],
                    'f1_score': training_results['metrics']['f1_score'],
                    'roc_auc': training_results['metrics']['roc_auc']
                },

                # Confusion matrix
                'confusion_matrix': training_results['confusion_matrix'],

                # Feature importance (top 10)
                'top_features': dict(
                    sorted(
                        training_results['feature_importance'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                )
            }

            # Deactivate all previous models
            await metadata_collection.update_many(
                {'is_active': True},
                {'$set': {'is_active': False}}
            )

            # Insert new model metadata
            result = await metadata_collection.insert_one(metadata_doc)

            print(f"   ✅ Model metadata saved with ID: {result.inserted_id}")

        except Exception as e:
            print(f"   ❌ Failed to save metadata: {e}")
            # Don't fail the entire request if metadata save fails
            # The model is still trained and saved to disk
            print(f"   ⚠️  Warning: Model trained successfully but metadata save failed")

        # ========================================
        # STEP 9: Prepare Response
        # ========================================
        print("\n📋 Step 9: Preparing response...")

        # Calculate some summary statistics
        top_5_features = dict(
            sorted(
                training_results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        )

        response = {
            'success': True,
            'message': 'Model trained successfully',

            # Training summary
            'training_summary': {
                'total_customers': customer_count,
                'training_samples': training_results['training_samples'],
                'test_samples': training_results['test_samples'],
                'training_time_seconds': training_results['training_time_seconds'],
                'model_saved_at': training_results['model_path']
            },

            # Performance metrics
            'metrics': {
                'accuracy': round(training_results['metrics']['accuracy'] * 100, 2),
                'precision': round(training_results['metrics']['precision'] * 100, 2),
                'recall': round(training_results['metrics']['recall'] * 100, 2),
                'f1_score': round(training_results['metrics']['f1_score'] * 100, 2),
                'roc_auc': round(training_results['metrics']['roc_auc'], 3)
            },

            # Confusion matrix
            'confusion_matrix': training_results['confusion_matrix'],

            # Top 5 most important features
            'top_features': {
                feature: round(importance * 100, 2)
                for feature, importance in top_5_features.items()
            },

            # Model info
            'model_info': {
                'model_id': training_results['timestamp'],
                'n_estimators': training_results['n_estimators'],
                'max_depth': training_results['max_depth'],
                'feature_count': training_results['feature_count']
            }
        }

        print("\n" + "="*70)
        print("✅ MODEL TRAINING ENDPOINT COMPLETE")
        print("="*70)
        print(f"Accuracy: {response['metrics']['accuracy']}%")
        print(f"F1-Score: {response['metrics']['f1_score']}%")
        print(f"ROC-AUC: {response['metrics']['roc_auc']}")
        print(f"Model saved: {training_results['model_path']}")
        print("="*70 + "\n")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response
        )

    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise

    except Exception as e:
        # Catch any unexpected errors
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during training: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info():
    """
    Get information about the currently active model.

    Returns:
        JSONResponse with model metadata including metrics and configuration

    Raises:
        HTTPException: If no model exists or database error occurs
    """

    try:
        # Connect to metadata collection
        metadata_collection = get_collection(Collections.MODEL_METADATA)

        # Find the active model
        active_model = await metadata_collection.find_one({'is_active': True})

        if not active_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No trained model found. Please train a model first using POST /api/train"
            )

        # Convert ObjectId to string
        active_model['_id'] = str(active_model['_id'])

        # Format the response
        response = {
            'success': True,
            'model_info': {
                'model_id': active_model['model_id'],
                'created_at': active_model['created_at'].isoformat(),
                'training_samples': active_model['training_samples'],
                'test_samples': active_model['test_samples'],
                'training_time_seconds': active_model['training_time_seconds']
            },
            'metrics': {
                'accuracy': round(active_model['metrics']['accuracy'] * 100, 2),
                'precision': round(active_model['metrics']['precision'] * 100, 2),
                'recall': round(active_model['metrics']['recall'] * 100, 2),
                'f1_score': round(active_model['metrics']['f1_score'] * 100, 2),
                'roc_auc': round(active_model['metrics']['roc_auc'], 3)
            },
            'confusion_matrix': active_model['confusion_matrix'],
            'top_features': active_model.get('top_features', {}),
            'model_config': {
                'n_estimators': active_model['n_estimators'],
                'max_depth': active_model['max_depth'],
                'feature_count': active_model['feature_count']
            }
        }

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )


@router.post("/predict/{customer_id}")
async def predict_customer_churn(customer_id: str):
    """
    Predict churn probability for a single customer.

    This endpoint:
    1. Fetches the customer from MongoDB
    2. Loads the latest trained model
    3. Makes a churn prediction
    4. Updates the customer document with prediction
    5. Returns prediction results

    Args:
        customer_id: The customer's ID (customerID field)

    Returns:
        JSONResponse with prediction including:
        - customer_id
        - churn_probability (0-100%)
        - risk_level (Low/Medium/High)
        - top_risk_factors
        - will_churn (boolean)

    Raises:
        HTTPException: For various failure scenarios
    """

    try:
        print(f"\n{'='*70}")
        print(f"🔮 PREDICTING CHURN FOR CUSTOMER: {customer_id}")
        print(f"{'='*70}")

        # ========================================
        # STEP 1: Get Customer from MongoDB
        # ========================================
        print("\n📋 Step 1: Fetching customer from database...")

        try:
            customers_collection = get_collection(Collections.CUSTOMERS)

            # Find customer by customerID field
            customer = await customers_collection.find_one({"customerID": customer_id})

            if not customer:
                print(f"   ❌ Customer not found: {customer_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Customer with ID '{customer_id}' not found. Please check the customer ID and try again."
                )

            print(f"   ✅ Customer found: {customer_id}")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Database error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch customer from database: {str(e)}"
            )

        # ========================================
        # STEP 2: Get Latest Trained Model
        # ========================================
        print("\n🤖 Step 2: Loading latest trained model...")

        try:
            metadata_collection = get_collection(Collections.MODEL_METADATA)

            # Find the active model
            model_metadata = await metadata_collection.find_one(
                {"is_active": True},
                sort=[("created_at", -1)]  # Get most recent if multiple active
            )

            if not model_metadata:
                print("   ❌ No trained model found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No trained model found. Please train a model first using POST /api/ml/train"
                )

            model_path = model_metadata['model_path']
            print(f"   ✅ Found model: {model_metadata['model_id']}")
            print(f"   Model path: {model_path}")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Error loading model metadata: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model metadata: {str(e)}"
            )

        # ========================================
        # STEP 3: Verify Model File Exists
        # ========================================
        print("\n📂 Step 3: Verifying model file exists...")

        from pathlib import Path
        model_file = Path(model_path)

        if not model_file.exists():
            print(f"   ❌ Model file not found: {model_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model file not found at {model_path}. The model may have been deleted. Please retrain the model."
            )

        print(f"   ✅ Model file exists: {model_file.name}")

        # ========================================
        # STEP 4: Create Preprocessor and Predictor
        # ========================================
        print("\n🔧 Step 4: Initializing preprocessor and predictor...")

        try:
            # We need to recreate the preprocessor with the same configuration
            # For now, we'll create a fresh preprocessor and fit it on all data
            # (In production, you'd save the preprocessor along with the model)

            # Fetch all customers to fit the preprocessor
            all_customers_cursor = customers_collection.find()
            all_customers = await all_customers_cursor.to_list(length=None)

            if len(all_customers) == 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No customer data available to initialize preprocessor"
                )

            # Convert to DataFrame and fit preprocessor
            df_all = pd.DataFrame(all_customers)
            preprocessor = ChurnPreprocessor()

            # Fit the preprocessor (we need feature names for the predictor)
            try:
                X_temp, y_temp = preprocessor.fit_transform(df_all)
                print(f"   ✅ Preprocessor fitted with {len(preprocessor.feature_names)} features")
            except Exception as e:
                print(f"   ⚠️  Could not fit preprocessor, using transform only")
                # If fit_transform fails (no Churn field), just get feature names another way
                # This is a fallback for when predicting on data without historical churn
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize preprocessor: {str(e)}"
                )

            # Create predictor
            from app.services.ml_service import ChurnPredictor
            predictor = ChurnPredictor(model_path=str(model_file), preprocessor=preprocessor)

            print(f"   ✅ Predictor initialized")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Error initializing predictor: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize predictor: {str(e)}"
            )

        # ========================================
        # STEP 5: Make Prediction
        # ========================================
        print("\n🎯 Step 5: Making prediction...")

        try:
            # Make prediction
            prediction_result = predictor.predict(customer)

            print(f"   ✅ Prediction complete!")
            print(f"   Churn Probability: {prediction_result['churn_probability']:.2f}%")
            print(f"   Risk Level: {prediction_result['risk_level']}")

        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

        # ========================================
        # STEP 6: Update Customer Document
        # ========================================
        print("\n💾 Step 6: Updating customer document with prediction...")

        try:
            # Prepare update document
            update_doc = {
                'churn_probability': prediction_result['churn_probability'],
                'risk_level': prediction_result['risk_level'],
                'will_churn': prediction_result['will_churn'],
                'top_risk_factors': prediction_result['top_risk_factors'],
                'predicted_at': datetime.now(timezone.utc),
                'model_id': model_metadata['model_id']
            }

            # Update the customer document
            await customers_collection.update_one(
                {"customerID": customer_id},
                {"$set": {"prediction": update_doc}}
            )

            print(f"   ✅ Customer document updated")

        except Exception as e:
            print(f"   ⚠️  Warning: Failed to update customer document: {e}")
            # Don't fail the request if update fails - prediction still succeeded
            print(f"   Prediction completed successfully, but document update failed")

        # ========================================
        # STEP 7: Prepare Response
        # ========================================
        print("\n📋 Step 7: Preparing response...")

        response = {
            'success': True,
            'message': 'Prediction completed successfully',
            'customer_id': customer_id,
            'prediction': {
                'churn_probability': prediction_result['churn_probability'],
                'risk_level': prediction_result['risk_level'],
                'will_churn': prediction_result['will_churn'],
                'predicted_at': prediction_result['predicted_at']
            },
            'top_risk_factors': prediction_result['top_risk_factors'][:5],  # Top 5
            'model_info': {
                'model_id': model_metadata['model_id'],
                'model_accuracy': round(model_metadata['metrics']['accuracy'] * 100, 2)
            }
        }

        print(f"\n{'='*70}")
        print(f"✅ PREDICTION COMPLETE")
        print(f"{'='*70}")
        print(f"Customer: {customer_id}")
        print(f"Risk: {prediction_result['risk_level']} ({prediction_result['churn_probability']:.1f}%)")
        print(f"{'='*70}\n")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response
        )

    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise

    except Exception as e:
        # Catch any unexpected errors
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(e)}"
        )


@router.post("/predict-all")
async def predict_all_customers():
    """
    Predict churn probability for ALL customers in the database.

    This endpoint:
    1. Loads the latest trained model once
    2. Fetches all customers from MongoDB
    3. Makes predictions for each customer
    4. Updates customer documents with predictions (in batches)
    5. Handles errors gracefully for individual customers
    6. Returns summary of successful/failed predictions

    Performance:
    - Processes customers in batches for memory efficiency
    - Uses bulk MongoDB operations for faster updates
    - Handles individual prediction failures gracefully

    Returns:
        JSONResponse with prediction summary:
        - total_customers
        - successful_predictions
        - failed_predictions
        - processing_time
        - errors (if any)

    Raises:
        HTTPException: If critical errors occur (model not found, database errors)
    """

    import time
    start_time = time.time()

    try:
        print(f"\n{'='*70}")
        print(f"🔮 BATCH PREDICTION - PREDICTING ALL CUSTOMERS")
        print(f"{'='*70}")

        # ========================================
        # STEP 1: Load Latest Trained Model
        # ========================================
        print("\n🤖 Step 1: Loading latest trained model...")

        try:
            metadata_collection = get_collection(Collections.MODEL_METADATA)

            # Find the active model
            model_metadata = await metadata_collection.find_one(
                {"is_active": True},
                sort=[("created_at", -1)]
            )

            if not model_metadata:
                print("   ❌ No trained model found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No trained model found. Please train a model first using POST /api/ml/train"
                )

            model_path = model_metadata['model_path']
            print(f"   ✅ Found model: {model_metadata['model_id']}")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Error loading model metadata: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model metadata: {str(e)}"
            )

        # ========================================
        # STEP 2: Verify Model File Exists
        # ========================================
        print("\n📂 Step 2: Verifying model file...")

        from pathlib import Path
        model_file = Path(model_path)

        if not model_file.exists():
            print(f"   ❌ Model file not found: {model_path}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model file not found at {model_path}. Please retrain the model."
            )

        print(f"   ✅ Model file exists")

        # ========================================
        # STEP 3: Get All Customers
        # ========================================
        print("\n📦 Step 3: Fetching all customers from database...")

        try:
            customers_collection = get_collection(Collections.CUSTOMERS)

            # Fetch all customers at once (async operation)
            cursor = customers_collection.find()
            all_customers = await cursor.to_list(length=None)

            total_customers = len(all_customers)

            if total_customers == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No customers found in database. Please upload customer data first."
                )

            print(f"   ✅ Fetched {total_customers} customers")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Database error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch customers: {str(e)}"
            )

        # ========================================
        # STEP 4: Initialize Preprocessor and Predictor
        # ========================================
        print("\n🔧 Step 4: Initializing preprocessor and predictor...")

        try:
            # Fit preprocessor on all data
            df_all = pd.DataFrame(all_customers)
            preprocessor = ChurnPreprocessor()

            # Fit the preprocessor
            try:
                X_temp, y_temp = preprocessor.fit_transform(df_all)
                print(f"   ✅ Preprocessor fitted with {len(preprocessor.feature_names)} features")
            except Exception as e:
                print(f"   ❌ Failed to fit preprocessor: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize preprocessor: {str(e)}"
                )

            # Create predictor (load model once for all predictions)
            from app.services.ml_service import ChurnPredictor
            predictor = ChurnPredictor(model_path=str(model_file), preprocessor=preprocessor)

            print(f"   ✅ Predictor initialized")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Error initializing predictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize predictor: {str(e)}"
            )

        # ========================================
        # STEP 5: Make Predictions for All Customers
        # ========================================
        print(f"\n🎯 Step 5: Making predictions for {total_customers} customers...")
        print(f"   Processing in batches for optimal performance...")

        successful_predictions = 0
        failed_predictions = 0
        prediction_errors = []
        updates_to_perform = []

        # Process customers in batches
        BATCH_SIZE = 100  # Process 100 customers at a time

        for batch_start in range(0, total_customers, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_customers)
            batch = all_customers[batch_start:batch_end]

            print(f"   Processing batch {batch_start//BATCH_SIZE + 1} "
                  f"(customers {batch_start+1}-{batch_end})...")

            for customer in batch:
                customer_id = customer.get('customerID', 'Unknown')

                try:
                    # Make prediction for this customer
                    prediction_result = predictor.predict(customer)

                    # Prepare update document
                    update_doc = {
                        'churn_probability': prediction_result['churn_probability'],
                        'risk_level': prediction_result['risk_level'],
                        'will_churn': prediction_result['will_churn'],
                        'top_risk_factors': prediction_result['top_risk_factors'],
                        'predicted_at': datetime.now(timezone.utc),
                        'model_id': model_metadata['model_id']
                    }

                    # Store update to perform later (batch update)
                    updates_to_perform.append({
                        'filter': {'customerID': customer_id},
                        'update': {'$set': {'prediction': update_doc}}
                    })

                    successful_predictions += 1

                except Exception as e:
                    # Handle individual prediction failure gracefully
                    failed_predictions += 1
                    error_msg = f"Customer {customer_id}: {str(e)}"
                    prediction_errors.append(error_msg)
                    print(f"   ⚠️  Failed to predict for {customer_id}: {e}")
                    continue

        print(f"\n   ✅ Predictions complete!")
        print(f"   Successful: {successful_predictions}")
        print(f"   Failed: {failed_predictions}")

        # ========================================
        # STEP 6: Batch Update MongoDB
        # ========================================
        print(f"\n💾 Step 6: Updating customer documents in MongoDB...")
        print(f"   Using bulk operations for optimal performance...")

        updated_count = 0

        if updates_to_perform:
            try:
                # Perform bulk updates in batches for efficiency
                BULK_BATCH_SIZE = 500

                for i in range(0, len(updates_to_perform), BULK_BATCH_SIZE):
                    bulk_batch = updates_to_perform[i:i + BULK_BATCH_SIZE]

                    # Perform bulk update
                    from pymongo import UpdateOne
                    bulk_operations = [
                        UpdateOne(
                            update['filter'],
                            update['update']
                        )
                        for update in bulk_batch
                    ]

                    result = await customers_collection.bulk_write(bulk_operations)
                    updated_count += result.modified_count

                print(f"   ✅ Updated {updated_count} customer documents")

            except Exception as e:
                print(f"   ⚠️  Warning: Bulk update encountered issues: {e}")
                # Don't fail the entire request - predictions still succeeded

        # ========================================
        # STEP 7: Calculate Statistics
        # ========================================
        processing_time = time.time() - start_time

        print(f"\n📊 Step 7: Generating statistics...")

        # Calculate risk distribution
        risk_distribution = {
            'High': 0,
            'Medium': 0,
            'Low': 0
        }

        for update in updates_to_perform:
            risk_level = update['update']['$set']['prediction']['risk_level']
            risk_distribution[risk_level] += 1

        # ========================================
        # STEP 8: Prepare Response
        # ========================================
        response = {
            'success': True,
            'message': f'Batch prediction completed for {total_customers} customers',

            'summary': {
                'total_customers': total_customers,
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'updated_documents': updated_count,
                'processing_time_seconds': round(processing_time, 2)
            },

            'risk_distribution': risk_distribution,

            'performance': {
                'predictions_per_second': round(successful_predictions / processing_time, 2),
                'average_time_per_prediction_ms': round((processing_time / total_customers) * 1000, 2)
            },

            'model_info': {
                'model_id': model_metadata['model_id'],
                'model_accuracy': round(model_metadata['metrics']['accuracy'] * 100, 2)
            }
        }

        # Add errors if any (limit to first 10 to avoid huge response)
        if prediction_errors:
            response['errors'] = prediction_errors[:10]
            if len(prediction_errors) > 10:
                response['errors_note'] = f"Showing first 10 of {len(prediction_errors)} errors"

        print(f"\n{'='*70}")
        print(f"✅ BATCH PREDICTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total: {total_customers} customers")
        print(f"Success: {successful_predictions}")
        print(f"Failed: {failed_predictions}")
        print(f"Time: {processing_time:.2f} seconds")
        print(f"Rate: {successful_predictions/processing_time:.2f} predictions/second")
        print(f"{'='*70}\n")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during batch prediction: {str(e)}"
        )


@router.post("/predict-all")
async def predict_all_customers():
    """
    Predict churn probability for ALL customers in the database.

    This endpoint:
    1. Loads the latest trained model (once)
    2. Fetches all customers from MongoDB
    3. Makes predictions for each customer
    4. Updates all customer documents with predictions
    5. Handles errors gracefully for individual customers
    6. Returns summary of successful and failed predictions

    This is useful for:
    - Initial prediction run after training
    - Batch updates of all customer risk scores
    - Regular scheduled re-scoring

    Returns:
        JSONResponse with batch prediction results including:
        - total_customers
        - successful_predictions
        - failed_predictions
        - errors (list of failures)
        - processing_time

    Note: This may take several minutes for large datasets
    """

    try:
        from time import time
        start_time = time()

        print(f"\n{'='*70}")
        print(f"🔮 BATCH PREDICTION - PREDICTING ALL CUSTOMERS")
        print(f"{'='*70}")

        # ========================================
        # STEP 1: Get Latest Trained Model
        # ========================================
        print("\n🤖 Step 1: Loading latest trained model...")

        try:
            metadata_collection = get_collection(Collections.MODEL_METADATA)
            customers_collection = get_collection(Collections.CUSTOMERS)

            # Find the active model
            model_metadata = await metadata_collection.find_one(
                {"is_active": True},
                sort=[("created_at", -1)]
            )

            if not model_metadata:
                print("   ❌ No trained model found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No trained model found. Please train a model first using POST /api/ml/train"
                )

            model_path = model_metadata['model_path']
            model_id = model_metadata['model_id']
            print(f"   ✅ Found model: {model_id}")

            # Verify model file exists
            from pathlib import Path
            model_file = Path(model_path)

            if not model_file.exists():
                print(f"   ❌ Model file not found: {model_path}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Model file not found. Please retrain the model."
                )

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Error loading model metadata: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model metadata: {str(e)}"
            )

        # ========================================
        # STEP 2: Fetch All Customers
        # ========================================
        print("\n📦 Step 2: Fetching all customers from database...")

        try:
            cursor = customers_collection.find()
            all_customers = await cursor.to_list(length=None)

            total_customers = len(all_customers)

            if total_customers == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No customers found in database"
                )

            print(f"   ✅ Fetched {total_customers} customers")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Failed to fetch customers: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch customers: {str(e)}"
            )

        # ========================================
        # STEP 3: Initialize Preprocessor and Predictor
        # ========================================
        print("\n🔧 Step 3: Initializing preprocessor and predictor...")

        try:
            # Convert to DataFrame and fit preprocessor
            df_all = pd.DataFrame(all_customers)
            preprocessor = ChurnPreprocessor()

            # Fit the preprocessor
            try:
                X_temp, y_temp = preprocessor.fit_transform(df_all)
                print(f"   ✅ Preprocessor fitted with {len(preprocessor.feature_names)} features")
            except Exception as e:
                print(f"   ⚠️  Preprocessor fit failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize preprocessor: {str(e)}"
                )

            # Create predictor (loads model once)
            from app.services.ml_service import ChurnPredictor
            predictor = ChurnPredictor(model_path=str(model_file), preprocessor=preprocessor)
            predictor.load_model()  # Load model once for all predictions

            print(f"   ✅ Predictor initialized and model loaded")

        except HTTPException:
            raise
        except Exception as e:
            print(f"   ❌ Error initializing predictor: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize predictor: {str(e)}"
            )

        # ========================================
        # STEP 4: Predict for Each Customer
        # ========================================
        print(f"\n🎯 Step 4: Making predictions for {total_customers} customers...")
        print(f"   (This may take a few minutes...)")

        successful_predictions = 0
        failed_predictions = 0
        errors = []
        updates = []  # Batch updates for MongoDB

        # Progress tracking
        milestone = max(1, total_customers // 10)  # Print every 10%

        for idx, customer in enumerate(all_customers, 1):
            customer_id = customer.get('customerID', f'Unknown_{idx}')

            try:
                # Make prediction
                prediction_result = predictor.predict(customer)

                # Prepare update document
                update_doc = {
                    'churn_probability': prediction_result['churn_probability'],
                    'risk_level': prediction_result['risk_level'],
                    'will_churn': prediction_result['will_churn'],
                    'top_risk_factors': prediction_result['top_risk_factors'],
                    'predicted_at': datetime.now(timezone.utc),
                    'model_id': model_id
                }

                # Add to batch updates
                updates.append({
                    'customer_id': customer_id,
                    'update': update_doc
                })

                successful_predictions += 1

                # Progress update
                if idx % milestone == 0 or idx == total_customers:
                    progress = (idx / total_customers) * 100
                    print(f"   Progress: {idx}/{total_customers} ({progress:.0f}%) - "
                          f"{successful_predictions} successful, {failed_predictions} failed")

            except Exception as e:
                failed_predictions += 1
                error_msg = f"Customer {customer_id}: {str(e)}"
                errors.append(error_msg)
                print(f"   ⚠️  Failed: {error_msg}")

                # Don't fail entire batch for individual errors
                continue

        print(f"   ✅ Predictions complete: {successful_predictions} successful, {failed_predictions} failed")

        # ========================================
        # STEP 5: Batch Update MongoDB
        # ========================================
        print(f"\n💾 Step 5: Updating customer documents in database...")

        updated_count = 0
        update_errors = 0

        try:
            # Update in batches for efficiency
            batch_size = 100

            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]

                # Use bulk write for efficiency
                from pymongo import UpdateOne
                bulk_operations = [
                    UpdateOne(
                        {"customerID": update['customer_id']},
                        {"$set": {"prediction": update['update']}}
                    )
                    for update in batch
                ]

                try:
                    result = await customers_collection.bulk_write(bulk_operations)
                    updated_count += result.modified_count
                except Exception as e:
                    print(f"   ⚠️  Batch update error: {e}")
                    update_errors += len(batch)

            print(f"   ✅ Updated {updated_count} customer documents")

            if update_errors > 0:
                print(f"   ⚠️  {update_errors} updates failed")

        except Exception as e:
            print(f"   ⚠️  Warning: Batch update failed: {e}")
            # Don't fail the entire request - predictions were successful
            print(f"   Predictions completed but document updates failed")

        # ========================================
        # STEP 6: Calculate Statistics
        # ========================================
        print(f"\n📊 Step 6: Calculating statistics...")

        # Calculate risk distribution
        risk_distribution = {
            'High': 0,
            'Medium': 0,
            'Low': 0
        }

        for update in updates:
            risk_level = update['update']['risk_level']
            risk_distribution[risk_level] += 1

        processing_time = time() - start_time

        print(f"   High Risk: {risk_distribution['High']} ({risk_distribution['High']/successful_predictions*100:.1f}%)")
        print(f"   Medium Risk: {risk_distribution['Medium']} ({risk_distribution['Medium']/successful_predictions*100:.1f}%)")
        print(f"   Low Risk: {risk_distribution['Low']} ({risk_distribution['Low']/successful_predictions*100:.1f}%)")

        # ========================================
        # STEP 7: Prepare Response
        # ========================================
        response = {
            'success': True,
            'message': f'Batch prediction completed for {total_customers} customers',

            'summary': {
                'total_customers': total_customers,
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'updated_in_database': updated_count,
                'processing_time_seconds': round(processing_time, 2)
            },

            'risk_distribution': {
                'high': risk_distribution['High'],
                'medium': risk_distribution['Medium'],
                'low': risk_distribution['Low']
            },

            'model_info': {
                'model_id': model_id,
                'model_accuracy': round(model_metadata['metrics']['accuracy'] * 100, 2)
            }
        }

        # Add errors if any (limit to first 10 to avoid huge response)
        if errors:
            response['errors'] = errors[:10]
            if len(errors) > 10:
                response['errors'].append(f"... and {len(errors) - 10} more errors")

        print(f"\n{'='*70}")
        print(f"✅ BATCH PREDICTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total: {total_customers} | Success: {successful_predictions} | Failed: {failed_predictions}")
        print(f"Time: {processing_time:.2f}s | Avg: {processing_time/total_customers:.3f}s per customer")
        print(f"High Risk: {risk_distribution['High']} | Medium: {risk_distribution['Medium']} | Low: {risk_distribution['Low']}")
        print(f"{'='*70}\n")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )