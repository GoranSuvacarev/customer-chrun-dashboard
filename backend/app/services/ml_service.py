"""
Machine Learning Service for Customer Churn Prediction

This module contains three main classes:
1. ChurnPreprocessor - Handles data preparation and transformation
2. ChurnModelTrainer - Handles model training and evaluation
3. ChurnPredictor - Handles making predictions on new customer data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
from datetime import datetime
from pathlib import Path


class ChurnPreprocessor:
    """
    Handles all data preprocessing and transformation for the churn model.

    This class is responsible for:
    - Flattening nested service data
    - Encoding categorical variables (state, gender, contract_type, etc.)
    - Scaling numeric features (age, tenure, charges)
    - Handling missing values
    - Creating feature matrix (X) and target variable (y)
    """

    def __init__(self):
        """
        Initialize the preprocessor with empty scalers and encoders.
        These will be fitted during the fit_transform step.
        """
        # StandardScaler: Transforms numeric features to mean=0, std=1
        # Why? ML models perform better when features are on similar scales
        # Example: tenure (0-72) and MonthlyCharges (0-120) → both scaled to ~(-2 to 2)
        self.scaler = StandardScaler()

        # LabelEncoders: Convert categorical text to numbers
        # We need separate encoders for each categorical column
        # Example: gender ('Male', 'Female') → (0, 1)
        self.label_encoders = {}

        # Store the feature names after they're determined
        # This ensures we always use the same features in the same order
        self.feature_names = None

        # Store categorical and numeric column names
        # We'll identify these during fit_transform
        self.categorical_columns = []
        self.numeric_columns = []

    def _get_feature_names(self, df):
        """
        Determine which columns from the dataset will be used as features.

        Args:
            df: pandas DataFrame with customer data

        Returns:
            List of column names to use as features

        Notes:
            - Excludes: _id, customer_id, churned (target), prediction data
            - Includes: all numeric and categorical customer attributes
            - Includes: flattened service fields
        """
        # Columns to EXCLUDE (not useful for prediction)
        exclude_columns = [
            '_id',              # MongoDB document ID
            'customerID',       # Just an identifier, not predictive
            'Churn',            # This is our TARGET variable (what we're predicting)
            'churned',          # Alternative name for target
            'churn_probability', # Prediction output (if exists)
            'risk_level',       # Prediction output (if exists)
            'predicted_at',     # Prediction metadata (if exists)
            'top_risk_factors', # Prediction metadata (if exists)
            'prediction'        # Prediction output column (data leakage!)
        ]

        # Get all column names from the DataFrame
        all_columns = df.columns.tolist()

        # Filter out excluded columns
        # List comprehension: keep column if it's NOT in exclude_columns
        feature_columns = [
            col for col in all_columns
            if col not in exclude_columns
        ]

        # Log what we're using (helpful for debugging)
        print(f"📊 Using {len(feature_columns)} features for model training")
        print(f"Features: {feature_columns[:10]}...")  # Show first 10

        return feature_columns

    def _flatten_services(self, df):
        """
        Convert nested 'services' dictionary into flat columns.

        Args:
            df: pandas DataFrame with 'services' column

        Returns:
            DataFrame with services flattened into separate boolean columns

        Example:
            Before: services: {internet: true, phone: false}
            After: service_internet: 1, service_phone: 0
        """
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()

        # Check if 'services' column exists (it might already be flattened)
        if 'services' not in df_copy.columns:
            # Services are already flat columns - nothing to do
            print("ℹ️  Services already flattened, skipping...")
            return df_copy

        # Extract the services dictionary for each row
        # pd.json_normalize flattens nested dictionaries into columns
        # Example: {'PhoneService': 'Yes', 'InternetService': 'DSL'}
        #       -> PhoneService: 'Yes', InternetService: 'DSL'
        services_df = pd.json_normalize(df_copy['services'])

        # Add the flattened service columns to the main DataFrame
        # concat combines DataFrames side-by-side (axis=1 means columns)
        df_copy = pd.concat([df_copy, services_df], axis=1)

        # Remove the original nested 'services' column (we don't need it anymore)
        df_copy = df_copy.drop('services', axis=1)

        print(f"✅ Flattened services into {len(services_df.columns)} columns")
        print(f"   Service columns: {list(services_df.columns)[:5]}...")

        return df_copy

    def _handle_missing_values(self, df):
        """
        Fill or remove missing values in the dataset.

        Args:
            df: pandas DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled

        Strategy:
            - Numeric columns: fill with median or 0
            - Categorical columns: fill with mode or 'Unknown'
        """
        df_copy = df.copy()

        # Count missing values before handling
        missing_before = df_copy.isnull().sum().sum()

        if missing_before > 0:
            print(f"⚠️  Found {missing_before} missing values")

            # Strategy 1: Fill numeric columns with median
            # Median is better than mean because it's not affected by outliers
            # Example: tenure has values [1, 2, 3, 100] -> median=2.5 (mean would be 26.5)
            numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                if df_copy[col].isnull().any():
                    median_value = df_copy[col].median()
                    # If median is NaN (all values missing), use 0
                    fill_value = median_value if pd.notna(median_value) else 0
                    df_copy[col] = df_copy[col].fillna(fill_value)
                    print(f"   Filled {col} with {fill_value}")

            # Strategy 2: Fill categorical columns with mode (most common value)
            # Example: If most customers have 'Electronic check', use that
            categorical_columns = df_copy.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df_copy[col].isnull().any():
                    # Get the mode (most frequent value)
                    mode_value = df_copy[col].mode()
                    # If mode exists, use it; otherwise use 'Unknown'
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    df_copy[col] = df_copy[col].fillna(fill_value)
                    print(f"   Filled {col} with '{fill_value}'")

            missing_after = df_copy.isnull().sum().sum()
            print(f"✅ Missing values reduced from {missing_before} to {missing_after}")
        else:
            print("✅ No missing values found")

        return df_copy

    def _encode_categorical(self, df, fit=True):
        """
        Convert categorical variables (strings) to numeric codes.

        Args:
            df: pandas DataFrame with categorical columns
            fit: If True, fit the encoders. If False, use existing encoders

        Returns:
            DataFrame with categorical columns encoded as numbers

        Columns to encode:
            - state (e.g., 'CA' -> 5, 'NY' -> 33)
            - gender (e.g., 'Male' -> 1, 'Female' -> 0)
            - contract_type (e.g., 'Monthly' -> 0, 'Annual' -> 1)
            - payment_method (e.g., 'Credit Card' -> 0)
        """
        df_copy = df.copy()

        # Identify categorical columns (object/string type)
        # These need to be converted to numbers for ML
        categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()

        # Exclude certain columns that shouldn't be encoded
        exclude_from_encoding = ['customerID', '_id', 'Churn', 'churned']
        categorical_columns = [col for col in categorical_columns if col not in exclude_from_encoding]

        if fit:
            # TRAINING MODE: Create and fit new encoders
            print(f"🔢 Encoding {len(categorical_columns)} categorical columns...")

            for col in categorical_columns:
                # Create a new LabelEncoder for this column
                # LabelEncoder converts strings to numbers (alphabetically)
                # Example: ['Female', 'Male', 'Male'] -> [0, 1, 1]
                encoder = LabelEncoder()

                # Fit the encoder on this column's unique values
                # This teaches it: 'Female' = 0, 'Male' = 1
                df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))

                # Store the encoder for later use during prediction
                self.label_encoders[col] = encoder

                print(f"   {col}: {len(encoder.classes_)} unique values encoded")

            # Store which columns are categorical for later
            self.categorical_columns = categorical_columns

        else:
            # PREDICTION MODE: Use existing encoders
            print(f"🔢 Encoding using existing encoders...")

            for col in self.categorical_columns:
                if col in df_copy.columns:
                    # Use the pre-fitted encoder from training
                    encoder = self.label_encoders[col]

                    # Handle unknown categories (values not seen during training)
                    # Example: Training had 'Male', 'Female' but new data has 'Other'
                    try:
                        df_copy[col] = encoder.transform(df_copy[col].astype(str))
                    except ValueError:
                        # If we encounter unknown values, map them to the first class
                        print(f"   ⚠️  Unknown values in {col}, using default encoding")
                        df_copy[col] = df_copy[col].apply(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_
                            else encoder.transform([encoder.classes_[0]])[0]
                        )

        print(f"✅ Categorical encoding complete")
        return df_copy

    def _scale_numeric(self, df, fit=True):
        """
        Standardize numeric features to similar scales.

        Args:
            df: pandas DataFrame with numeric columns
            fit: If True, fit the scaler. If False, use existing scaler

        Returns:
            DataFrame with numeric columns scaled

        Uses StandardScaler:
            - Transforms data to mean=0, std=1
            - Example: age 45 -> 0.23, monthly_charges 85 -> 0.67
        """
        df_copy = df.copy()

        # Identify numeric columns (int64, float64)
        # These are columns with numbers that need to be scaled
        numeric_columns = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Exclude certain columns that shouldn't be scaled
        # SeniorCitizen is already 0/1, so doesn't need scaling
        exclude_from_scaling = ['SeniorCitizen']
        numeric_columns = [col for col in numeric_columns if col not in exclude_from_scaling]

        if len(numeric_columns) == 0:
            print("ℹ️  No numeric columns to scale")
            return df_copy

        if fit:
            # TRAINING MODE: Fit the scaler
            print(f"📏 Scaling {len(numeric_columns)} numeric columns...")

            # StandardScaler transforms each column to:
            # - Mean = 0 (center the data)
            # - Standard deviation = 1 (scale the data)
            #
            # Formula: (value - mean) / std_deviation
            # Example for MonthlyCharges:
            #   Original: [20, 50, 80, 100]
            #   Mean: 62.5, Std: 32.5
            #   Scaled: [-1.31, -0.38, 0.54, 1.15]
            #
            # WHY? ML models work better when features are on similar scales
            # tenure (0-72) and MonthlyCharges (18-118) → both become ~(-2 to 2)

            # Fit and transform the numeric columns
            df_copy[numeric_columns] = self.scaler.fit_transform(df_copy[numeric_columns])

            # Store which columns are numeric for later
            self.numeric_columns = numeric_columns

            print(f"   Scaled columns: {numeric_columns[:5]}...")
            print(f"✅ Numeric scaling complete")

        else:
            # PREDICTION MODE: Use existing scaler
            print(f"📏 Scaling using existing scaler...")

            # Transform using the SAME mean and std from training
            # This ensures new data is on the same scale as training data
            # Example: If training mean was 62.5, we use that same value
            df_copy[self.numeric_columns] = self.scaler.transform(df_copy[self.numeric_columns])

            print(f"✅ Numeric scaling complete")

        return df_copy

    def fit_transform(self, df):
        """
        Fit the preprocessor on training data and transform it.

        This is used during model training.

        Args:
            df: pandas DataFrame with customer data including 'churned' column

        Returns:
            X: Feature matrix (2D numpy array)
            y: Target vector (1D numpy array) - 0 or 1

        Steps:
            1. Flatten services
            2. Handle missing values
            3. Separate features and target
            4. Encode categorical (fit=True)
            5. Scale numeric (fit=True)
            6. Return X and y
        """
        print("\n" + "="*60)
        print("🔧 FITTING PREPROCESSOR ON TRAINING DATA")
        print("="*60)

        # STEP 1: Flatten nested services into separate columns
        # MongoDB: services: {PhoneService: "Yes", ...}
        # After: PhoneService: "Yes", InternetService: "DSL", ...
        print("\n📦 Step 1: Flattening services...")
        df = self._flatten_services(df)

        # STEP 2: Handle any missing values (NaN)
        # Fills missing data so model doesn't break
        print("\n🔍 Step 2: Handling missing values...")
        df = self._handle_missing_values(df)

        # STEP 3: Separate features (X) from target variable (y)
        # Target is what we're predicting: did the customer churn?
        print("\n🎯 Step 3: Separating features and target...")

        # The target column can be named 'Churn' or 'churned'
        target_column = 'Churn' if 'Churn' in df.columns else 'churned'

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract the target (y)
        # Convert 'Yes'/'No' to 1/0 for the ML model
        # 'Yes' = customer churned (1), 'No' = customer stayed (0)
        y = df[target_column].apply(lambda x: 1 if x == 'Yes' else 0).values

        # Remove target and non-feature columns from X
        # These columns are not useful for prediction
        columns_to_drop = [
            target_column, 'churned', 'Churn',  # Target variables
            'customerID', '_id',                # Identifiers
            'churn_probability', 'risk_level',  # Prediction outputs
            'predicted_at', 'top_risk_factors', # Prediction metadata
            'prediction',                       # Prediction column (data leakage!)
            'created_at', 'updated_at'          # Timestamps
        ]

        # Drop columns that exist in the DataFrame
        df_features = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors='ignore'
        )

        # Store feature names for later use
        # This ensures predictions use the same features in the same order
        self.feature_names = df_features.columns.tolist()

        print(f"   Features: {len(self.feature_names)} columns")
        print(f"   Target: {len(y)} samples ({y.sum()} churned, {len(y) - y.sum()} stayed)")

        # STEP 4: Encode categorical variables (text → numbers)
        # fit=True means CREATE NEW encoders and LEARN the mappings
        # Example: gender: 'Male' → 1, 'Female' → 0
        print("\n🔢 Step 4: Encoding categorical features...")
        df_features = self._encode_categorical(df_features, fit=True)

        # STEP 5: Scale numeric features (standardize to mean=0, std=1)
        # fit=True means CALCULATE mean and std from THIS data
        # Example: MonthlyCharges: 85.50 → 0.34
        print("\n📏 Step 5: Scaling numeric features...")
        df_features = self._scale_numeric(df_features, fit=True)

        # Convert final DataFrame to numpy array (what ML models expect)
        X = df_features.values

        print("\n" + "="*60)
        print(f"✅ PREPROCESSING COMPLETE!")
        print(f"   X shape: {X.shape} (samples, features)")
        print(f"   y shape: {y.shape} (samples,)")
        print(f"   Features: {self.feature_names[:5]}...")
        print("="*60 + "\n")

        return X, y

    def transform(self, df):
        """
        Transform new data using already-fitted preprocessor.

        This is used during prediction.

        Args:
            df: pandas DataFrame with customer data (no 'churned' column needed)

        Returns:
            X: Feature matrix (2D numpy array)

        Steps:
            1. Flatten services
            2. Handle missing values
            3. Encode categorical (fit=False, use existing encoders)
            4. Scale numeric (fit=False, use existing scaler)
            5. Return X
        """
        print("\n" + "="*60)
        print("🔄 TRANSFORMING NEW DATA FOR PREDICTION")
        print("="*60)

        # Check that the preprocessor has been fitted first
        if self.feature_names is None:
            raise ValueError(
                "Preprocessor has not been fitted yet. "
                "Call fit_transform() on training data first."
            )

        # STEP 1: Flatten nested services (same as training)
        print("\n📦 Step 1: Flattening services...")
        df = self._flatten_services(df)

        # STEP 2: Handle missing values (same as training)
        print("\n🔍 Step 2: Handling missing values...")
        df = self._handle_missing_values(df)

        # STEP 3: Select only the feature columns (no target needed)
        # Remove identifiers and metadata
        columns_to_drop = [
            'churned', 'Churn',                 # Target (if present)
            'customerID', '_id',                # Identifiers
            'churn_probability', 'risk_level',  # Old predictions
            'predicted_at', 'top_risk_factors', # Old metadata
            'prediction',                       # Prediction column (data leakage!)
            'created_at', 'updated_at'          # Timestamps
        ]

        df_features = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors='ignore'
        )

        # STEP 4: Encode categorical using EXISTING encoders
        # fit=False means USE the encoders we created during training
        # This ensures new data is encoded the SAME WAY as training data
        # Example: If 'Male' was 1 during training, it's still 1 now
        print("\n🔢 Step 3: Encoding categorical features...")
        df_features = self._encode_categorical(df_features, fit=False)

        # STEP 5: Scale numeric using EXISTING scaler
        # fit=False means USE the mean/std we calculated during training
        # This ensures new data is on the SAME SCALE as training data
        # Example: If training mean was 62.5, we use that same value
        print("\n📏 Step 4: Scaling numeric features...")
        df_features = self._scale_numeric(df_features, fit=False)

        # Ensure columns are in the same order as training
        # ML models are sensitive to feature order!
        # If training was [age, tenure, charges], prediction must be too
        try:
            df_features = df_features[self.feature_names]
        except KeyError as e:
            missing_cols = set(self.feature_names) - set(df_features.columns)
            extra_cols = set(df_features.columns) - set(self.feature_names)
            error_msg = f"Feature mismatch!\n"
            if missing_cols:
                error_msg += f"Missing columns: {missing_cols}\n"
            if extra_cols:
                error_msg += f"Extra columns: {extra_cols}\n"
            raise ValueError(error_msg)

        # Convert to numpy array
        X = df_features.values

        print("\n" + "="*60)
        print(f"✅ TRANSFORMATION COMPLETE!")
        print(f"   X shape: {X.shape} (samples, features)")
        print(f"   Features: {self.feature_names[:5]}...")
        print("="*60 + "\n")

        return X


class ChurnModelTrainer:
    """
    Handles training and evaluating the Random Forest churn prediction model.

    This class is responsible for:
    - Training a Random Forest classifier
    - Evaluating model performance (accuracy, precision, recall, etc.)
    - Calculating feature importances
    - Saving the trained model to disk
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the model trainer with a Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest (default: 100)
            random_state: Random seed for reproducibility (default: 42)

        Hyperparameters explained:
            - n_estimators: More trees = better performance but slower
            - max_depth: Maximum tree depth (prevents overfitting)
            - min_samples_split: Minimum samples to split a node
            - random_state: Ensures same results each run
        """

        # Initialize Random Forest Classifier with carefully chosen hyperparameters
        self.model = RandomForestClassifier(
            # ==================================================
            # CORE HYPERPARAMETERS
            # ==================================================

            n_estimators=n_estimators,
            # Number of decision trees in the forest
            # More trees = better performance but slower training
            # 100 is a good balance between accuracy and speed
            # Each tree "votes" on the final prediction

            random_state=random_state,
            # Random seed for reproducibility
            # Same seed = same results every time you train
            # Important for debugging and comparing models

            # ==================================================
            # TREE STRUCTURE HYPERPARAMETERS
            # ==================================================

            max_depth=10,
            # Maximum depth of each tree (how many splits)
            # Deeper trees = more complex patterns BUT risk overfitting
            # 10 levels is good for this dataset size
            # Example: Level 1: "tenure < 12?", Level 2: "Monthly > 70?", etc.

            min_samples_split=20,
            # Minimum samples required to split a node
            # Higher = simpler trees, less overfitting
            # 20 means "don't split unless at least 20 customers in this node"
            # Prevents creating splits based on just a few outliers

            min_samples_leaf=10,
            # Minimum samples required in a leaf node (final decision)
            # Higher = more conservative predictions
            # 10 means "each final prediction must represent at least 10 customers"
            # Prevents overfitting to individual cases

            # ==================================================
            # FEATURE SAMPLING HYPERPARAMETERS
            # ==================================================

            max_features='sqrt',
            # Number of features to consider at each split
            # 'sqrt' = consider √19 ≈ 4 features at each split
            # Why random subset? Makes trees more diverse/independent
            # More diversity = better ensemble performance
            # Options: 'sqrt' (default), 'log2', None (use all), or integer

            # ==================================================
            # PERFORMANCE HYPERPARAMETERS
            # ==================================================

            n_jobs=-1,
            # Number of CPU cores to use for training
            # -1 = use all available cores (parallel processing)
            # Makes training much faster on multi-core systems
            # Example: 4 cores = ~4x faster training

            class_weight='balanced',
            # How to weight the classes (churned vs stayed)
            # 'balanced' = automatically adjust weights inversely proportional to frequencies
            # Important because we have imbalanced data (24% churned, 76% stayed)
            # Without this, model might just predict "will stay" for everyone
            # Balanced = gives churners more importance during training

            # ==================================================
            # OUTPUT SETTINGS
            # ==================================================

            verbose=0
            # 0 = silent, 1 = progress messages, 2 = detailed logs
            # We'll add our own logging, so keep this silent
        )

        print("🌲 Random Forest Classifier initialized")
        print(f"   Trees: {n_estimators}")
        print(f"   Max depth: {self.model.max_depth}")
        print(f"   Min samples split: {self.model.min_samples_split}")
        print(f"   Class weighting: {self.model.class_weight}")
        print(f"   Feature sampling: {self.model.max_features}")
        print(f"   Parallel cores: All available (n_jobs=-1)")

    def _split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction to use for testing (default: 0.2 = 20%)

        Returns:
            X_train, X_test, y_train, y_test

        Uses stratification to maintain class balance:
            - If 30% of customers churned in full dataset
            - Then 30% will churn in both train and test sets
        """

        print("\n📊 Splitting data into train and test sets...")

        # ==================================================
        # WHAT IS TRAIN/TEST SPLIT?
        # ==================================================
        # We split data into two parts:
        # 1. TRAINING SET (80%): Model learns from this
        # 2. TEST SET (20%): Model is evaluated on this (never seen before)
        #
        # Why? To check if model generalizes to NEW data
        # If we test on training data, model might just "memorize" answers

        # ==================================================
        # WHAT IS STRATIFICATION?
        # ==================================================
        # Stratify = maintain the same class distribution in both sets
        #
        # Example WITHOUT stratification:
        # Full data: 70% stayed, 30% churned
        # Train: 80% stayed, 20% churned  ← Imbalanced!
        # Test: 50% stayed, 50% churned    ← Very different!
        #
        # Example WITH stratification:
        # Full data: 70% stayed, 30% churned
        # Train: 70% stayed, 30% churned  ← Same ratio!
        # Test: 70% stayed, 30% churned   ← Same ratio!
        #
        # Why important? Ensures consistent evaluation

        # Calculate class distribution before split
        total_samples = len(y)
        churned_count = np.sum(y == 1)
        stayed_count = np.sum(y == 0)
        churn_rate = churned_count / total_samples * 100

        print(f"   Total samples: {total_samples}")
        print(f"   Churned: {churned_count} ({churn_rate:.1f}%)")
        print(f"   Stayed: {stayed_count} ({100-churn_rate:.1f}%)")

        # Perform stratified split
        # stratify=y means: keep same churn rate in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,      # 20% for testing
            random_state=42,          # Reproducibility
            stratify=y                # KEY: Maintain class distribution!
        )

        # Calculate split sizes
        train_samples = len(y_train)
        test_samples = len(y_test)

        # Calculate class distribution after split
        train_churn_rate = np.sum(y_train == 1) / len(y_train) * 100
        test_churn_rate = np.sum(y_test == 1) / len(y_test) * 100

        print(f"\n   ✅ Split complete:")
        print(f"   Training set: {train_samples} samples ({train_samples/total_samples*100:.0f}%)")
        print(f"      - Churned: {np.sum(y_train == 1)} ({train_churn_rate:.1f}%)")
        print(f"      - Stayed: {np.sum(y_train == 0)} ({100-train_churn_rate:.1f}%)")
        print(f"   Test set: {test_samples} samples ({test_samples/total_samples*100:.0f}%)")
        print(f"      - Churned: {np.sum(y_test == 1)} ({test_churn_rate:.1f}%)")
        print(f"      - Stayed: {np.sum(y_test == 0)} ({100-test_churn_rate:.1f}%)")

        # Verify stratification worked (churn rates should be nearly identical)
        if abs(train_churn_rate - test_churn_rate) < 5:
            print(f"   ✅ Stratification successful (rates within 5%)")
        else:
            print(f"   ⚠️  Warning: Stratification may not be perfect")

        return X_train, X_test, y_train, y_test

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate all evaluation metrics for the model.

        Args:
            y_true: Actual labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities (0.0 to 1.0)

        Returns:
            Dictionary with all metrics:
                - accuracy: Overall correctness
                - precision: When we predict churn, how often are we right?
                - recall: Of actual churners, how many did we catch?
                - f1_score: Balance of precision and recall
                - roc_auc: How well we separate churners from non-churners
        """

        # ==================================================
        # METRIC 1: ACCURACY
        # ==================================================
        # "What percentage of predictions were correct?"
        # Formula: (Correct predictions) / (Total predictions)
        # Example: 85 out of 100 predictions correct = 85% accuracy
        #
        # LIMITATION: Misleading with imbalanced data
        # If 90% customers stay, predicting "all stay" gives 90% accuracy
        # but catches 0% of churners (useless!)
        accuracy = accuracy_score(y_true, y_pred)

        # ==================================================
        # METRIC 2: PRECISION
        # ==================================================
        # "When we predict a customer will churn, how often are we right?"
        # Formula: True Positives / (True Positives + False Positives)
        #
        # Example:
        # - We predicted 30 customers will churn
        # - 24 actually did churn (True Positives)
        # - 6 didn't churn (False Positives - false alarms)
        # - Precision = 24/30 = 80%
        #
        # HIGH PRECISION = Few false alarms
        # Important when: Cost of false alarms is high
        # (e.g., don't want to waste retention offers on loyal customers)
        precision = precision_score(y_true, y_pred, zero_division=0)

        # ==================================================
        # METRIC 3: RECALL (Sensitivity)
        # ==================================================
        # "Of all customers who actually churned, how many did we catch?"
        # Formula: True Positives / (True Positives + False Negatives)
        #
        # Example:
        # - 40 customers actually churned
        # - We caught 32 of them (True Positives)
        # - We missed 8 of them (False Negatives)
        # - Recall = 32/40 = 80%
        #
        # HIGH RECALL = Catch most churners
        # Important when: Cost of missing churners is high
        # (e.g., losing a customer is expensive)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # ==================================================
        # METRIC 4: F1-SCORE
        # ==================================================
        # "Harmonic mean of precision and recall"
        # Formula: 2 × (Precision × Recall) / (Precision + Recall)
        #
        # Why harmonic mean? Penalizes extreme values
        # - Precision=100%, Recall=10% → F1=18% (not 55%)
        # - Precision=80%, Recall=80% → F1=80%
        #
        # USE WHEN: You want balance between precision and recall
        # Good single metric for model comparison
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ==================================================
        # METRIC 5: ROC-AUC (Area Under ROC Curve)
        # ==================================================
        # "How well can the model separate churners from non-churners?"
        # Range: 0.5 (random guessing) to 1.0 (perfect)
        #
        # Interpretation:
        # - 0.5 = Random coin flip (terrible)
        # - 0.7 = Acceptable
        # - 0.8 = Good
        # - 0.9 = Excellent
        # - 1.0 = Perfect (rare, might be overfitting)
        #
        # Why useful? Works well even with imbalanced classes
        # Measures model's ability to rank churners higher than non-churners
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

    def _get_feature_importance(self, feature_names):
        """
        Extract which features matter most to the model.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores

        Example output:
            {
                'monthly_charges': 0.25,  # 25% importance
                'tenure_months': 0.18,     # 18% importance
                'contract_type': 0.15,     # 15% importance
                ...
            }
        """

        # Random Forest tracks how much each feature helps reduce impurity
        # Impurity = how mixed the classes are at a node
        #
        # Example: At a tree node
        # - Before split: 50 churners, 50 stayers (50% churn rate)
        # - After splitting on "tenure < 12":
        #   - Left: 40 churners, 10 stayers (80% churn rate)
        #   - Right: 10 churners, 40 stayers (20% churn rate)
        # - This split REDUCED impurity significantly
        # - tenure gets credit for being useful!
        #
        # Importance = average reduction in impurity across all trees

        # Get importance scores from the trained model
        # This is calculated automatically during training
        importances = self.model.feature_importances_

        # Create dictionary mapping feature names to importance scores
        feature_importance_dict = {}
        for feature_name, importance in zip(feature_names, importances):
            feature_importance_dict[feature_name] = float(importance)

        # Verify importances sum to 1.0 (they should)
        total = sum(feature_importance_dict.values())
        assert abs(total - 1.0) < 0.01, f"Feature importances should sum to 1.0, got {total}"

        return feature_importance_dict

    def _save_model(self, model_path):
        """
        Save the trained model to disk using joblib.

        Args:
            model_path: Path where model file should be saved

        Saves:
            - The trained RandomForestClassifier
            - Can be loaded later for predictions
        """

        # joblib is better than pickle for ML models because:
        # 1. More efficient with large numpy arrays
        # 2. Better compression
        # 3. Faster loading
        # 4. Industry standard for scikit-learn models

        try:
            joblib.dump(self.model, model_path)
            # Verify file was created
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file was not created at {model_path}")
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")

    def train(self, X, y, feature_names):
        """
        Complete training pipeline - train, evaluate, and save model.

        Args:
            X: Feature matrix from preprocessor
            y: Target vector from preprocessor
            feature_names: List of feature column names

        Returns:
            Dictionary with training results:
                - metrics (accuracy, precision, recall, f1, roc_auc)
                - feature_importance (which features matter most)
                - confusion_matrix (breakdown of predictions)
                - model_path (where model was saved)
                - training_samples (how many customers used)

        Steps:
            1. Split data into train/test
            2. Train Random Forest on training data
            3. Make predictions on test data
            4. Calculate all metrics
            5. Get feature importances
            6. Save model to models/ folder
            7. Return results
        """

        print("\n" + "="*70)
        print("🚀 STARTING MODEL TRAINING")
        print("="*70)

        # ========================================
        # STEP 1: Split Data into Train/Test
        # ========================================
        X_train, X_test, y_train, y_test = self._split_data(X, y, test_size=0.2)

        # ========================================
        # STEP 2: Train the Random Forest Model
        # ========================================
        print("\n🌲 Training Random Forest model...")
        print(f"   Training on {len(X_train)} samples...")

        # This is where the magic happens!
        # The model learns patterns from the training data
        # Each of the 100 trees learns different aspects
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"   ✅ Training complete in {training_time:.2f} seconds")

        # ========================================
        # STEP 3: Make Predictions on Test Set
        # ========================================
        print("\n🔮 Making predictions on test set...")

        # Predict class labels (0 or 1)
        # 0 = will stay, 1 = will churn
        y_pred = self.model.predict(X_test)

        # Predict probabilities [prob_of_0, prob_of_1]
        # Example: [0.3, 0.7] means 70% chance of churn
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Get probability of class 1 (churn)

        print(f"   ✅ Predictions complete for {len(X_test)} samples")

        # ========================================
        # STEP 4: Calculate Performance Metrics
        # ========================================
        print("\n📊 Calculating performance metrics...")

        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        print(f"\n   RESULTS:")
        print(f"   {'Metric':<20} {'Score':<10} {'Explanation'}")
        print(f"   {'-'*60}")
        print(f"   {'Accuracy':<20} {metrics['accuracy']:.1%}      Overall correctness")
        print(f"   {'Precision':<20} {metrics['precision']:.1%}      When we predict churn, how often right?")
        print(f"   {'Recall':<20} {metrics['recall']:.1%}      Of actual churners, how many caught?")
        print(f"   {'F1-Score':<20} {metrics['f1_score']:.1%}      Balance of precision & recall")
        print(f"   {'ROC-AUC':<20} {metrics['roc_auc']:.1%}      Ability to separate classes")

        # ========================================
        # STEP 5: Get Feature Importances
        # ========================================
        print("\n🎯 Analyzing feature importance...")

        feature_importance = self._get_feature_importance(feature_names)

        # Show top 10 most important features
        print(f"\n   Top 10 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            bar_length = int(importance * 50)  # Visual bar
            bar = '█' * bar_length
            print(f"   {i:2}. {feature:<25} {importance:>6.1%} {bar}")

        # ========================================
        # STEP 6: Create Confusion Matrix
        # ========================================
        print("\n📋 Creating confusion matrix...")

        # Confusion matrix shows breakdown of predictions
        cm = confusion_matrix(y_test, y_pred)

        # Extract values from confusion matrix
        # [[TN, FP],
        #  [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        print(f"\n   Confusion Matrix:")
        print(f"   ┌─────────────────────────────────┐")
        print(f"   │           PREDICTED             │")
        print(f"   │         Stay  │  Churn          │")
        print(f"   ├─────────┼─────────┼─────────────┤")
        print(f"   │ ACTUAL  │         │             │")
        print(f"   │ Stay    │  {tn:4}   │   {fp:4}       │ (True Negatives: {tn}, False Positives: {fp})")
        print(f"   │ Churn   │  {fn:4}   │   {tp:4}       │ (False Negatives: {fn}, True Positives: {tp})")
        print(f"   └─────────┴─────────┴─────────────┘")
        print(f"\n   Interpretation:")
        print(f"   - True Negatives (TN): {tn} - Correctly predicted 'will stay'")
        print(f"   - True Positives (TP): {tp} - Correctly predicted 'will churn'")
        print(f"   - False Positives (FP): {fp} - Predicted churn but stayed (false alarm)")
        print(f"   - False Negatives (FN): {fn} - Predicted stay but churned (missed)")

        # ========================================
        # STEP 7: Save Model to Disk
        # ========================================
        print("\n💾 Saving trained model...")

        # Create models directory if it doesn't exist
        MODEL_DIR.mkdir(exist_ok=True)

        # Generate unique model filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"churn_model_{timestamp}.pkl"
        model_path = MODEL_DIR / model_filename

        # Save the trained model using joblib
        # joblib is efficient for large numpy arrays (better than pickle for ML models)
        self._save_model(str(model_path))

        print(f"   ✅ Model saved to: {model_path}")

        # ========================================
        # STEP 8: Return All Results
        # ========================================
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)

        # Package all results into a dictionary
        results = {
            # Performance metrics
            'metrics': metrics,

            # Feature importance rankings
            'feature_importance': feature_importance,

            # Confusion matrix breakdown
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },

            # Model metadata
            'model_path': str(model_path),
            'model_filename': model_filename,
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'training_time_seconds': round(training_time, 2),
            'timestamp': timestamp,

            # Model configuration
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'feature_count': len(feature_names)
        }

        return results


class ChurnPredictor:
    """
    Handles making churn predictions on new customer data.

    This class is responsible for:
    - Loading a saved model from disk
    - Preprocessing new customer data
    - Making churn predictions
    - Determining risk levels (Low/Medium/High)
    - Identifying top risk factors for each customer
    """

    def __init__(self, model_path, preprocessor):
        """
        Initialize the predictor with a saved model and preprocessor.

        Args:
            model_path: Path to the saved model file (.pkl)
            preprocessor: Fitted ChurnPreprocessor instance
        """
        self.model_path = model_path
        self.preprocessor = preprocessor
        self.model = None

        # Validate inputs
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        if preprocessor.feature_names is None:
            raise ValueError(
                "Preprocessor has not been fitted. "
                "Please fit the preprocessor before creating a predictor."
            )

        print(f"🔮 ChurnPredictor initialized")
        print(f"   Model path: {model_path}")
        print(f"   Features: {len(preprocessor.feature_names)}")

    def load_model(self):
        """
        Load the trained model from disk.

        Returns:
            Loaded RandomForestClassifier

        Raises:
            FileNotFoundError if model file doesn't exist
        """
        try:
            print(f"📂 Loading model from {self.model_path}...")

            # Load the model using joblib
            # joblib is efficient for large numpy arrays (scikit-learn standard)
            self.model = joblib.load(self.model_path)

            # Verify it's a valid model
            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded object is not a valid model (missing predict method)")

            # Verify it's trained
            if not hasattr(self.model, 'n_features_in_'):
                raise ValueError("Model appears to be untrained")

            print(f"   ✅ Model loaded successfully")
            print(f"   Model type: {type(self.model).__name__}")
            print(f"   Number of trees: {self.model.n_estimators}")
            print(f"   Expected features: {self.model.n_features_in_}")

            return self.model

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def _determine_risk_level(self, probability):
        """
        Convert churn probability to risk category.

        Args:
            probability: Churn probability (0.0 to 1.0)

        Returns:
            'Low', 'Medium', or 'High'

        Thresholds:
            - Low: < 30% chance of churn
            - Medium: 30% - 70% chance of churn
            - High: > 70% chance of churn
        """

        # Convert probability to percentage for easier understanding
        percentage = probability * 100

        # Determine risk level based on thresholds
        # These thresholds can be adjusted based on business needs

        if probability < 0.30:
            # LOW RISK: < 30% chance of churning
            # Customer is likely to stay
            # Action: Monitor normally, no immediate intervention needed
            risk_level = "Low"

        elif probability < 0.70:
            # MEDIUM RISK: 30-70% chance of churning
            # Customer is uncertain, could go either way
            # Action: Proactive outreach, consider light retention offer
            risk_level = "Medium"

        else:
            # HIGH RISK: > 70% chance of churning
            # Customer is likely to churn
            # Action: Immediate intervention, strong retention offer
            risk_level = "High"

        return risk_level

    def _get_top_risk_factors(self, customer_data, feature_importances):
        """
        Identify which factors contribute most to this customer's churn risk.

        Args:
            customer_data: Dictionary with customer attributes
            feature_importances: Dictionary from trained model

        Returns:
            List of top 5 risk factors with their values

        Example output:
            [
                {'factor': 'monthly_charges', 'value': 95.50, 'importance': 0.25},
                {'factor': 'contract_type', 'value': 'Monthly', 'importance': 0.18},
                ...
            ]
        """

        # ==================================================
        # HOW RISK FACTORS ARE DETERMINED
        # ==================================================
        #
        # We combine two pieces of information:
        # 1. FEATURE IMPORTANCE: Which features matter most to the model overall?
        #    (e.g., tenure is 18.5% important, contract is 12.8% important)
        #
        # 2. CUSTOMER VALUES: What are this customer's actual values?
        #    (e.g., this customer has tenure=2 months, contract=monthly)
        #
        # We identify features that are BOTH:
        # - High importance (model cares about them)
        # - Risky values (customer has concerning values)
        #
        # Example:
        # Feature: tenure
        # Importance: 18.5% (very important!)
        # Customer value: 2 months (very low - risky!)
        # → This is a TOP risk factor

        risk_factors = []

        # Sort features by importance (most important first)
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get top features (those that matter most to the model)
        # We look at more than 5 to have options, then filter
        top_features = sorted_features[:15]

        for feature_name, importance in top_features:
            # Get the customer's value for this feature
            # Handle both direct keys and nested structures

            # Try to get value from customer data
            if feature_name in customer_data:
                value = customer_data[feature_name]
            elif 'services' in customer_data and feature_name in customer_data['services']:
                value = customer_data['services'][feature_name]
            else:
                # Feature might have been created during preprocessing
                # (e.g., encoded values) - skip if not in raw data
                continue

            # Determine if this value is "risky"
            # Different strategies for different feature types
            is_risky = self._is_risky_value(feature_name, value)

            # Only include if it's actually a risk factor for this customer
            if is_risky:
                risk_factors.append({
                    'factor': feature_name,
                    'value': value,
                    'importance': importance,
                    'explanation': self._get_risk_explanation(feature_name, value)
                })

        # Return top 5 risk factors
        # Sort by importance to prioritize what matters most
        risk_factors.sort(key=lambda x: x['importance'], reverse=True)
        return risk_factors[:5]

    def _is_risky_value(self, feature_name, value):
        """
        Determine if a feature value is risky (contributes to churn).

        Args:
            feature_name: Name of the feature
            value: Value of the feature for this customer

        Returns:
            Boolean indicating if this value is risky
        """

        # ==================================================
        # RISKY VALUE PATTERNS (Based on Domain Knowledge)
        # ==================================================

        # Tenure: Low tenure is risky (new customers churn more)
        if feature_name == 'tenure':
            return value < 12  # Less than 1 year is risky

        # Monthly Charges: High charges are risky
        if 'Monthly' in feature_name and 'Charges' in feature_name:
            return value > 70  # Above $70/month is risky

        # Contract: Month-to-month is risky (no commitment)
        if feature_name == 'Contract':
            return value == 'Month-to-month'

        # Payment Method: Electronic check is risky (easier to cancel)
        if feature_name == 'PaymentMethod':
            return value == 'Electronic check'

        # Internet Service: Fiber optic can be risky (expensive)
        if feature_name == 'InternetService':
            return value == 'Fiber optic'

        # Services: "No" for add-on services indicates low engagement
        if feature_name in ['OnlineSecurity', 'OnlineBackup', 'TechSupport',
                           'DeviceProtection', 'StreamingTV', 'StreamingMovies']:
            return value == 'No' or value == 'No internet service'

        # Paperless Billing: "Yes" can be risky (less engagement)
        if feature_name == 'PaperlessBilling':
            return value == 'Yes'

        # Default: Consider it a potential risk if we got here
        # (Model thought it was important)
        return True

    def _get_risk_explanation(self, feature_name, value):
        """
        Generate a human-readable explanation for why this is a risk factor.

        Args:
            feature_name: Name of the feature
            value: Value of the feature

        Returns:
            String explanation
        """

        explanations = {
            'tenure': f"Short tenure ({value} months) - new customers churn more",
            'MonthlyCharges': f"High monthly charges (${value}) - price sensitive",
            'Contract': f"{value} contract - no long-term commitment",
            'PaymentMethod': f"{value} - easier to cancel",
            'InternetService': f"{value} service - higher expectations",
            'OnlineSecurity': "No online security - low service engagement",
            'OnlineBackup': "No online backup - low service engagement",
            'TechSupport': "No tech support - low service engagement",
            'DeviceProtection': "No device protection - low service engagement",
            'PaperlessBilling': "Paperless billing - less engagement with statements"
        }

        return explanations.get(feature_name, f"{feature_name}: {value}")

    def predict(self, customer_data):
        """
        Make a churn prediction for a single customer.

        Args:
            customer_data: Dictionary with customer attributes

        Returns:
            Dictionary with prediction results:
                - customer_id: Customer identifier
                - churn_probability: Probability of churn (0-100%)
                - risk_level: 'Low', 'Medium', or 'High'
                - top_risk_factors: List of key factors driving the prediction
                - predicted_at: Timestamp of prediction

        Steps:
            1. Convert customer_data to DataFrame
            2. Preprocess using fitted preprocessor
            3. Load model and make prediction
            4. Get probability score
            5. Determine risk level
            6. Identify top risk factors
            7. Return prediction object
        """

        print("\n" + "="*70)
        print("🔮 MAKING CHURN PREDICTION")
        print("="*70)

        # ========================================
        # STEP 1: Validate Input
        # ========================================
        if not isinstance(customer_data, dict):
            raise ValueError("customer_data must be a dictionary")

        # Extract customer ID (if available)
        customer_id = customer_data.get('customerID', customer_data.get('_id', 'unknown'))

        print(f"\n📋 Customer ID: {customer_id}")

        # ========================================
        # STEP 2: Convert to DataFrame
        # ========================================
        # The preprocessor expects a pandas DataFrame
        # Even though we have just one customer, we need a DataFrame with 1 row

        df = pd.DataFrame([customer_data])
        print(f"   Converted to DataFrame: {df.shape}")

        # ========================================
        # STEP 3: Preprocess Data
        # ========================================
        print("\n🔧 Preprocessing customer data...")

        # Use the FITTED preprocessor (same one used during training)
        # This ensures consistent transformation
        X = self.preprocessor.transform(df)

        print(f"   Preprocessed shape: {X.shape}")
        print(f"   Features: {len(self.preprocessor.feature_names)}")

        # ========================================
        # STEP 4: Load Model (if not already loaded)
        # ========================================
        if self.model is None:
            self.load_model()

        # ========================================
        # STEP 5: Make Prediction
        # ========================================
        print("\n🎯 Making prediction...")

        # Get probability scores [prob_stay, prob_churn]
        # We want prob_churn (second value)
        probabilities = self.model.predict_proba(X)
        churn_probability = probabilities[0][1]  # Probability of class 1 (churn)

        # Get binary prediction (0 or 1)
        prediction = self.model.predict(X)[0]

        print(f"   Churn probability: {churn_probability:.1%}")
        print(f"   Binary prediction: {'Will churn' if prediction == 1 else 'Will stay'}")

        # ========================================
        # STEP 6: Determine Risk Level
        # ========================================
        risk_level = self._determine_risk_level(churn_probability)

        print(f"   Risk level: {risk_level}")

        # ========================================
        # STEP 7: Get Feature Importances from Model
        # ========================================
        # Extract feature importances from the trained model
        feature_importances = {}
        for feature_name, importance in zip(
            self.preprocessor.feature_names,
            self.model.feature_importances_
        ):
            feature_importances[feature_name] = importance

        # ========================================
        # STEP 8: Identify Top Risk Factors
        # ========================================
        print("\n🔍 Analyzing risk factors...")

        top_risk_factors = self._get_top_risk_factors(customer_data, feature_importances)

        if top_risk_factors:
            print(f"\n   Top {len(top_risk_factors)} Risk Factors:")
            for i, factor in enumerate(top_risk_factors, 1):
                print(f"   {i}. {factor['factor']}: {factor['value']}")
                print(f"      → {factor['explanation']}")
        else:
            print("   No specific risk factors identified")

        # ========================================
        # STEP 9: Create Prediction Object
        # ========================================
        prediction_result = {
            'customer_id': str(customer_id),
            'churn_probability': round(float(churn_probability * 100), 2),  # Convert to percentage
            'risk_level': risk_level,
            'will_churn': bool(prediction == 1),
            'top_risk_factors': top_risk_factors,
            'predicted_at': datetime.now().isoformat(),
            'model_path': str(self.model_path)
        }

        print("\n" + "="*70)
        print("✅ PREDICTION COMPLETE")
        print("="*70)
        print(f"   Customer: {customer_id}")
        print(f"   Churn Risk: {churn_probability:.1%} ({risk_level})")
        print(f"   Prediction: {'⚠️  WILL CHURN' if prediction == 1 else '✅ WILL STAY'}")
        print("="*70 + "\n")

        return prediction_result


# Model file paths
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)