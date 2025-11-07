"""
Data Processor Module
Handles data loading, preprocessing, and feature detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

class DataProcessor:
    """Process and prepare data for analysis"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.target_column = None
        self.protected_attributes = []
        self.numerical_features = []
        self.categorical_features = []
        self.feature_names = []
        self.encoders = {}
        self.target_encoder = None  # Add target encoder
        self.scaler = StandardScaler()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Auto-detect column types
        self._detect_column_types()
    
    def _detect_column_types(self):
        """Automatically detect numerical and categorical columns with enhanced logic"""
        for col in self.df.columns:
            # Skip if all null
            if self.df[col].isnull().all():
                continue
                
            # Get non-null values for analysis
            non_null_values = self.df[col].dropna()
            
            if len(non_null_values) == 0:
                continue
            
            # Check data type
            if self.df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical despite being numeric
                unique_count = non_null_values.nunique()
                unique_ratio = unique_count / len(non_null_values) if len(non_null_values) > 0 else 0
                
                # Heuristics for categorical detection:
                # 1. Very few unique values (< 10)
                # 2. Low unique ratio (< 5% of total)
                # 3. Binary values (0/1, 1/2, etc.)
                is_binary = unique_count == 2 and set(non_null_values.unique()).issubset({0, 1, 1.0, 0.0, 2, 1, 2.0})
                is_small_discrete = unique_count < 10 and unique_ratio < 0.05
                
                if is_binary or is_small_discrete:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            else:
                # String, object, or category type
                self.categorical_features.append(col)
    
    def _detect_pii_columns(self):
        """Detect potential PII columns"""
        pii_keywords = [
            'name', 'email', 'phone', 'address', 'ssn', 'social',
            'passport', 'license', 'id', 'zip', 'postal'
        ]
        
        pii_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in pii_keywords):
                pii_columns.append(col)
        
        return pii_columns
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for model training with robust handling of edge cases"""
        # Handle missing values - use different strategies based on data type
        print(f"Initial dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Count missing values before handling
        missing_counts = self.df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            print(f"Columns with missing values: {dict(cols_with_missing)}")
        
        # For numerical columns: fill with median
        for col in self.numerical_features:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} missing values with median: {median_val}")
        
        # For categorical columns: fill with mode or 'Unknown'
        for col in self.categorical_features:
            if col in self.df.columns and self.df[col].isnull().any():
                if self.df[col].mode().empty:
                    self.df[col].fillna('Unknown', inplace=True)
                else:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"  Filled {col} missing values with mode: {mode_val}")
        
        # Drop rows with remaining missing values
        rows_before = len(self.df)
        self.df = self.df.dropna()
        rows_dropped = rows_before - len(self.df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with missing values")
        
        # Separate features and target
        if self.target_column is None:
            # Auto-detect target (last column or column with 'target', 'label', 'status')
            target_candidates = [col for col in self.df.columns 
                               if any(keyword in col.lower() for keyword in ['target', 'label', 'status', 'class', 'outcome', 'result'])]
            self.target_column = target_candidates[0] if target_candidates else self.df.columns[-1]
            print(f"Auto-detected target column: {self.target_column}")
        
        # Prepare features
        feature_cols = [col for col in self.df.columns if col != self.target_column]
        X = self.df[feature_cols].copy()
        y = self.df[self.target_column].copy()
        
        # Encode target variable if it's categorical
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
            y = pd.Series(y_encoded, index=y.index, name=self.target_column)
            encoding_map = dict(enumerate(self.target_encoder.classes_))
            print(f"Target '{self.target_column}' encoded: {encoding_map}")
        elif y.dtype in ['float64', 'int64']:
            # Check if numeric target needs binarization
            unique_values = y.unique()
            if len(unique_values) == 2:
                print(f"Binary target detected with values: {sorted(unique_values)}")
                # Ensure 0/1 encoding
                if not set(unique_values).issubset({0, 1}):
                    min_val = min(unique_values)
                    y = (y != min_val).astype(int)
                    print(f"Converted to 0/1 encoding (1 = positive class)")
        
        # Encode categorical variables with better handling
        for col in self.categorical_features:
            if col in X.columns:
                # Handle high cardinality features
                unique_count = X[col].nunique()
                if unique_count > 50:
                    print(f"  ⚠️  High cardinality feature '{col}' ({unique_count} unique values) - consider feature engineering")
                
                le = LabelEncoder()
                # Convert to string to handle mixed types
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                print(f"Encoded '{col}': {unique_count} categories")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"\nTarget distribution:")
        for val, count in class_counts.items():
            print(f"  Class {val}: {count} ({count/len(y)*100:.1f}%)")
        
        # Determine if stratification is needed
        min_class_count = class_counts.min()
        use_stratify = y.nunique() < 10 and min_class_count >= 2
        
        # Split data
        if use_stratify:
            print(f"Using stratified split (min class count: {min_class_count})")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            print(f"Using random split (class imbalance or regression)")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        print(f"Train set: {len(self.X_train)} samples, Test set: {len(self.X_test)} samples")
        
        # Scale numerical features
        numerical_cols = [col for col in self.numerical_features if col in self.X_train.columns]
        if numerical_cols:
            print(f"Scaling {len(numerical_cols)} numerical features")
            self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
            self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_summary(self):
        """Get summary statistics of the dataset"""
        summary = {
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'numerical_features': len(self.numerical_features),
            'categorical_features': len(self.categorical_features),
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_column': self.target_column,
            'protected_attributes': self.protected_attributes,
            'pii_columns': self._detect_pii_columns(),
            'target_distribution': self.df[self.target_column].value_counts().to_dict() if self.target_column else {}
        }
        return summary
    
    def get_protected_attribute_stats(self):
        """Get statistics for protected attributes"""
        stats = {}
        for attr in self.protected_attributes:
            if attr in self.df.columns:
                stats[attr] = {
                    'unique_values': self.df[attr].nunique(),
                    'value_counts': self.df[attr].value_counts().to_dict(),
                    'missing_count': self.df[attr].isnull().sum()
                }
        return stats
