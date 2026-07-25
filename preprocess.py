import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path="koi_data.csv"):
    # Load dataset (comment='#' skips any NASA metadata comment lines)
    df = pd.read_csv(file_path, comment='#')
    
    # Select features and target
    features = ['koi_period', 'koi_duration', 'koi_prad', 'koi_depth']
    target = 'koi_disposition'  # Options: CONFIRMED, CANDIDATE, FALSE POSITIVE
    
    # 1. Clean data: Drop missing values
    df_clean = df.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]
    
    # 2. Encode target labels to numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 3. Split Dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, le = load_and_preprocess_data()
    print("\n🎉 Preprocessing Complete!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")
    print(f"Target Classes:     {dict(zip(le.classes_, range(len(le.classes_))))}")