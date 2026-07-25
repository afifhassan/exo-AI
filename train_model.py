import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from preprocess import load_and_preprocess_data

def train_exoai():
    print("⏳ Loading preprocessed data...")
    # Load processed data
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_preprocess_data("koi_data.csv")
    
    # Define hyperparameter configuration
    print("🤖 Initializing Random Forest with specified hyperparameters...")
    rf_model = RandomForestClassifier(
        n_estimators=100,      # 100 Estimators
        max_depth=15,          # Max Depth: 15
        criterion='gini',      # Gini Impurity
        random_state=42
    )
    
    # 5-fold Cross-Validation
    print("📊 Running 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"🎯 5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")
    
    # Train final model
    rf_model.fit(X_train, y_train)
    
    # Save model and preprocessors for the web interface
    joblib.dump(rf_model, "exoai_rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    
    print("💾 Model and preprocessing artifacts saved successfully!")

if __name__ == "__main__":
    train_exoai()