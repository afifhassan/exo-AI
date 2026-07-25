import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess_data

def train_model():
    print("⏳ Loading preprocessed data...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_preprocess_data()
    
    print("🤖 Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save artifacts for the Streamlit app
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'encoder.pkl')
    print("💾 Model, Scaler, and Label Encoder saved successfully to disk!")

if __name__ == "__main__":
    train_model()