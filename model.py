# model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    # Load dataset
    df = pd.read_csv('task_dataset.csv')
    
    # Feature selection (using the most relevant features)
    features = ['execution_time', 'cpu_req', 'memory_req', 'storage_req', 'data_transfer_size']
    X = df[features]
    y = df['priority']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(rf, 'task_priority_model.pkl')
    
    return rf

def predict_priority(execution_time, cpu_req, memory_req, storage_req, data_transfer_size):
    # Load model
    try:
        model = joblib.load('task_priority_model.pkl')
    except:
        model = train_model()
    
    # Create input array
    input_data = [[execution_time, cpu_req, memory_req, storage_req, data_transfer_size]]
    
    # Predict
    priority = model.predict(input_data)[0]
    probability = max(model.predict_proba(input_data)[0])
    
    return priority, probability

if __name__ == "__main__":
    train_model()
