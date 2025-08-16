# 1) Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

# 2) Read dataset
data = pd.read_csv('data.csv')

# Define features and target
x = data[["Memory_GB", "Bandwidth_Mbps", "Time_Remaining_Hrs"]]
y = data["Task_Name"]

# 3) Analyze dataset
print("\nDataset Analysis:")
print("Shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())
print("\nTarget distribution:")
print(y.value_counts())

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=0.2, random_state=42
)

# 4) Train models and store accuracy
models = [
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('SVM', make_pipeline(StandardScaler(), SVC(random_state=42)))
]

accuracies = []
trained_models = []

for name, model in models:
    # Train model
    model.fit(x_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # 5) Store model
    filename = f"{name.replace(' ', '_').lower()}_model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    trained_models.append(model)
    
    # 6) Print accuracy
    print(f"\n{name} Accuracy: {acc:.4f}")

# 7) Visualize accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar([m[0] for m in models], accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
