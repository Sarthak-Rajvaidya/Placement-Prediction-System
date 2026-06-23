import pandas as pd

# Import preprocessing tools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Drop unnecessary columns
df.drop(['sl_no', 'salary'], axis=1, inplace=True)

# -------------------------------
# Label Encoding
# -------------------------------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -------------------------------
# Separate Features and Target
# -------------------------------
X = df.drop('status', axis=1)
y = df['status']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Models Dictionary
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

# -------------------------------
# Train and Evaluate Models
# -------------------------------
for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Print Results
    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


import pickle

# Train the best model
best_model = LogisticRegression()

best_model.fit(X_train, y_train)

# Save model
pickle.dump(best_model, open("models/model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")