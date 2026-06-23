import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Drop unnecessary columns
df.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Dictionary to store encoders
encoders = {}

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Features and target
X = df.drop('status', axis=1)
y = df['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    C=100,
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

# Save encoders
pickle.dump(encoders, open("models/encoders.pkl", "wb"))

print("Model, scaler and encoders saved successfully!")