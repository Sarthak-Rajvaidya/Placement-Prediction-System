import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Drop columns
df.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Encoding
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('status', axis=1)
y = df['status']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# XGBoost model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Cross validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=10,
    scoring='accuracy'
)

print("Scores:")
print(scores)

print("\nAverage Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())