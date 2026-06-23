import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------------------
# Load dataset
# ---------------------------

df = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Drop unnecessary columns
df.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Label Encoding
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

# Models
models = {
    "Logistic Regression": LogisticRegression(C=100, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_split=10),
    "Random Forest": RandomForestClassifier(
        max_depth=5,
        min_samples_split=2,
        n_estimators=50,
        random_state=42
    ),
    "SVM": SVC()
}

# Cross Validation
for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=10,
        scoring='accuracy'
    )

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("Scores:")
    print(scores)

    print("Average Accuracy:", scores.mean())

    print("Standard Deviation:", scores.std())