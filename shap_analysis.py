import pandas as pd
import shap

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/Placement_Data_Full_Class.csv")

df.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Encoding
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('status', axis=1)
y = df['status']

# Train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# SHAP explainer
explainer = shap.Explainer(model)

shap_values = explainer(X)

# Summary plot
shap.plots.beeswarm(shap_values)