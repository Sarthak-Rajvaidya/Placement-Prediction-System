import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Drop unnecessary columns
df.drop(['sl_no', 'salary'], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('status', axis=1)
y = df['status']

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X, y)

# Get feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Sort by absolute coefficient value
importance['Absolute Coefficient'] = importance['Coefficient'].abs()

importance = importance.sort_values(
    by='Absolute Coefficient',
    ascending=False
)

print(importance)

# Plot
plt.figure(figsize=(10,6))

sns.barplot(
    x='Absolute Coefficient',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")
plt.show()