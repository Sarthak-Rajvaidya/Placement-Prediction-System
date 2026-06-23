import pandas as pd

df = pd.read_csv("data/Placement_Data_Full_Class.csv")

print(df.head())

df.drop(['sl_no', 'salary'], axis=1, inplace=True)

print(df.columns)


print(df.dtypes)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print(df.head())

X = df.drop('status', axis=1)

y = df['status']

print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)