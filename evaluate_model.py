import matplotlib.pyplot as plt

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'SVM'
]

accuracies = [
    0.8837,
    0.8372,
    0.7907,
    0.8139
]

plt.figure(figsize=(8,5))

plt.bar(model_names, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.ylim(0.7,1)

plt.show()