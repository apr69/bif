import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

genomics_data = pd.read_csv('Genomics.csv')

genomics_data.head()

genomics_data = genomics_data.dropna()


label_encoder = LabelEncoder()
genomics_data['variant_label'] = label_encoder.fit_transform(genomics_data['variant_name'])


X = genomics_data[['specimens', 'percentage', 'specimens_7d_avg', 'percentage_7d_avg']]
y = genomics_data['variant_label']
X.fillna(0, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)


print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_rf))



svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("SVM Accuracy Score:", accuracy_score(y_test, y_pred_svm))

feature_importances = rf_classifier.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance (Random Forest)')
plt.show()

from tabulate import tabulate

rf_accuracy = accuracy_score(y_test, y_pred_rf)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

data = [
    {'Algorithm': 'Random Forest', 'Accuracy': rf_accuracy},
    {'Algorithm': 'SVM', 'Accuracy': svm_accuracy}
]

table = tabulate(data, headers='keys', tablefmt='grid')
print(table)

models = ['Random Forest', 'SVM']
accuracies = [rf_accuracy, svm_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['green', 'blue'])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Tuned Random Forest vs Tuned SVM')
plt.show()
