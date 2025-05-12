import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
api = KaggleApi()
api.authenticate()
api.dataset_download_files('kaushil268/disease-prediction-using-machine-learning', path='.', unzip=True)
df=pd.read_csv('Training.csv')
X=df.drop('prognosis',axis=1)
y=df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print("\n--- Model Accuracy ---")
print("Accuracy:", accuracy_score(y_test,y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test,y_pred))