# emotion_detector.py

import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("emotions.csv")  # Make sure this CSV is in the same folder

# 2. Data Preprocessing
df['clean_text'] = df['text'].apply(nfx.remove_userhandles)
df['clean_text'] = df['clean_text'].apply(nfx.remove_stopwords)
df['clean_text'] = df['clean_text'].apply(nfx.remove_punctuations)

# 3. Split Data
X = df['clean_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build ML Pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression(max_iter=200))
])

# 5. Train the Model
pipe.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipe.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
sns.heatmap(cm, annot=True, xticklabels=pipe.classes_, yticklabels=pipe.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 7. Save Model
dump(pipe, 'emotion_model.joblib')
print("Model saved as emotion_model.joblib")

# 8. Optional: Predict a custom sentence
text = input("Enter a sentence to detect emotion: ")
prediction = pipe.predict([text])
print(f"Predicted Emotion: {prediction[0]}")
