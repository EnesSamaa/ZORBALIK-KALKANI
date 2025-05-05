import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import json

# Temizleme fonksiyonu (model_egit.py'deki ile aynı olmalı)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Model ve tokenizer yükle
model = load_model("model/lstm_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Veri setini yükle ve hazırla
df = pd.read_csv("dataset.csv")
df['clean_text'] = df['tweet_text'].apply(clean_text)

X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=100)

y = label_encoder.transform(df['cyberbullying_type'])
y = np.eye(len(label_encoder.classes_))[y]  # One-hot encode

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tahmin
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
num_classes = len(label_encoder.classes_)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for t, p in zip(y_true, y_pred):
    confusion_matrix[t, p] += 1

# Accuracy
accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

# Precision ve Recall
precision = []
recall = []

for i in range(num_classes):
    tp = confusion_matrix[i, i]
    fp = np.sum(confusion_matrix[:, i]) - tp
    fn = np.sum(confusion_matrix[i, :]) - tp

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0

    precision.append(p)
    recall.append(r)

# Sonuçları yazdır
print("Confusion Matrix:\n", confusion_matrix)
print(f"Accuracy: {accuracy:.4f}")
print("Precision (per class):", precision)
print("Recall (per class):", recall)

# Sonuçları JSON olarak kaydet
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": [round(p, 4) for p in precision],
    "recall": [round(r, 4) for r in recall],
    "labels": list(label_encoder.classes_)
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrikler 'model/metrics.json' dosyasına kaydedildi.")
