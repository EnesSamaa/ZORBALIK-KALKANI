# model_egit.py

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

# Veri yolu
DATA_PATH = "dataset.csv"
MAX_LEN = 100
VOCAB_SIZE = 10000
EMBED_DIM = 128

# 1. Veri Yükle
df = pd.read_csv(DATA_PATH)
print("Toplam satır sayısı:", len(df))


# 2. Metin Temizleme
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['clean_text'] = df['tweet_text'].apply(clean_text)

# 3. Etiket Encode
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['cyberbullying_type'])
y = to_categorical(df['label_encoded'])

# 4. Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN)

# 5. Eğitim / Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 7. Eğit
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 8. Kaydet
model.save("model/lstm_model.h5")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model ve tokenizer başarıyla kaydedildi.")
print("Eğitime alınan örnek sayısı:", len(X_train))
print("Test verisi sayısı:", len(X_test))

model.save("model/lstm_model.h5")


# 9. Modeli Değerlendir
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 10. Doğruluğu Dosyaya Yaz
with open("model/accuracy.txt", "w") as f:
    f.write(str(test_accuracy))

# 9.1 Confusion Matrix ve Metrik Hesaplamaları

# 1. Gerçek ve tahmin etiketlerini çıkar
y_true = np.argmax(y_test, axis=1)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Confusion Matrix oluştur
num_classes = y_test.shape[1]
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true, pred in zip(y_true, y_pred):
    conf_matrix[true][pred] += 1

print("\nConfusion Matrix:")
print(conf_matrix)

# 3. Accuracy hesapla
correct = np.sum(y_true == y_pred)
total = len(y_true)
accuracy = correct / total
print(f"\nAccuracy: {accuracy:.4f}")

# 4. Precision & Recall hesapla
print("\nPrecision & Recall per class:")
for i in range(num_classes):
    tp = conf_matrix[i][i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"Class {i} -> Precision: {precision:.4f}, Recall: {recall:.4f}")

print("\nSınıf İsimleri:")
print(label_encoder.classes_)

metrics = {
    "accuracy": accuracy,
    "confusion_matrix": conf_matrix.tolist(),  # NumPy dizisini listeye çeviriyoruz
    "precision_recall": []
}

for i in range(num_classes):
    tp = conf_matrix[i][i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics["precision_recall"].append({
        "class_index": i,
        "class_label": label_encoder.classes_[i],
        "precision": precision,
        "recall": recall
    })

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)







