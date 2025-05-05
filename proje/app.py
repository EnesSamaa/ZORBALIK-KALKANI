from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Metrikleri oku
with open("model/metrics.json", "r") as f:
    metrics = json.load(f)


# Model dosyalarını yükle
model = tf.keras.models.load_model("model/lstm_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("model/accuracy.txt", "r") as f:
    model_accuracy = float(f.read())

MAX_LEN = 100

# Temizleme fonksiyonu
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tahmin fonksiyonu
def predict_text(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)
    return label_encoder.inverse_transform([np.argmax(pred)])[0]

# Flask uygulaması
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_text = ""
    if request.method == "POST":
        user_text = request.form["text"]
        prediction = predict_text(user_text)

    return render_template("index.html", prediction=prediction, accuracy=model_accuracy, metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)

from evaluation import calculate_confusion_matrix, calculate_precision_recall, calculate_accuracy
