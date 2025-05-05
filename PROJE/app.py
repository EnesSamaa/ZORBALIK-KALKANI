from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Flask uygulaması
app = Flask(__name__)

# Modeli ve vectorizer'ı yükle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Ana sayfa
@app.route('/')
def home():
    return render_template('index.html')

# Kullanıcıdan metin alıp tahmin yapma
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']  # Kullanıcıdan alınan metin
        text_tfidf = vectorizer.transform([text])  # Metni vektörleştir
        prediction = model.predict(text_tfidf)  # Tahmin yap
        return render_template('i.html', prediction=prediction[0], text=text)

if __name__ == "__main__":
    app.run(debug=True)


import pickle

# Flask uygulamanla aynı klasörde 'model.pkl' dosyasını aç
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model başarıyla yüklendi!")

import pickle

# Flask uygulamanla aynı klasörde 'vectorizer.pkl' dosyasını aç
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Vectorizer başarıyla yüklendi!")
