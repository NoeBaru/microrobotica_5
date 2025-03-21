# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Carica il modello di machine learning
model = joblib.load('./es30_fiat500.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prezzi', methods=['POST'])
def prezzi():
    marca = request.form['marca']
    modello = request.form['modello']
    anno = int(request.form['anno'])
    chilometraggio = int(request.form['chilometraggio'])
    
    # Prepara i dati per la previsione
    dati = pd.DataFrame([[marca, modello, anno, chilometraggio]], columns=['marca', 'modello', 'anno', 'chilometraggio'])
    prezzo = model.predict(dati)
    
    return render_template('risultato.html', prezzo=prezzo[0])

if __name__ == '__main__':
    app.run(debug=True)
