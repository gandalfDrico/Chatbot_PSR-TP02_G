import json
import pickle

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite solicitudes desde el frontend de la escuela

# Cargar el modelo de IA y las estructuras de datos (generados por train.py)
model = tf.keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

ERROR_THRESHOLD = 0.60  # Umbral mínimo de certeza (RF-04)


def clean_up_sentence(sentence):
    """Divide la oración en palabras en minúsculas y sin signos."""
    return [w.lower().strip('?.!,;:¿¡') for w in sentence.split()]


def bow(sentence, words):
    """Convierte la oración en un Bag of Words (vector de 0 y 1)."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag, dtype=float)


# RUTA DE RED PRINCIPAL (API REST)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "No enviaste ningún mensaje."}), 400

    # 1. Predecir con la red neuronal
    p = bow(user_message, words)
    res = model.predict(np.array([p]), verbose=0)[0]

    # 2. Filtrar por umbral de certeza (RF-04)
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    # 3. Determinar respuesta o aplicar Fallback
    if results:
        tag = classes[results[0][0]]
        response_text = "Lo siento, no entendí tu consulta."
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response_text = intent['responses'][0]
                break
    else:
        # Mensaje por defecto si la red neuronal no está segura
        response_text = ("Lo siento, no logré entender tu consulta sobre el trámite. "
                         "Por favor, comunícate con Secretaría al correo "
                         "secretaria@escuelatecnica.edu.ar.")

    return jsonify({"response": response_text})


if __name__ == '__main__':
    # El servidor corre en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
