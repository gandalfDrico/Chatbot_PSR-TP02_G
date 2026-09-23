import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# 1. Cargar la base de conocimiento
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []          # todas las palabras que aparecen en los patterns
classes = []        # los tags (categorías de intención)
documents = []      # cada pattern con su tag
ignore_letters = ['?', '¿', '!', '¡', '.', ',', ';', ':']

# 2. Preprocesamiento básico (tokenización manual por espacios)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # pasar a minúsculas y quitar signos de cada palabra
        word_list = [w.lower().strip('?.!,;:¿¡') for w in pattern.split()]
        word_list = [w for w in word_list if w not in ignore_letters]
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Sacar duplicados y ordenar
words = sorted(set(words))
classes = sorted(set(classes))

# Guardar las estructuras para usarlas después en el servidor
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 3. Crear el set de entrenamiento (Bag of Words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Bolsa de palabras: 1 si la palabra de `words` está en el pattern, si no 0
    bag = []
    for word in words:
        bag.append(1) if word in doc[0] else bag.append(0)

    # Fila de salida: 1 en la posición de la categoría que corresponde
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mezclar y convertir a arreglos de NumPy
np.random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

# 4. Arquitectura de la Red Neuronal (MLP secuencial con Keras)
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')  # clasificación multi-clase
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save('chatbot_model.h5')
print("¡Modelo entrenado y guardado con éxito!")
