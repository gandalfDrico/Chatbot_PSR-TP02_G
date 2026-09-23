# Chatbot Escuela Técnica (PSR-TP03-C2)

Asistente virtual que resuelve dudas sobre trámites administrativos y académicos de la escuela.
Usa una red neuronal (MLP con TensorFlow/Keras) para clasificar la intención del usuario y se
expone como una API REST con Flask para ser consumida por una ventana de chat en HTML.

## Estructura del proyecto

| Archivo | Qué hace |
|---|---|
| `intents.json` | Base de conocimiento: intenciones, ejemplos de preguntas (patterns) y respuestas |
| `train.py` | Script de entrenamiento: preprocesa el texto, entrena la red neuronal y guarda el modelo (`chatbot_model.h5`, `words.pkl`, `classes.pkl`) |
| `app.py` | Servidor Flask que expone el endpoint `POST /chat` y aplica el umbral de certeza del 60% |
| `index.html` | Ventana de chat flotante que se conecta al servidor con `fetch()` |
| `requirements.txt` | Dependencias de Python |

## Requisitos

- Python 3.x
- pip

## Instalación

Desde una terminal, dentro de la carpeta del proyecto:

```bash
# 1. Crear el entorno virtual
python -m venv venv

# 2. Activarlo (Windows)
venv\Scripts\activate

# 3. Instalar las dependencias
pip install -r requirements.txt
```

## Entrenar el modelo

Antes de levantar el servidor hay que entrenar la red neuronal una sola vez:

```bash
python train.py
```

Esto genera los archivos `chatbot_model.h5`, `words.pkl` y `classes.pkl`.
Si modificás `intents.json`, volvé a ejecutar este script.

## Ejecutar el servidor

```bash
python app.py
```

El servidor queda escuchando en `http://localhost:5000`.

## Usar el chatbot

Abrir `index.html` en el navegador (con el servidor corriendo) y escribir en la ventana de chat.

## Demostración de red (testeo de la API con curl)

Con el servidor corriendo, abrir **otra terminal** y ejecutar:

```bash
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"message\": \"Como saco la constancia de alumno?\"}"
```

```bash
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"message\": \"Requisitos para las practicas\"}"
```

Consulta fuera de tema (prueba del fallback, umbral del 60%):

```bash
curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"message\": \"¿Qué tiempo hace en Marte?\"}"
```

En Windows (PowerShell) el comando con comillas simples queda más simple:

```powershell
curl.exe -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"message": "Hola"}'
```

La respuesta siempre viene en formato JSON: `{"response": "..."}`.

## Subir el proyecto a GitHub

```bash
git init
git add .
git commit -m "Chatbot escuela: intents, entrenamiento, servidor y frontend"
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git
git push -u origin main
```
