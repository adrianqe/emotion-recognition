from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import os
import tensorflow as tf

app = Flask(__name__)

# Configuración
EMOTIONS = ['Enojado', 'Disgusto', 'Miedo',
            'Feliz', 'Neutral', 'Triste', 'Sorpresa']
EMOJIS = ['😠', '🤢', '😨', '😊', '😐', '😢', '😮']
IMG_SIZE = 64

# Cargar modelo
MODEL_PATH = os.path.join('..', 'models', 'emotion-recognition-model.h5')
print(f"Cargando modelo desde: {MODEL_PATH}")

try:
    print(f"TensorFlow versión: {tf.__version__}")
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado, descargando...")
        os.makedirs('../models', exist_ok=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id='adrianqe/cnn-emotion-recognition-73',
            filename='emotion-recognition-model.h5',
            local_dir='../models'
        )

    model = load_model(MODEL_PATH)
    MODEL_INFO = {"name": "CNN Optimizado", "accuracy": "73.78%"}
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    MODEL_INFO = {"name": "Error", "accuracy": "N/A"}
    model = None

# Detector de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predicción de imagen desde el navegador"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo no disponible'
            })

        # Recibir imagen en base64
        data = request.json
        image_data = data['image'].split(',')[1]

        # Decodificar imagen
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            return jsonify({
                'success': True,
                'faces_detected': 0,
                'message': 'No se detectó ningún rostro'
            })

        results = []

        for (x, y, w, h) in faces:
            # Extraer y preprocesar rostro
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized / 255.0
            face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # Predecir
            predictions = model.predict(face_input, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[emotion_idx] * 100)
            emotion = EMOTIONS[emotion_idx]
            emoji = EMOJIS[emotion_idx]

            results.append({
                'emotion': emotion,
                'emoji': emoji,
                'confidence': confidence,
                'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'probabilities': {
                    EMOTIONS[i]: float(predictions[i] * 100)
                    for i in range(len(EMOTIONS))
                }
            })

        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'results': results
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_INFO,
        "model_loaded": model is not None
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
