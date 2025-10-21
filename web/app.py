from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import gc

app = Flask(__name__)

# Configuración
EMOTIONS = ['Enojado', 'Disgusto', 'Miedo',
            'Feliz', 'Neutral', 'Triste', 'Sorpresa']
EMOJIS = ['😠', '🤢', '😨', '😊', '😐', '😢', '😮']
IMG_SIZE = 64

# Variable global para modelo (carga lazy)
model = None
MODEL_INFO = {"name": "Cargando...", "accuracy": "N/A"}
face_cascade = None


def load_model_lazy():
    """Carga el modelo solo cuando se necesita"""
    global model, MODEL_INFO, face_cascade

    if model is not None:
        return model

    MODEL_PATH = os.path.join('..', 'models', 'emotion-recognition-model.h5')
    print(f"Cargando modelo desde: {MODEL_PATH}")

    try:
        if not os.path.exists(MODEL_PATH):
            print("Modelo no encontrado, descargando desde Hugging Face...")
            os.makedirs('../models', exist_ok=True)

            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id='adrianqe/cnn-emotion-recognition-73',
                filename='emotion-recognition-model.h5',
                local_dir='../models',
                local_dir_use_symlinks=False
            )
            print("✅ Modelo descargado exitosamente")

        # Cargar con configuración de memoria optimizada
        import tensorflow as tf

        # Limitar memoria de TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        from keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Cargar detector de rostros
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        MODEL_INFO = {"name": "CNN Optimizado", "accuracy": "73.78%"}
        print("✅ Modelo cargado exitosamente")

        # Limpiar memoria
        gc.collect()

        return model

    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        import traceback
        traceback.print_exc()
        MODEL_INFO = {"name": "Error", "accuracy": "N/A"}
        return None


@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predicción de imagen desde el navegador"""
    try:
        # Cargar modelo si no está cargado
        current_model = load_model_lazy()

        if current_model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo no disponible. Intenta recargar la página.'
            }), 500

        # Recibir imagen
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

        import base64
        image_data = data['image'].split(',')[1]

        # Decodificar
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Imagen inválida'}), 400

        # Reducir tamaño para mejor rendimiento
        max_size = 480
        height, width = img.shape[:2]
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # IMPORTANTE: Detectar rostros primero
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Si no se detectan rostros, usar imagen completa como fallback
        if len(faces) == 0:
            print("⚠️ No se detectó rostro, usando imagen completa")
            face_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            predictions = current_model.predict(face_input, verbose=0)[0]
            emotion_idx = int(np.argmax(predictions))
            confidence = float(predictions[emotion_idx] * 100)
            emotion = EMOTIONS[emotion_idx]
            emoji = EMOJIS[emotion_idx]

            # Limpiar memoria
            del img, gray, face_resized, face_normalized, face_input
            gc.collect()

            return jsonify({
                'success': True,
                'faces_detected': 0,
                'results': [{
                    'emotion': emotion,
                    'emoji': emoji,
                    'confidence': confidence,
                    'probabilities': {
                        EMOTIONS[i]: float(predictions[i] * 100)
                        for i in range(len(EMOTIONS))
                    }
                }]
            })

        # Procesar solo el primer rostro (optimización)
        results = []
        (x, y, w, h) = faces[0]

        # Extraer región del rostro
        face_roi = gray[y:y+h, x:x+w]

        # Preprocesar rostro
        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # Predecir
        predictions = current_model.predict(face_input, verbose=0)[0]
        emotion_idx = int(np.argmax(predictions))
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

        # Limpiar memoria
        del img, gray, face_roi, face_resized, face_normalized, face_input
        gc.collect()

        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'results': results
        })

    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error en el servidor: {str(e)}'
        }), 500


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
