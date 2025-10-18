from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)

# Configuración
EMOTIONS = ['Enojado', 'Disgusto', 'Miedo',
            'Feliz', 'Neutral', 'Triste', 'Sorpresa']
EMOJIS = ['😠', '🤢', '😨', '😊', '😐', '😢', '😮']
IMG_SIZE = 64

# Cargar modelo
MODEL_PATH = os.path.join('..', 'models', 'mejor_modelo_opt.h5')
print(f"Cargando modelo desde: {MODEL_PATH}")

try:
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado, descargando...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id='adrianqe/emotion-recognition-model',
            filename='mejor_modelo_opt.h5',
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

# Variable global para cámara
camera = None


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def detect_emotion(frame):
    if model is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        predictions = model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx] * 100
        emotion = EMOTIONS[emotion_idx]
        emoji = EMOJIS[emotion_idx]

        color = (0, 255, 0) if confidence > 70 else (
            255, 165, 0) if confidence > 50 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        text = f"{emoji} {emotion}"
        cv2.putText(frame, text, (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{confidence:.1f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def generate_frames():
    cam = get_camera()

    while True:
        success, frame = cam.read()
        if not success:
            break

        frame = detect_emotion(frame)

        cv2.putText(frame, f"{MODEL_INFO['name']} ({MODEL_INFO['accuracy']})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return {"status": "ok", "model": MODEL_INFO}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
