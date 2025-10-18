from flask import Flask, render_template, Response, jsonify
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
print("Cargando modelo...")
try:
    model = load_model('../models/mejor_modelo_opt.h5')
    MODEL_INFO = {"name": "CNN Optimizado", "accuracy": "73.78%"}
except:
    model = load_model('../models/mejor_modelo.h5')
    MODEL_INFO = {"name": "CNN Básico", "accuracy": "65.78%"}

# Detector de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)


def detect_emotion(frame):
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
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = detect_emotion(frame)

        cv2.putText(frame, f"{MODEL_INFO['name']} ({MODEL_INFO['accuracy']})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("\n🎭 Servidor iniciado en http://localhost:5000")
    print(f"📊 Modelo: {MODEL_INFO['name']} - {MODEL_INFO['accuracy']}")
    print("⚠️  Presiona Ctrl+C para detener\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
