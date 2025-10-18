# emotion-recognition# 🎭 Sistema de Reconocimiento de Emociones

Sistema de reconocimiento de emociones faciales en tiempo real utilizando Deep Learning.

## 📊 Características

- **Modelo:** CNN Optimizado
- **Precisión:** 73.78%
- **Emociones:** 7 (Feliz, Triste, Enojado, Miedo, Sorpresa, Disgusto, Neutral)
- **Dataset:** FER2013 + RAFDB (49,779 imágenes)
- **Tecnologías:** Python, TensorFlow, Keras, OpenCV, Flask

## 🚀 Instalación

### Requisitos

- Python 3.11.9
- Cámara web
- 8GB RAM mínimo

### Pasos

1. **Clonar el repositorio:**

```bash
git clone https://github.com/adrianqe/emotion-recognition.git
cd emotion-recognition
```

2. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

3. **Descargar el modelo entrenado:**

Descarga el modelo desde: [Google Drive / Dropbox / tu enlace]

Coloca el archivo `mejor_modelo_opt.h5` en la carpeta `models/`

4. **Ejecutar la aplicación:**

```bash
cd web
python app.py
```

5. **Abrir en navegador:**

```
http://localhost:5000
```

## 📁 Estructura del Proyecto

```
emotion-recognition/
├── web/
│   ├── app.py              # Backend Flask
│   └── templates/
│       └── index.html      # Frontend
├── models/
│   └── mejor_modelo_opt.h5 # Modelo entrenado (descargar)
├── requirements.txt
└── README.md
```

## 🎯 Uso

1. Permite el acceso a la cámara
2. Coloca tu rostro frente a la cámara
3. El sistema detectará automáticamente tu emoción
4. Haz diferentes expresiones para probar las 7 emociones

## 🧠 Arquitectura del Modelo

- **Tipo:** CNN (Convolutional Neural Network)
- **Capas:** 4 bloques convolucionales
- **Input:** Imágenes 64x64 en escala de grises
- **Output:** 7 clases (emociones)
- **Optimizador:** Adam
- **Función de pérdida:** Sparse Categorical Crossentropy

## 📈 Resultados

| Modelo             | Precisión  | Tiempo Entrenamiento |
| ------------------ | ---------- | -------------------- |
| CNN Básico         | 65.78%     | ~20 min              |
| Transfer Learning  | 61.48%     | ~30 min              |
| **CNN Optimizado** | **73.78%** | **~25 min**          |

## 🛠️ Tecnologías

- **Backend:** Python, Flask
- **ML/DL:** TensorFlow, Keras
- **Visión:** OpenCV
- **Frontend:** HTML, CSS, JavaScript

## 📝 Créditos

Proyecto desarrollado como parte del curso de Inteligencia Artificial.

**Dataset:** FER2013 + RAFDB

## 📄 Licencia

MIT License
