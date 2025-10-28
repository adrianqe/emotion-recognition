## Sistema de Reconocimiento de Emociones

Sistema de reconocimiento de emociones faciales en tiempo real utilizando Deep Learning.
## Live Demo
https://emotion-recognition-0w5a.onrender.com

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

**Nota:** Si tienes problemas con TensorFlow, usa:
```bash
pip install tensorflow==2.20.0 numpy==1.26.0
```

3. **Descargar el modelo entrenado:**
```bash
# Opción 1: Descarga manual
# Descarga desde: https://huggingface.co/adrianqe/cnn-emotion-recognition-73

# Opción 2: Usando wget/curl
wget https://huggingface.co/adrianqe/cnn-emotion-recognition-73/resolve/main/emotion-recognition-model.h5 -P models/

# Opción 3: Usando Python
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='adrianqe/cnn-emotion-recognition-73', filename='emotion-recognition-model.h5', local_dir='models')"
```

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
| CNN Básico         | 65.78%     | ~35 min              |
| Transfer Learning  | 61.48%     | ~40 min              |
| **CNN Optimizado** | **73.78%** | **~35 min**          |

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
