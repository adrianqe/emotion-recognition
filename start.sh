#!/bin/bash

# Descargar modelo si no existe
if [ ! -f models/emotion-recognition-model.h5 ]; then
    echo "Descargando modelo..."
    mkdir -p models
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='adrianqe/cnn-emotion-recognition-73', filename='emotion-recognition-model.h5', local_dir='models', local_dir_use_symlinks=False)"
fi

# Iniciar aplicación con configuración optimizada para Render Free
cd web
gunicorn --bind 0.0.0.0:$PORT app:app \
    --timeout 120 \
    --workers 1 \
    --threads 2 \
    --worker-class sync \
    --max-requests 100 \
    --max-requests-jitter 10 \
    --preload