#!/bin/bash

# Descargar modelo si no existe
if [ ! -f models/mejor_modelo_opt.h5 ]; then
    echo "Descargando modelo..."
    mkdir -p models
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='adrianqe/cnn-emotion-recognition-73', filename='emotion-recognition-model.h5', local_dir='models')"
fi

# Iniciar aplicación
cd web
gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120