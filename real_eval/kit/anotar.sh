#!/usr/bin/env bash
# Abre la herramienta de anotacion (Mac / Linux). La primera vez pide tu nombre y lo recuerda.
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python)
if [ ! -s mi_nombre.txt ]; then
    read -r -p "Tu nombre (sin espacios ni acentos, por ejemplo ana): " NAME
    [ -n "$NAME" ] || exit 1
    echo "$NAME" > mi_nombre.txt
fi
NAME=$(tr -d '[:space:]' < mi_nombre.txt)
echo "Anotador: $NAME"
"$PY" -c "import cv2, numpy" 2>/dev/null || "$PY" -m pip install --quiet --user opencv-python numpy
"$PY" pose_train/annotate_tool.py --annotator "$NAME"
