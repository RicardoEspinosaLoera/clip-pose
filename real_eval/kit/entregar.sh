#!/usr/bin/env bash
# Empaqueta tus anotaciones en anotaciones_<nombre>.zip para enviarlas (Mac / Linux).
cd "$(dirname "$0")" || exit 1
[ -s mi_nombre.txt ] || { echo "No encuentro mi_nombre.txt: primero anota con anotar.sh"; exit 1; }
NAME=$(tr -d '[:space:]' < mi_nombre.txt)
PY=$(command -v python3 || command -v python)
"$PY" -c "import shutil; shutil.make_archive('anotaciones_$NAME', 'zip', 'real_eval/annotations', '$NAME')"
echo "$(ls real_eval/annotations/$NAME/*.json | wc -l) frames anotados o excluidos de 200"
echo "Creado anotaciones_$NAME.zip : enviaselo a Ricardo."
