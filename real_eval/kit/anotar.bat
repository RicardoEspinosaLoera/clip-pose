@echo off
rem Abre la herramienta de anotacion. La primera vez pide tu nombre y lo recuerda en mi_nombre.txt.
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
if exist mi_nombre.txt goto have_name
set /p NAME=Tu nombre (sin espacios ni acentos, por ejemplo ana):
if "%NAME%"=="" goto end
>mi_nombre.txt echo %NAME%
:have_name
set /p NAME=<mi_nombre.txt
echo Anotador: %NAME%
%PY% -c "import cv2, numpy" 2>nul || %PY% -m pip install --quiet --user opencv-python numpy
%PY% pose_train\annotate_tool.py --annotator %NAME%
:end
pause
