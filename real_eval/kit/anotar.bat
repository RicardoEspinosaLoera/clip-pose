@echo off
rem Abre la herramienta de anotacion. La primera vez pide tu nombre y lo recuerda en mi_nombre.txt.
cd /d "%~dp0"

rem --- 1. encontrar un Python real (no el acceso directo de la Microsoft Store) ---
set PY=
for %%P in (py python) do (
    if not defined PY (
        %%P -c "import sys; assert sys.version_info >= (3, 9); print(sys.executable)" >"%TEMP%\anotar_py.txt" 2>nul && set PY=%%P
    )
)
if not defined PY (
    echo.
    echo No se encontro Python 3.9 o mas reciente.
    echo  - Instalalo desde https://www.python.org/downloads/ y marca "Add Python to PATH".
    echo  - Si se abre la Microsoft Store o aparece "downloading", cierrala: es un acceso directo de Windows,
    echo    no Python. Instala desde python.org y vuelve a abrir este archivo.
    echo.
    pause
    exit /b 1
)
set /p PYEXE=<"%TEMP%\anotar_py.txt"
echo Python: %PYEXE%

rem --- 2. instalar OpenCV y NumPy si faltan (unos 40 MB, solo la primera vez) ---
%PY% -c "import cv2, numpy" 2>nul
if errorlevel 1 (
    %PY% -m pip --version >nul 2>nul
    if errorlevel 1 (
        echo.
        echo Este Python ^(%PYEXE%^) no tiene pip, asi que no puede instalar OpenCV.
        echo Instala Python desde https://www.python.org/downloads/ marcando "Add Python to PATH" y vuelve a abrir este archivo.
        pause
        exit /b 1
    )
    echo.
    echo Instalando OpenCV y NumPy, unos 40 MB. Solo pasa la primera vez; puede tardar unos minutos.
    %PY% -m pip install --user --disable-pip-version-check opencv-python numpy
    %PY% -c "import cv2, numpy" 2>nul
    if errorlevel 1 (
        echo.
        echo No se pudo instalar OpenCV. Revisa la conexion a internet ^(o el proxy de la red^) y vuelve a intentarlo.
        echo Tambien puedes instalarlo a mano en una terminal:  %PY% -m pip install --user opencv-python numpy
        pause
        exit /b 1
    )
)

rem --- 3. nombre del anotador ---
if exist mi_nombre.txt goto have_name
set /p NAME=Tu nombre (sin espacios ni acentos, por ejemplo ana):
if "%NAME%"=="" goto end
>mi_nombre.txt echo %NAME%
:have_name
set /p NAME=<mi_nombre.txt
echo Anotador: %NAME%
echo Se abre la ventana de anotacion. Para salir guardando: tecla q.
%PY% pose_train\annotate_tool.py --annotator %NAME%
:end
pause
