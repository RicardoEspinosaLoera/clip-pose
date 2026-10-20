@echo off
rem Empaqueta tus anotaciones en anotaciones_<nombre>.zip para enviarlas.
cd /d "%~dp0"
if not exist mi_nombre.txt (echo No encuentro mi_nombre.txt: primero anota con anotar.bat & pause & exit /b 1)
set /p NAME=<mi_nombre.txt
powershell -NoProfile -Command "Compress-Archive -Path 'real_eval\annotations\%NAME%' -DestinationPath 'anotaciones_%NAME%.zip' -Force"
for /f %%c in ('dir /b "real_eval\annotations\%NAME%\*.json" ^| find /c /v ""') do echo %%c frames anotados o excluidos de 200
echo Creado anotaciones_%NAME%.zip : enviaselo a Ricardo.
pause
