@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem --- 1. Рабочее окружение уже есть (Python 3.11-3.14 и CatBoost) - используем его и ничего не пересоздаём
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys, catboost; sys.exit(0 if (3,11)<=sys.version_info[:2]<(3,15) else 1)" >nul 2>&1 && goto run
)

rem --- 2. Ищем Python 3.13 (под него есть CatBoost), потом любой 3.11+
set "PY="
py -V:3.13 -c "import sys" >nul 2>&1 && set "PY=py -V:3.13"
if not defined PY py -3.13 -c "import sys" >nul 2>&1 && set "PY=py -3.13"
if not defined PY if exist "%LOCALAPPDATA%\Python\pythoncore-3.13-64\python.exe" set "PY="%LOCALAPPDATA%\Python\pythoncore-3.13-64\python.exe""
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe""
if not defined PY if exist "%ProgramFiles%\Python313\python.exe" set "PY="%ProgramFiles%\Python313\python.exe""
if not defined PY py -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)" >nul 2>&1 && set "PY=py"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo.
  echo Python не установлен ^(команда python отвечает заглушкой Microsoft Store^).
  echo.
  echo 1. Скачайте Python 3.13 для Windows: https://www.python.org/downloads/windows/
  echo 2. В первом окне установщика отметьте "Add python.exe to PATH" и нажмите Install Now.
  echo 3. Закройте это окно и снова запустите start.bat.
  echo.
  pause
  exit /b 1
)
echo Использую: %PY%
%PY% --version
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2]<(3,15) else 1)" >nul 2>&1 || (
  echo ВНИМАНИЕ: для этой версии Python нет CatBoost - вместо него будет логистическая регрессия.
  echo Чтобы был CatBoost, установите Python 3.13: py install 3.13
)

rem --- окружение без CatBoost или от другой версии - пересоздаём
if exist .venv (
  echo Пересоздаю окружение .venv ...
  rmdir /s /q .venv
)
echo Создаю окружение .venv ...
%PY% -m venv .venv || (echo Не удалось создать окружение & pause & exit /b 1)

:run
echo Устанавливаю библиотеки ^(первый раз 1-3 минуты^) ...
.venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt || (echo Ошибка установки библиотек & pause & exit /b 1)
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe run_all.py %*
pause
