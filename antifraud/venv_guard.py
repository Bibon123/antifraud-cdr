"""Запуск всегда из окружения проекта (.venv), даже если в терминале выбран другой Python.

На компьютере может стоять несколько версий Python (например, 3.13 и 3.15). start.bat создаёт .venv
на 3.13 с CatBoost, но `python ...` в терминале VS Code берёт системный Python. Если .venv есть, а
запущены не из неё, — перезапускаем ту же команду интерпретатором .venv.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def venv_python() -> str | None:
    p = (os.path.join(ROOT, ".venv", "Scripts", "python.exe") if os.name == "nt"
         else os.path.join(ROOT, ".venv", "bin", "python"))
    return p if os.path.exists(p) else None


def in_venv() -> bool:
    return os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.join(ROOT, ".venv"))


def ensure_venv(args: list[str]) -> None:
    """Если есть .venv и мы не в ней — выполнить `args` её Python и выйти с тем же кодом."""
    py = venv_python()
    if not py or in_venv() or os.environ.get("ANTIFRAUD_NO_REEXEC"):
        return
    print(f"Python {sys.version.split()[0]} не из окружения проекта — перезапускаю через .venv", flush=True)
    env = dict(os.environ, ANTIFRAUD_NO_REEXEC="1")
    try:
        code = subprocess.call([py] + args, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)
