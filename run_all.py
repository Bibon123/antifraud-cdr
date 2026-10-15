"""Запуск одной кнопкой (▶ в VS Code): прогон по CDR, проверка на синтетике, затем веб-консоль.

  python run_all.py                       # ищет cdr_out.zip / cdr_out рядом с проектом, уровнем выше или в «Загрузках»
  python run_all.py путь\\к\\cdr_out.zip
  python run_all.py --no-serve            # только посчитать файлы ответа
"""
import os
import sys

from antifraud.venv_guard import ensure_venv

ensure_venv([os.path.abspath(__file__)] + sys.argv[1:])

from antifraud.__main__ import find_source, main  # noqa: E402

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = find_source(args[0] if args else None)
    main(["run", src])
    main(["evaluate"])
    if "--no-serve" not in sys.argv:
        main(["serve", "--open"])
