"""Командная строка.

  python -m antifraud run      [источник] [--out out] [--no-ml] [--no-realtime]
  python -m antifraud evaluate [--out out]
  python -m antifraud serve    [--out out] [--port 8000]
  python -m antifraud synth    папка [--scale 1.0]

Источник — папка с *.cdr или cdr_out.zip. Без аргумента ищется cdr_out(.zip)
рядом с проектом или уровнем выше.
"""
from __future__ import annotations

import argparse
import os
import sys

from .venv_guard import ensure_venv

if __name__ == "__main__":
    ensure_venv(["-m", "antifraud"] + sys.argv[1:])

from .config import DEFAULT_OUT, ROOT  # noqa: E402


def find_source(arg: str | None) -> str:
    if arg:
        if not os.path.exists(arg):
            sys.exit(f"Не найден источник: {arg}")
        return arg
    for base in (ROOT, os.path.dirname(ROOT), os.getcwd(), os.path.join(os.path.expanduser("~"), "Downloads")):
        for cand in ("cdr_out", "cdr_out.zip"):
            p = os.path.join(base, cand)
            if os.path.exists(p):
                return p
    sys.exit("Не найден cdr_out.zip / папка cdr_out. Передайте путь: python -m antifraud run путь\\к\\cdr_out.zip")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="antifraud", description="Антифрод по метаданным вызовов (CDR)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="полный прогон по CDR")
    r.add_argument("source", nargs="?"); r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--no-ml", action="store_true"); r.add_argument("--no-realtime", action="store_true")
    e = sub.add_parser("evaluate", help="проверка на синтетике со сдвигом параметров")
    e.add_argument("--out", default=DEFAULT_OUT)
    s = sub.add_parser("serve", help="веб-интерфейс и API")
    s.add_argument("--out", default=DEFAULT_OUT); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--open", action="store_true", help="открыть браузер")
    g = sub.add_parser("synth", help="сгенерировать синтетические CDR с разметкой")
    g.add_argument("folder"); g.add_argument("--scale", type=float, default=1.0); g.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .pipeline import run
        src = find_source(a.source)
        print(f"Источник: {src}")
        try:
            run(src, a.out, with_ml=False if a.no_ml else None, with_realtime=not a.no_realtime)
        except (FileNotFoundError, ValueError, OSError) as e:
            sys.exit(f"Ошибка: {e}")
    elif a.cmd == "evaluate":
        from .evaluate import run_all
        print("Сценарии на синтетике:")
        run_all(a.out)
        print(f"Отчёт: {os.path.join(a.out, 'eval.json')}")
    elif a.cmd == "serve":
        os.environ["ANTIFRAUD_OUT"] = os.path.abspath(a.out)
        if not os.path.exists(os.path.join(a.out, "summary.json")):
            sys.exit(f"В {a.out} нет результатов. Сначала: python -m antifraud run")
        import socket
        import uvicorn
        port = a.port
        for p in range(a.port, a.port + 20):                 # порт занят (сервер уже запущен?) — берём следующий
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
                if sk.connect_ex(("127.0.0.1", p)) != 0:
                    port = p
                    break
        else:
            sys.exit(f"Порты {a.port}–{a.port + 19} заняты. Укажите другой: --port 9000")
        if port != a.port:
            print(f"Порт {a.port} занят, использую {port}")
        url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '127.0.0.1') else a.host}:{port}"
        print(f"Откройте в браузере: {url}   (остановить: Ctrl+C)")
        if a.open:
            import threading
            import webbrowser
            threading.Timer(2.5, lambda: webbrowser.open(url)).start()
        uvicorn.run("app.main:app", host=a.host, port=port, app_dir=ROOT)
    elif a.cmd == "synth":
        import json
        from .synth import Synth
        sy = Synth(seed=a.seed, scale=a.scale).build()
        os.makedirs(a.folder, exist_ok=True)
        for name, buf in sy.sources():
            with open(os.path.join(a.folder, name), "w", encoding="utf-8") as fo:
                fo.write(buf.getvalue())
        with open(os.path.join(a.folder, "truth.json"), "w", encoding="utf-8") as fo:
            json.dump({"fraud": sy.truth, "legit": sy.legit}, fo, ensure_ascii=False, indent=1)
        print(f"Сгенерировано {len(sy.rows):,} записей в {a.folder} (+ truth.json)")


if __name__ == "__main__":
    main()
