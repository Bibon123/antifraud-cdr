"""Загрузка конфигурации порогов (config.toml в корне проекта)."""
import os
import tomllib
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "config.toml")
DEFAULT_OUT = os.path.join(ROOT, "out")


@lru_cache(maxsize=8)
def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_update(base: dict, upd: dict) -> dict:
    """Копия base с переопределёнными ключами (для экспериментов с порогами)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
