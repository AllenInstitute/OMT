"""Paths and data locations for the experiments in this folder.

Values are read from ``config.toml`` (shared, committed) and then overridden by
``config.local.toml`` (your machine only, gitignored) if that file exists.
Relative entries under ``[paths]`` are resolved against this folder, not the
current working directory.

Usage::

    from config import CONFIG, PATHS, add_code_dirs
    add_code_dirs()                    # put PATHS["code_dirs"] on sys.path
    emb = PATHS["embeddings"]          # absolute path as a str
"""
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

HERE = Path(__file__).resolve().parent


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve(value):
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    if isinstance(value, str):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else (HERE / p).resolve())
    return value


def load_config():
    with open(HERE / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    local = HERE / "config.local.toml"
    if local.exists():
        with open(local, "rb") as f:
            _merge(cfg, tomllib.load(f))
    cfg["paths"] = {k: _resolve(v) for k, v in cfg.get("paths", {}).items()}
    return cfg


CONFIG = load_config()
PATHS = CONFIG["paths"]


def add_code_dirs():
    """Prepend every directory in PATHS["code_dirs"] to sys.path (in order)."""
    for d in reversed(PATHS.get("code_dirs", [])):
        if d not in sys.path:
            sys.path.insert(0, d)
