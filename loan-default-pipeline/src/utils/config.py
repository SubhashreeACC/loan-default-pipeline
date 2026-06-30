# src/utils/config.py
"""Singleton config loader with environment variable substitution."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config/config.yaml"))

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR:-default} patterns with environment values."""
    def _replace(match):
        var_name, default = match.group(1), match.group(2) or ""
        return os.environ.get(var_name, default)
    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_values(obj: Any) -> Any:
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_values(i) for i in obj]
    return obj


@lru_cache(maxsize=1)
def get_config() -> dict:
    path = CONFIG_PATH
    if not path.exists():
        # Try relative to repo root
        path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _resolve_values(raw)
