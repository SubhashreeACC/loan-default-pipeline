# tests/unit/test_utils.py
"""Unit tests for config loading and DB utility helpers."""
import os
from pathlib import Path

import pandas as pd
import pytest

from src.utils.config import _substitute_env_vars, _resolve_values, get_config


class TestEnvVarSubstitution:
    def test_substitutes_existing_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        result = _substitute_env_vars("${MY_TEST_VAR}")
        assert result == "hello"

    def test_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("UNSET_VAR", raising=False)
        result = _substitute_env_vars("${UNSET_VAR:-fallback}")
        assert result == "fallback"

    def test_no_substitution_for_plain_string(self):
        result = _substitute_env_vars("plain_value")
        assert result == "plain_value"

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "production-db")
        result = _substitute_env_vars("${DB_HOST:-localhost}")
        assert result == "production-db"


class TestResolveValues:
    def test_resolves_nested_dict(self, monkeypatch):
        monkeypatch.setenv("PORT", "5432")
        obj = {"database": {"port": "${PORT:-1234}"}}
        resolved = _resolve_values(obj)
        assert resolved["database"]["port"] == "5432"

    def test_resolves_lists(self, monkeypatch):
        monkeypatch.setenv("VAL", "x")
        obj = ["${VAL:-default}", "plain"]
        resolved = _resolve_values(obj)
        assert resolved == ["x", "plain"]

    def test_non_string_values_passthrough(self):
        obj = {"a": 1, "b": True, "c": None}
        resolved = _resolve_values(obj)
        assert resolved == obj


class TestGetConfig:
    def test_config_loads_without_error(self):
        cfg = get_config()
        assert "database" in cfg
        assert "model" in cfg
        assert "mlflow" in cfg

    def test_config_is_cached(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_model_validation_thresholds_present(self):
        cfg = get_config()
        val_cfg = cfg["model"]["validation"]
        assert 0 < val_cfg["min_auc_roc"] <= 1
