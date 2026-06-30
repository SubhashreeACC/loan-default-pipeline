# tests/unit/test_validation.py
"""Unit tests for the schema/quick-check validation helpers."""

import pandas as pd

from src.data.validation import validate_schema


class TestValidateSchema:
    def test_no_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        missing = validate_schema(df, required_cols=["a", "b"])
        assert missing == []

    def test_detects_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        missing = validate_schema(df, required_cols=["a", "b", "c", "d"])
        assert set(missing) == {"c", "d"}

    def test_empty_required_list(self):
        df = pd.DataFrame({"a": [1]})
        assert validate_schema(df, required_cols=[]) == []

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        missing = validate_schema(df, required_cols=["a", "b"])
        assert set(missing) == {"a", "b"}
