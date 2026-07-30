"""Small util surface used by skills."""

import pytest

from weather_skills_core import util
from weather_skills_core.errors import UsageError


def test_is_transient():
    assert util.is_transient(Exception("503 Service Unavailable")) is True
    assert util.is_transient(Exception("404 Not Found")) is False


def test_require_env(monkeypatch):
    monkeypatch.setenv("WSC_TEST_USER", "u")
    monkeypatch.setenv("WSC_TEST_PASS", "p")
    assert util.require_env("WSC_TEST_USER", "WSC_TEST_PASS") == ("u", "p")

    monkeypatch.delenv("WSC_TEST_PASS", raising=False)
    with pytest.raises(UsageError, match="WSC_TEST_PASS") as excinfo:
        util.require_env("WSC_TEST_USER", "WSC_TEST_PASS")
    assert excinfo.value.exit_code == 2
