"""Конфиг: приоритет источников, понятные ошибки, redaction секретов."""

from __future__ import annotations

import pytest

from advent_core.config import DEFAULT_MODEL, Config, ConfigError, redact


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Изолируем тесты от реального .env разработчика."""
    for key in (
        "MISTRAL_API_KEY",
        "MISTRAL_MODEL",
        "MISTRAL_TEMPERATURE",
        "MISTRAL_TOP_P",
        "MISTRAL_MAX_TOKENS",
        "MISTRAL_SEED",
        "MISTRAL_STOP",
        "MISTRAL_REASONING_EFFORT",
        "GITHUB_TOKEN",
        "ADVENT_SYSTEM_PROMPT",
    ):
        monkeypatch.delenv(key, raising=False)
    # resolve() вызывает load_dotenv, который иначе подтянет настоящий .env.
    monkeypatch.setattr("advent_core.config.load_env", lambda: None)


def test_missing_key_gives_actionable_error():
    with pytest.raises(ConfigError) as exc:
        Config.resolve()
    assert "MISTRAL_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


def test_default_model_when_nothing_set(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    assert Config.resolve().model == DEFAULT_MODEL


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_MODEL", "ministral-8b-latest")
    assert Config.resolve().model == "ministral-8b-latest"


def test_cli_argument_beats_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_MODEL", "ministral-8b-latest")
    assert Config.resolve(model="mistral-large-latest").model == "mistral-large-latest"


def test_numeric_env_is_parsed(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "0.3")
    monkeypatch.setenv("MISTRAL_MAX_TOKENS", "256")
    config = Config.resolve()
    assert config.params.temperature == 0.3
    assert config.params.max_tokens == 256


def test_zero_temperature_from_cli_is_not_swallowed(monkeypatch):
    """0.0 — валидное значение, а не 'не задано'."""
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "0.9")
    assert Config.resolve(temperature=0.0).params.temperature == 0.0


def test_broken_numeric_env_is_reported(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_MAX_TOKENS", "много")
    with pytest.raises(ConfigError):
        Config.resolve()


def test_missing_system_prompt_file_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    with pytest.raises(ConfigError):
        Config.resolve(system=tmp_path / "нет-такого.md")


def test_redact_hides_secret(monkeypatch):
    secret = "sk-super-secret-value-123456"
    monkeypatch.setenv("MISTRAL_API_KEY", secret)
    assert secret not in redact(f"401 Unauthorized: key={secret}")
    assert "***" in redact(f"key={secret}")


def test_redact_ignores_short_values(monkeypatch):
    """Короткий 'секрет' мог бы вырезать куски обычного текста."""
    monkeypatch.setenv("MISTRAL_API_KEY", "abc")
    assert redact("abcdefg") == "abcdefg"


def test_generation_params_come_from_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_TOP_P", "0.8")
    monkeypatch.setenv("MISTRAL_SEED", "42")
    monkeypatch.setenv("MISTRAL_STOP", "КОНЕЦ, ###")
    monkeypatch.setenv("MISTRAL_REASONING_EFFORT", "high")

    params = Config.resolve().params
    assert params.top_p == 0.8
    assert params.random_seed == 42
    assert params.stop == ["КОНЕЦ", "###"]
    assert params.reasoning_effort == "high"


def test_out_of_range_param_from_env_is_reported(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k" * 32)
    monkeypatch.setenv("MISTRAL_TEMPERATURE", "5")
    with pytest.raises(ConfigError):
        Config.resolve()
