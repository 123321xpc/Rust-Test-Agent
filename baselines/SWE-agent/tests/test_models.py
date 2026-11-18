from __future__ import annotations

from unittest.mock import ANY, MagicMock, Mock, patch

import pytest

from sweagent.agent.models import ModelArguments, OpenAIModel, get_model, resolve_api_env


@pytest.fixture()
def openai_mock_client():
    model = Mock()
    response = Mock()
    choice = Mock()
    choice.message.content = "test"
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 10
    model.chat.completions.create = MagicMock(return_value=response)

    return model


TEST_HISTORY = [{"role": "system", "content": "Hello, how are you?"}]


def test_openai_model(openai_mock_client):
    for model_name in list(OpenAIModel.MODELS) + list(OpenAIModel.SHORTCUTS):
        TEST_MODEL_ARGUMENTS = ModelArguments(model_name)
        with patch("sweagent.agent.models.keys_config"), patch("sweagent.agent.models.OpenAI"):
            model = OpenAIModel(TEST_MODEL_ARGUMENTS, [])
        model.client = openai_mock_client
        model.query(TEST_HISTORY)


def test_configured_openai_compatible_model(monkeypatch):
    monkeypatch.setenv("MODEL", "custom-model")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("BASE_URL", "https://example.com/v1")
    with patch("sweagent.agent.models.OpenAI") as client:
        model = get_model(ModelArguments("configured"))
    assert model.api_model == "custom-model"
    client.assert_called_once_with(api_key="test-key", base_url="https://example.com/v1",
                                   http_client=ANY, max_retries=0)


def test_configured_model_defaults_to_yj_and_supports_api_limit(monkeypatch):
    for key in ("MODEL", "API_KEY", "BASE_URL", "API_LIMIT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("YJ_BASE_URL", "https://yj.example/v1")
    monkeypatch.setenv("YJ_API_KEY", "yj-key")
    monkeypatch.setenv("YJ_MODEL", "yj-model")
    monkeypatch.setenv("DS_BASE_URL", "https://ds.example/v1")
    monkeypatch.setenv("DS_API_KEY", "ds-key")
    monkeypatch.setenv("DS_MODEL", "ds-model")
    resolve_api_env(api_limit=42)
    with patch("sweagent.agent.models.OpenAI") as client:
        model = get_model(ModelArguments("configured"))
    assert model.api_model == "yj-model"
    assert model.api_budget.limit == 42
    client.assert_called_once_with(api_key="yj-key", base_url="https://yj.example/v1",
                                   http_client=ANY, max_retries=0)

    resolve_api_env(api="ds", api_limit=15)
    with patch("sweagent.agent.models.OpenAI") as client2:
        ds_model = get_model(ModelArguments("configured"))
    assert ds_model.api_model == "ds-model"
    assert ds_model.api_budget.limit == 15
    client2.assert_called_once_with(api_key="ds-key", base_url="https://ds.example/v1",
                                    http_client=ANY, max_retries=0)
