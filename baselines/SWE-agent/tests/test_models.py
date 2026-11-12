from __future__ import annotations

from unittest.mock import ANY, MagicMock, Mock, patch

import pytest

from sweagent.agent.models import ModelArguments, OpenAIModel, get_model


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
