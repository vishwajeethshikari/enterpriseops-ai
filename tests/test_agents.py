
import pytest
import agents


def test_ask_llm_json_returns_valid_dictionary(monkeypatch):
    def mock_chat(**kwargs):
        return {
            "message": {
                "content": '{"status": "success"}'
            }
        }

    monkeypatch.setattr(agents.ollama, "chat", mock_chat)

    result = agents.ask_llm_json("Test prompt")

    assert result == {"status": "success"}
    assert isinstance(result, dict)


def test_ask_llm_json_rejects_invalid_json(monkeypatch):
    def mock_chat(**kwargs):
        return {
            "message": {
                "content": "This is not valid JSON"
            }
        }

    monkeypatch.setattr(agents.ollama, "chat", mock_chat)

    with pytest.raises(RuntimeError):
        agents.ask_llm_json("Test prompt")
