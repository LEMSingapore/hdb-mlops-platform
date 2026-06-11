"""Tests for ChatConfig loading the Anthropic API key.

The chat app resolves ANTHROPIC_API_KEY through ChatConfig so the key works
whether it is exported in the shell or written to a .env file at the project
root. The env_file is resolved relative to the working directory, so these
tests change into a tmp_path holding a fixture .env.
"""

from ui.chat_app.config import ChatConfig


def test_key_loaded_from_dotenv_when_env_var_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    config = ChatConfig()

    assert config.anthropic_api_key == "sk-ant-from-dotenv"


def test_env_var_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")

    config = ChatConfig()

    assert config.anthropic_api_key == "sk-ant-from-env"


def test_key_empty_when_neither_env_var_nor_dotenv_present(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    config = ChatConfig()

    assert config.anthropic_api_key == ""
