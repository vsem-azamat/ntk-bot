from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps import llm
from bot import db
from config import cnfg


def test_probability_defaults_to_env_then_uses_stored(monkeypatch):
    monkeypatch.setattr(cnfg, "LLM_ANSWER_PROBABILITY", 0.05)
    assert llm.answer_probability() == 0.05
    llm.set_answer_probability("-1")
    assert llm.answer_probability() == -1.0
    llm._settings.reset()  # a restart reads the stored value
    assert llm.answer_probability() == -1.0
    db.set_setting(llm.PROBABILITY_KEY, "garbage")
    llm._settings.reset()
    assert llm.answer_probability() == 0.05


def test_settings_survive_database_outage(monkeypatch):
    llm.set_system_prompt("persona")
    llm.set_answer_probability("0.05")
    assert llm.ensure_settings()

    def down(*_):
        raise RuntimeError("database down")

    monkeypatch.setattr(db, "get_setting", down)
    assert (llm.system_prompt(), llm.answer_probability()) == ("persona", 0.05)


def test_unloadable_settings_give_no_prompt(monkeypatch):
    def down(*_):
        raise RuntimeError("database down")

    monkeypatch.setattr(db, "get_setting", down)
    assert llm.system_prompt() == ""
    assert not llm.settings_loaded()


def test_set_prompt_keeps_previous_and_ignores_repeats():
    assert llm.set_system_prompt("v1")
    assert llm.set_system_prompt("v2")
    assert not llm.set_system_prompt("v2")  # repeated /setprompt keeps the rollback copy
    assert llm.system_prompt() == "v2"
    assert db.get_setting(llm.PREVIOUS_PROMPT_KEY) == "v1"


def test_prompt_is_imported_lazily_on_first_load(tmp_path, monkeypatch):
    f = tmp_path / ".instructions"
    f.write_text("\ufefflegacy persona", encoding="utf-8")
    monkeypatch.setattr(cnfg, "LEGACY_INSTRUCTIONS_PATH", str(f))
    assert llm.system_prompt() == "legacy persona"


def test_legacy_prompt_imported_once(tmp_path):
    f = tmp_path / ".instructions"
    f.write_text("  old prompt \n", encoding="utf-8")
    assert llm.import_legacy_prompt(str(f)) is True
    assert llm.system_prompt() == "old prompt"
    f.write_text("changed", encoding="utf-8")
    assert llm.import_legacy_prompt(str(f)) is False
    assert llm.system_prompt() == "old prompt"


def _session(payload):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = AsyncMock(return_value=payload)
    post_cm = MagicMock()
    post_cm.__aenter__ = AsyncMock(return_value=response)
    post_cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post = MagicMock(return_value=post_cm)
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return session_cm, session


@pytest.mark.asyncio
async def test_reply_uses_prompt_and_reasoning_budget(monkeypatch):
    monkeypatch.setattr(cnfg, "OPENROUTER_API_KEY", "k")
    cm, session = _session({"choices": [{"message": {"content": "  Pohoda)))0 "}}]})
    with patch("aiohttp.ClientSession", return_value=cm):
        assert await llm.get_llm_response("hi", "persona") == "Pohoda)))0"
    body = session.post.call_args.kwargs["json"]
    assert body["messages"][0] == {"role": "system", "content": "persona"}
    assert body["max_tokens"] >= 1024 and body["reasoning"]["exclude"] is True


@pytest.mark.asyncio
async def test_empty_model_content_gives_no_reply(monkeypatch):
    monkeypatch.setattr(cnfg, "OPENROUTER_API_KEY", "k")
    cm, _ = _session({"choices": [{"message": {"content": None}}]})
    with patch("aiohttp.ClientSession", return_value=cm):
        assert await llm.get_llm_response("hi", "persona") is None
