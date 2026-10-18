import pytest

from app.config import Settings


def test_mesh_base_url_literal():
    assert Settings().mesh_base_url == "https://api.meshapi.ai/v1"


def test_verifier_must_differ_from_drafter_family():
    with pytest.raises(ValueError, match="different model family"):
        Settings(model_drafter="openai/gpt-5-nano", model_verifier="openai/gpt-4.1-nano")
