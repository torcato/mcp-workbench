from pathlib import Path

import pytest

from app.prompts.manager import PromptManager
from app.prompts.models import PromptOverride


def test_load_default_profiles(tmp_path: Path) -> None:
    source = tmp_path / "profiles.yaml"
    source.write_text(
        """default_profile: default
profiles:
  default:
    name: Default
    system_prompt: |
      Default system text.
""",
        encoding="utf-8",
    )

    manager = PromptManager(profiles_path=source)
    profile = manager.get_profile()

    assert profile.name == "Default"
    assert profile.system_prompt.strip() == "Default system text."
    assert manager.list_profiles() == ("default",)


def test_compose_prompt_with_override(tmp_path: Path) -> None:
    source = tmp_path / "profiles.yaml"
    source.write_text(
        """profiles:
  default:
    name: Default
    system_prompt: |
      Base system text.
    user_prompt: |
      Base user text.
""",
        encoding="utf-8",
    )

    manager = PromptManager(profiles_path=source)
    override = PromptOverride(system_prompt="Override system.", user_prompt="Override user.")
    messages = manager.compose_prompt(override=override)

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "Override system."
    assert messages[1].role == "user"
    assert messages[1].content == "Override user."


def test_missing_profile_raises(tmp_path: Path) -> None:
    source = tmp_path / "profiles.yaml"
    source.write_text(
        """profiles:
  default:
    name: Default
    system_prompt: |
      Default.
""",
        encoding="utf-8",
    )

    manager = PromptManager(profiles_path=source)

    with pytest.raises(KeyError, match="Prompt profile not found"):
        manager.get_profile("missing")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    source = tmp_path / "profiles.yaml"
    source.write_text("not a yaml", encoding="utf-8")

    manager = PromptManager(profiles_path=source)
    with pytest.raises(ValueError):
        manager.load_profiles()
