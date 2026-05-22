from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from app.llm.base import ChatMessage
from app.prompts.models import PromptOverride, PromptProfile, PromptProfiles


class PromptManager:
    def __init__(self, profiles_path: str | Path, default_profile: str = "default") -> None:
        self.profiles_path = Path(profiles_path)
        self.default_profile = default_profile
        self._profiles: PromptProfiles | None = None

    def load_profiles(self) -> PromptProfiles:
        if not self.profiles_path.exists():
            raise FileNotFoundError(f"Prompt profiles file not found: {self.profiles_path}")

        try:
            raw = yaml.safe_load(self.profiles_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid prompt profiles YAML: {self.profiles_path}") from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Prompt profiles file must contain a YAML mapping")

        raw = dict(raw)
        if "default_profile" not in raw:
            raw["default_profile"] = self.default_profile

        self._profiles = PromptProfiles(**raw)
        return self._profiles

    @property
    def profiles(self) -> PromptProfiles:
        if self._profiles is None:
            return self.load_profiles()
        return self._profiles

    def get_profile(self, profile_name: str | None = None) -> PromptProfile:
        profiles = self.profiles
        key = profile_name or profiles.default_profile
        if key not in profiles.profiles:
            raise KeyError(f"Prompt profile not found: {key}")
        return profiles.profiles[key]

    def compose_prompt(self, profile_name: str | None = None, override: PromptOverride | None = None) -> list[ChatMessage]:
        profile = self.get_profile(profile_name)
        system_prompt = override.system_prompt if override and override.system_prompt is not None else profile.system_prompt
        user_prompt = override.user_prompt if override and override.user_prompt is not None else profile.user_prompt

        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        if user_prompt:
            messages.append(ChatMessage(role="user", content=user_prompt))

        return messages

    def list_profiles(self) -> Iterable[str]:
        return tuple(self.profiles.profiles.keys())
