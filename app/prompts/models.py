from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PromptProfile(BaseModel):
    name: str
    description: str | None = None
    system_prompt: str
    user_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptProfiles(BaseModel):
    profiles: dict[str, PromptProfile]
    default_profile: str = "default"

    @model_validator(mode="after")
    def validate_profiles(self) -> "PromptProfiles":
        if not self.profiles:
            raise ValueError("profiles section must contain at least one profile")

        if self.default_profile not in self.profiles:
            raise ValueError("default_profile must reference an existing profile")

        return self


class PromptOverride(BaseModel):
    system_prompt: str | None = None
    user_prompt: str | None = None
