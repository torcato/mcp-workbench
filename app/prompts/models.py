from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PromptProfile(BaseModel):
    name: str
    description: str | None = None
    system_prompt: str
    user_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "system_prompt")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class PromptProfiles(BaseModel):
    profiles: dict[str, PromptProfile]
    default_profile: str = "default"

    @model_validator(mode="after")
    def validate_profiles(self) -> "PromptProfiles":
        if not self.profiles:
            raise ValueError("profiles section must contain at least one profile")

        if not self.default_profile.strip():
            raise ValueError("default_profile must not be empty")

        for key in self.profiles:
            if not key.strip():
                raise ValueError("profile names must not be empty")

        if self.default_profile not in self.profiles:
            raise ValueError("default_profile must reference an existing profile")

        return self


class PromptOverride(BaseModel):
    system_prompt: str | None = None
    user_prompt: str | None = None
