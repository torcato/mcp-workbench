from __future__ import annotations

import json
from typing import Iterator

import httpx

from app.llm.base import ChatMessage, LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-3.5-turbo",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for OpenAI-compatible providers")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[ChatMessage], model: str | None, temperature: float, stream: bool) -> dict:
        return {
            "model": model or self.default_model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
            "stream": stream,
        }

    def _extract_text(self, response_json: dict) -> str:
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from completion response")

        return "".join(
            choice.get("message", {}).get("content", "") for choice in choices
        )

    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        with httpx.Client(base_url=self.base_url, headers=self._headers(), transport=self._transport, timeout=60.0) as client:
            response = client.post(
                "/chat/completions",
                json=self._payload(messages, model, temperature, stream=False),
            )
            response.raise_for_status()
            return self._extract_text(response.json())

    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        with httpx.Client(base_url=self.base_url, headers=self._headers(), transport=self._transport, timeout=60.0) as client:
            with client.stream(
                "POST",
                "/chat/completions",
                json=self._payload(messages, model, temperature, stream=True),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        choice = chunk.get("choices", [])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
