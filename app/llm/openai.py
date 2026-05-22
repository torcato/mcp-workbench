from __future__ import annotations

import json
from uuid import uuid4
from typing import Iterator

import httpx

from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition


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

    def _payload(
        self,
        messages: list[ChatMessage],
        model: str | None,
        temperature: float,
        stream: bool,
        tools: list[ToolDefinition] | None = None,
    ) -> dict:
        payload = {
            "model": model or self.default_model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"
        return payload

    def _extract_completion(self, response_json: dict) -> ChatCompletion:
        choices = response_json.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from completion response")

        message = choices[0].get("message", {})
        tool_calls = []
        for raw_tool_call in message.get("tool_calls") or []:
            function = raw_tool_call.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tool call arguments for {function.get('name')}") from exc

            if not isinstance(arguments, dict):
                raise ValueError(f"Tool call arguments for {function.get('name')} must be a JSON object")

            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.get("id") or f"call_{uuid4().hex}",
                    name=function.get("name") or "",
                    arguments=arguments,
                    raw=raw_tool_call,
                )
            )

        return ChatCompletion(content=message.get("content"), tool_calls=tool_calls)

    def chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> str:
        completion = self.complete_chat(messages, model=model, temperature=temperature)
        return completion.content or ""

    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        with httpx.Client(base_url=self.base_url, headers=self._headers(), transport=self._transport, timeout=60.0) as client:
            response = client.post(
                "/chat/completions",
                json=self._payload(messages, model, temperature, stream=False, tools=tools),
            )
            response.raise_for_status()
            return self._extract_completion(response.json())

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
