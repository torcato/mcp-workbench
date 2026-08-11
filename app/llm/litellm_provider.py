from __future__ import annotations

import json
from types import ModuleType
from typing import Any, Iterator
from uuid import uuid4

from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition


class LiteLLMRequestError(RuntimeError):
    """Raised when LiteLLM fails to complete a request."""


class LiteLLMProvider(LLMProvider):
    def __init__(
        self,
        default_model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 60.0,
        num_retries: int = 5,
        use_system_prompt: bool = True,
        vertex_ai_project: str | None = None,
        vertex_ai_location: str | None = None,
        vertex_ai_credentials_path: str | None = None,
        litellm_module: Any | None = None,
    ) -> None:
        if not default_model or not default_model.strip():
            raise ValueError("LiteLLM model is required")
        if timeout_seconds <= 0:
            raise ValueError("LiteLLM timeout must be greater than zero")
        if num_retries < 0:
            raise ValueError("LiteLLM retries cannot be negative")

        self.default_model = default_model.strip()
        self.api_key = api_key
        self.api_base = api_base.rstrip("/") if api_base else None
        self.timeout_seconds = timeout_seconds
        self.num_retries = num_retries
        self.use_system_prompt = use_system_prompt
        self.vertex_ai_project = vertex_ai_project
        self.vertex_ai_location = vertex_ai_location
        self.vertex_ai_credentials_path = vertex_ai_credentials_path
        self._litellm = litellm_module or _load_litellm()

    @property
    def provider_name(self) -> str:
        return "litellm"

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
        try:
            response = self._litellm.completion(
                **self._completion_kwargs(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    stream=False,
                    tools=tools,
                )
            )
        except Exception as exc:
            raise LiteLLMRequestError(f"LiteLLM completion failed: {exc}") from exc

        return self._extract_completion(response)

    def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 1.0,
    ) -> Iterator[str]:
        try:
            response = self._litellm.completion(
                **self._completion_kwargs(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    stream=True,
                )
            )
            for chunk in response:
                choices = _get_value(chunk, "choices", [])
                if not choices:
                    continue
                delta = _get_value(choices[0], "delta", {})
                content = _get_value(delta, "content")
                if content:
                    yield content
        except Exception as exc:
            raise LiteLLMRequestError(f"LiteLLM streaming completion failed: {exc}") from exc

    def _completion_kwargs(
        self,
        messages: list[ChatMessage],
        model: str | None,
        temperature: float,
        stream: bool,
        tools: list[ToolDefinition] | None = None,
    ) -> dict[str, Any]:
        target_model = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": self._messages(messages),
            "temperature": temperature,
            "stream": stream,
            "num_retries": self.num_retries,
            "request_timeout": self.timeout_seconds,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base and not target_model.startswith("vertex_ai/"):
            kwargs["api_base"] = self.api_base
        if tools:
            kwargs["tools"] = _tool_payload(tools)
            kwargs["tool_choice"] = "auto"
        if target_model.startswith("vertex_ai/"):
            kwargs.update(self._vertex_ai_kwargs())
        return kwargs

    def _messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [
            message.model_dump(exclude_none=True)
            for message in messages
            if self.use_system_prompt or message.role != "system"
        ]

    def _vertex_ai_kwargs(self) -> dict[str, str]:
        kwargs = {}
        if self.vertex_ai_project:
            kwargs["vertex_project"] = self.vertex_ai_project
        if self.vertex_ai_location:
            kwargs["vertex_location"] = self.vertex_ai_location
        if self.vertex_ai_credentials_path:
            kwargs["vertex_credentials"] = self.vertex_ai_credentials_path
        return kwargs

    def _extract_completion(self, response: Any) -> ChatCompletion:
        choices = _get_value(response, "choices", [])
        if not choices:
            raise ValueError("No choices returned from LiteLLM completion response")

        message = _get_value(choices[0], "message", {})
        return ChatCompletion(
            content=_get_value(message, "content"),
            tool_calls=_extract_tool_calls(message),
        )


def _load_litellm() -> ModuleType:
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError(
            "LiteLLM provider requires the 'litellm' package. "
            "Install project dependencies before using LLM_PROVIDER=litellm."
        ) from exc

    litellm.suppress_debug_info = True
    return litellm


def _tool_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
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


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    tool_calls = []
    for raw_tool_call in _get_value(message, "tool_calls", []) or []:
        function = _get_value(raw_tool_call, "function", {})
        raw_arguments = _get_value(function, "arguments") or "{}"
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid tool call arguments for {_get_value(function, 'name')}") from exc

        if not isinstance(arguments, dict):
            raise ValueError(f"Tool call arguments for {_get_value(function, 'name')} must be a JSON object")

        tool_calls.append(
            ToolCall(
                id=_get_value(raw_tool_call, "id") or f"call_{uuid4().hex}",
                name=_get_value(function, "name") or "",
                arguments=arguments,
                raw=_as_dict(raw_tool_call),
            )
        )
    return tool_calls


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    return {}
