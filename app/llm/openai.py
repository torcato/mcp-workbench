from __future__ import annotations

import email.utils
import json
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Iterator
from uuid import uuid4

import httpx

from app.llm.base import ChatCompletion, ChatMessage, LLMProvider, ToolCall, ToolDefinition


MAX_ERROR_DETAIL_LENGTH = 1000
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4.1-mini",
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 0,
        retry_initial_delay_seconds: float = 1.0,
        retry_max_delay_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for OpenAI-compatible providers")
        if timeout_seconds <= 0:
            raise ValueError("OpenAI-compatible timeout must be greater than zero")
        if max_retries < 0:
            raise ValueError("OpenAI-compatible retries cannot be negative")
        if retry_initial_delay_seconds <= 0:
            raise ValueError("OpenAI-compatible retry initial delay must be greater than zero")
        if retry_max_delay_seconds <= 0:
            raise ValueError("OpenAI-compatible retry max delay must be greater than zero")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._transport = transport
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_initial_delay_seconds = retry_initial_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self._sleep = sleep

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

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _extract_error_detail(response)
            if detail:
                raise httpx.HTTPStatusError(
                    f"{exc}\nResponse body: {detail}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            raise

    def complete_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        model: str | None = None,
        temperature: float = 1.0,
    ) -> ChatCompletion:
        with httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            transport=self._transport,
            timeout=self.timeout_seconds,
        ) as client:
            response = self._request_with_retries(
                lambda: client.post(
                    "/chat/completions",
                    json=self._payload(messages, model, temperature, stream=False, tools=tools),
                )
            )
            return self._extract_completion(response.json())

    def stream_chat(self, messages: list[ChatMessage], model: str | None = None, temperature: float = 1.0) -> Iterator[str]:
        with httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            transport=self._transport,
            timeout=self.timeout_seconds,
        ) as client:
            for line in self._stream_lines_with_retries(
                lambda: client.stream(
                    "POST",
                    "/chat/completions",
                    json=self._payload(messages, model, temperature, stream=True),
                )
            ):
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

    def _request_with_retries(self, send: Callable[[], httpx.Response]) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = send()
                if not _is_retryable_status(response.status_code) or attempt >= self.max_retries:
                    self._raise_for_status(response)
                    return response
                self._sleep(self._retry_delay(attempt, response))
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                self._sleep(self._retry_delay(attempt, None))

        raise RuntimeError("Retry loop exhausted unexpectedly")

    def _stream_lines_with_retries(
        self,
        stream: Callable[[], AbstractContextManager[httpx.Response]],
    ) -> Iterator[str]:
        for attempt in range(self.max_retries + 1):
            try:
                with stream() as response:
                    if not _is_retryable_status(response.status_code) or attempt >= self.max_retries:
                        self._raise_for_status(response)
                        yield from response.iter_lines()
                        return
                    self._sleep(self._retry_delay(attempt, response))
            except httpx.TransportError:
                if attempt >= self.max_retries:
                    raise
                self._sleep(self._retry_delay(attempt, None))

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            return min(retry_after, self.retry_max_delay_seconds)
        delay = self.retry_initial_delay_seconds * (2 ** attempt)
        return min(delay, self.retry_max_delay_seconds)


def _extract_error_detail(response: httpx.Response) -> str | None:
    try:
        response.read()
    except httpx.HTTPError:
        pass

    try:
        text = response.text.strip()
    except RuntimeError:
        return None

    if not text:
        return None

    try:
        payload = response.json()
    except ValueError:
        return _truncate_error_detail(text)

    error = _extract_error_payload(payload)
    if error:
        message = error.get("message")
        status = error.get("status")
        code = error.get("code")
        parts = [str(part) for part in (status, code, message) if part]
        if parts:
            return _truncate_error_detail(" ".join(parts))

    return _truncate_error_detail(json.dumps(payload, separators=(",", ":")))


def _extract_error_payload(payload: object) -> dict | None:
    if isinstance(payload, dict):
        error = payload.get("error")
        return error if isinstance(error, dict) else None
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                error = item.get("error")
                if isinstance(error, dict):
                    return error
    return None


def _truncate_error_detail(detail: str) -> str:
    if len(detail) <= MAX_ERROR_DETAIL_LENGTH:
        return detail
    return f"{detail[:MAX_ERROR_DETAIL_LENGTH]}..."


def _is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None

    value = response.headers.get("retry-after")
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()

    return max(0.0, seconds)
