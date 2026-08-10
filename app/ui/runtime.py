from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from app.chat.tool_loop import ChatToolLoop, MCPToolExecution
from app.config import AppSettings
from app.llm.base import ChatMessage, LLMProvider
from app.llm.openai import OpenAIProvider
from app.mcp.manager import MCPManager
from app.prompts.manager import PromptManager


NO_MCP_SERVER = "none"


class TokenStream(Protocol):
    async def send_token(self, token: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class UIOptions:
    models: list[str]
    prompt_profiles: list[str]
    mcp_servers: list[str]


@dataclass
class ChatSessionState:
    model: str
    prompt_profile: str
    mcp_server: str = NO_MCP_SERVER
    messages: list[ChatMessage] = field(default_factory=list)


def build_ui_options(
    settings: AppSettings,
    prompt_manager: PromptManager,
    mcp_manager: MCPManager,
) -> UIOptions:
    models = settings.llm_models or [settings.llm_default_model]
    prompt_profiles = list(prompt_manager.list_profiles())
    mcp_servers = [NO_MCP_SERVER, *mcp_manager.list_servers()]
    return UIOptions(models=models, prompt_profiles=prompt_profiles, mcp_servers=mcp_servers)


def build_initial_messages(prompt_manager: PromptManager, profile_name: str) -> list[ChatMessage]:
    return prompt_manager.compose_prompt(profile_name=profile_name)


def resolve_initial_mcp_server(settings: AppSettings, options: UIOptions) -> str:
    if settings.default_mcp_server and settings.default_mcp_server in options.mcp_servers:
        return settings.default_mcp_server
    return NO_MCP_SERVER


def create_provider(settings: AppSettings) -> LLMProvider:
    provider_name = settings.llm_provider.lower().replace("-", "_")

    if provider_name in {"openai", "openai_compatible"}:
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is required to use the OpenAI-compatible provider")

        return OpenAIProvider(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_model=settings.llm_default_model,
        )

    if provider_name in {"vertex", "vertex_ai"}:
        if not settings.vertex_ai_project:
            raise RuntimeError("VERTEX_AI_PROJECT is required to use the Vertex AI provider")

        from app.llm.vertex import VertexAIProvider

        return VertexAIProvider(
            project=settings.vertex_ai_project,
            location=settings.vertex_ai_location,
            credentials_path=settings.vertex_ai_credentials_path,
            default_model=settings.llm_default_model,
        )

    raise RuntimeError(f"Unsupported LLM provider: {settings.llm_provider}")


async def apply_session_settings(
    state: ChatSessionState,
    settings_update: dict,
    prompt_manager: PromptManager,
    mcp_manager: MCPManager,
) -> ChatSessionState:
    next_model = settings_update.get("model") or state.model
    next_profile = settings_update.get("prompt_profile") or state.prompt_profile
    next_mcp_server = settings_update.get("mcp_server") or state.mcp_server

    if next_mcp_server != state.mcp_server:
        if state.mcp_server != NO_MCP_SERVER and state.mcp_server in mcp_manager.list_connected_servers():
            await mcp_manager.disconnect(state.mcp_server)
        if next_mcp_server != NO_MCP_SERVER:
            await mcp_manager.connect(next_mcp_server)

    if next_profile != state.prompt_profile:
        messages = build_initial_messages(prompt_manager, next_profile)
    else:
        messages = state.messages

    return ChatSessionState(
        model=next_model,
        prompt_profile=next_profile,
        mcp_server=next_mcp_server,
        messages=messages,
    )


async def run_chat_turn(
    state: ChatSessionState,
    user_content: str,
    provider: LLMProvider,
    mcp_manager: MCPManager,
    token_stream: TokenStream,
    tool_execution_callback: Callable[[MCPToolExecution], Awaitable[None]] | None = None,
) -> ChatSessionState:
    conversation = [*state.messages, ChatMessage(role="user", content=user_content)]

    if state.mcp_server == NO_MCP_SERVER:
        assistant_content = ""
        for token in provider.stream_chat(conversation, model=state.model):
            assistant_content += token
            await token_stream.send_token(token)

        conversation.append(ChatMessage(role="assistant", content=assistant_content))
        return ChatSessionState(
            model=state.model,
            prompt_profile=state.prompt_profile,
            mcp_server=state.mcp_server,
            messages=conversation,
        )

    loop = ChatToolLoop(
        provider=provider,
        mcp_manager=mcp_manager,
        tool_execution_callback=tool_execution_callback,
    )
    result = await loop.run(conversation, model=state.model)
    for token in _chunk_text(result.content):
        await token_stream.send_token(token)

    return ChatSessionState(
        model=state.model,
        prompt_profile=state.prompt_profile,
        mcp_server=state.mcp_server,
        messages=result.messages,
    )


def _chunk_text(text: str, size: int = 24) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]
