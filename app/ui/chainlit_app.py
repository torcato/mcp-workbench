from __future__ import annotations

import chainlit as cl

from app.config import get_settings
from app.mcp.manager import MCPManager
from app.prompts.manager import PromptManager
from app.ui.runtime import (
    NO_MCP_SERVER,
    ChatSessionState,
    apply_session_settings,
    build_initial_messages,
    build_ui_options,
    create_provider,
    run_chat_turn,
)


STATE_KEY = "chat_state"
PROMPT_MANAGER_KEY = "prompt_manager"
MCP_MANAGER_KEY = "mcp_manager"


class ChainlitTokenStream:
    def __init__(self, message: cl.Message) -> None:
        self.message = message

    async def send_token(self, token: str) -> None:
        await self.message.stream_token(token)


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    prompt_manager = PromptManager(settings.prompt_profiles_path, default_profile=settings.default_prompt_profile)
    mcp_manager = MCPManager(settings.mcp_servers)
    options = build_ui_options(settings, prompt_manager, mcp_manager)

    initial_model = settings.llm_default_model if settings.llm_default_model in options.models else options.models[0]
    initial_profile = (
        settings.default_prompt_profile
        if settings.default_prompt_profile in options.prompt_profiles
        else options.prompt_profiles[0]
    )

    cl.user_session.set(PROMPT_MANAGER_KEY, prompt_manager)
    cl.user_session.set(MCP_MANAGER_KEY, mcp_manager)
    cl.user_session.set(
        STATE_KEY,
        ChatSessionState(
            model=initial_model,
            prompt_profile=initial_profile,
            mcp_server=NO_MCP_SERVER,
            messages=build_initial_messages(prompt_manager, initial_profile),
        ),
    )

    await cl.ChatSettings(
        [
            cl.Select(
                id="model",
                label="Model",
                values=options.models,
                initial_index=options.models.index(initial_model),
            ),
            cl.Select(
                id="prompt_profile",
                label="Prompt profile",
                values=options.prompt_profiles,
                initial_index=options.prompt_profiles.index(initial_profile),
            ),
            cl.Select(
                id="mcp_server",
                label="MCP server",
                values=options.mcp_servers,
                initial_index=0,
            ),
        ]
    ).send()


@cl.on_settings_update
async def on_settings_update(settings_update: dict) -> None:
    prompt_manager = cl.user_session.get(PROMPT_MANAGER_KEY)
    mcp_manager = cl.user_session.get(MCP_MANAGER_KEY)
    state = cl.user_session.get(STATE_KEY)

    state = await apply_session_settings(
        state,
        settings_update,
        prompt_manager=prompt_manager,
        mcp_manager=mcp_manager,
    )
    cl.user_session.set(STATE_KEY, state)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = get_settings()
    state = cl.user_session.get(STATE_KEY)
    mcp_manager = cl.user_session.get(MCP_MANAGER_KEY)
    response = cl.Message(content="")
    await response.send()

    try:
        provider = create_provider(settings)
        state = await run_chat_turn(
            state,
            message.content,
            provider=provider,
            mcp_manager=mcp_manager,
            token_stream=ChainlitTokenStream(response),
        )
    except Exception as exc:
        await response.stream_token(f"Chat failed: {exc}")
    else:
        cl.user_session.set(STATE_KEY, state)

    await response.update()


@cl.on_chat_end
async def on_chat_end() -> None:
    mcp_manager = cl.user_session.get(MCP_MANAGER_KEY)
    if mcp_manager is not None:
        await mcp_manager.disconnect_all()
