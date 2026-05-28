from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from chainlit.input_widget import Select

from app.chat.tool_loop import MCPToolExecution
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
    resolve_initial_mcp_server,
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


class ChainlitToolExecutionStream:
    def __init__(self, parent_id: str | None = None) -> None:
        self.parent_id = parent_id

    async def send_execution(self, execution: MCPToolExecution) -> None:
        status = "allowed" if execution.allowed else "blocked"
        name = f"{execution.server_name}/{execution.tool_name}" if execution.server_name else execution.tool_name
        async with cl.Step(
            name=f"{name} ({status})",
            type="tool",
            parent_id=self.parent_id,
            default_open=False,
            show_input="json",
        ) as step:
            step.input = execution.arguments
            step.output = execution.result
            step.is_error = not execution.allowed or execution.result.startswith("Tool returned an error:")


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
    initial_mcp_server = resolve_initial_mcp_server(settings, options)

    if initial_mcp_server != NO_MCP_SERVER:
        try:
            await mcp_manager.connect(initial_mcp_server)
        except Exception as exc:
            initial_mcp_server = NO_MCP_SERVER
            await cl.Message(content=f"Default MCP server connection failed: {exc}").send()

    cl.user_session.set(PROMPT_MANAGER_KEY, prompt_manager)
    cl.user_session.set(MCP_MANAGER_KEY, mcp_manager)
    cl.user_session.set(
        STATE_KEY,
        ChatSessionState(
            model=initial_model,
            prompt_profile=initial_profile,
            mcp_server=initial_mcp_server,
            messages=build_initial_messages(prompt_manager, initial_profile),
        ),
    )

    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=options.models,
                initial_index=options.models.index(initial_model),
            ),
            Select(
                id="prompt_profile",
                label="Prompt profile",
                values=options.prompt_profiles,
                initial_index=options.prompt_profiles.index(initial_profile),
            ),
            Select(
                id="mcp_server",
                label="MCP server",
                values=options.mcp_servers,
                initial_index=options.mcp_servers.index(initial_mcp_server),
            ),
        ]
    ).send()


@cl.on_settings_update
async def on_settings_update(settings_update: dict) -> None:
    prompt_manager = cl.user_session.get(PROMPT_MANAGER_KEY)
    mcp_manager = cl.user_session.get(MCP_MANAGER_KEY)
    state = cl.user_session.get(STATE_KEY)

    try:
        state = await apply_session_settings(
            state,
            settings_update,
            prompt_manager=prompt_manager,
            mcp_manager=mcp_manager,
        )
    except Exception as exc:
        await cl.Message(content=f"Settings update failed: {exc}").send()
    else:
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
        tool_stream = ChainlitToolExecutionStream(parent_id=response.id)
        turn_kwargs = {
            "provider": provider,
            "mcp_manager": mcp_manager,
            "token_stream": ChainlitTokenStream(response),
        }
        if "tool_execution_callback" in inspect.signature(run_chat_turn).parameters:
            turn_kwargs["tool_execution_callback"] = tool_stream.send_execution

        state = await run_chat_turn(state, message.content, **turn_kwargs)
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
