# MCP Chat Platform

A self-hosted Python chat platform with first-class support for the Model Context Protocol (MCP), configurable system prompts, remote MCP servers, and multiple LLM providers.

Built for engineers who want full control over tooling, prompts, authentication, and integrations—not a black-box agent framework.

The platform is designed around the official Python MCP SDK, which supports MCP clients, tools, resources, prompts, and transports including stdio, SSE, and Streamable HTTP. :contentReference[oaicite:0]{index=0}

---

# Features

## MCP Support

- MCP client support
- multiple simultaneous MCP servers
- MCP tools
- MCP resources
- MCP prompts
- capability discovery
- server metadata inspection

Supported transports:

- stdio
- SSE
- Streamable HTTP

Authentication:

- OAuth
- PKCE
- token refresh
- secure credential storage

---

## LLM Provider Support

OpenAI-compatible providers:

- OpenAI
- Ollama
- llama.cpp server
- vLLM
- LM Studio
- LocalAI

Planned:

- Anthropic
- Google Gemini
- Azure OpenAI

---

## Prompt Management

Configurable prompt system:

- system prompt profiles
- per-session overrides
- MCP prompt composition
- role-based personas

Examples:

- Default assistant
- Senior backend engineer
- RAG evaluator
- DevOps operator
- Security reviewer

---

## Tool Safety

Tool execution governance:

- approval workflows
- allowlists
- denylists
- destructive tool confirmation
- audit logging
- per-tool policies

Example:

- allow document search automatically
- require confirmation for GitHub writes
- block shell execution

---

## Chat Experience

- streaming responses
- session history
- model selection
- prompt profile selection
- MCP server enable/disable
- OAuth connect/disconnect
- tool execution transparency

---

# Why This Project?

Most MCP chat clients today optimize for convenience.

This project optimizes for:

- protocol correctness
- extensibility
- Python hackability
- prompt control
- secure remote MCP usage

The target user is an engineer who wants:

> “A self-hosted MCP-native chat platform I can actually extend.”

---

# Architecture

High-level architecture:

```text
Browser UI
   │
   ▼
Chainlit / Web Frontend
   │
   ▼
FastAPI Backend
   │
   ▼
Agent Orchestrator
   ├── Prompt Manager
   ├── MCP Manager
   ├── OAuth Manager
   ├── Tool Policy Engine
   ├── Session Manager
   └── LLM Provider Layer
   │
   ├── OpenAI-compatible providers
   └── Remote MCP servers
```

Detailed design:

See:

```text
docs/architecture.md
```

---

# Tech Stack

## Backend

- Python 3.11+
- FastAPI
- official MCP Python SDK
- SQLAlchemy
- SQLite (v1)
- PostgreSQL (v2)

---

## UI

Initial:

- Chainlit

Future:

- React / Next.js frontend

---

## LLM

Provider abstraction over:

- OpenAI APIs
- OpenAI-compatible APIs
- Ollama
- llama.cpp

---

## Auth

- OAuth 2.0
- PKCE
- encrypted token storage

---

## Observability

Planned:

- structured logging
- Prometheus
- OpenTelemetry

---

# Project Structure

```text
mcp-chat/
  app/
    main.py
    config.py

    api/
      chat.py
      oauth.py
      config.py
      mcp.py

    llm/
      base.py
      openai.py
      ollama.py
      llamacpp.py

    mcp/
      manager.py
      client.py
      auth.py
      registry.py

    prompts/
      manager.py
      profiles.yaml

    tools/
      policy.py
      approvals.py

    sessions/
      memory.py
      storage.py

    ui/
      chainlit_app.py

  docs/
    architecture.md

  tests/

  docker/
```

---

# Example Configuration

## MCP Servers

```yaml
servers:
  github:
    transport: streamable_http
    url: https://example.com/mcp/github
    oauth: true
    enabled: true

  jira:
    transport: sse
    url: https://jira.example.com/mcp
    oauth: true

  docs:
    transport: stdio
    command: python
    args:
      - docs_server.py
```

---

## Prompt Profiles

```yaml
profiles:
  default:
    name: Default Assistant
    system_prompt: |
      You are a pragmatic assistant.
      Use tools when appropriate.

  coding:
    name: Senior Engineer
    system_prompt: |
      You are a senior backend engineer.
      Prefer inspection before action.
      Be concise.

  rag_eval:
    name: RAG Evaluator
    system_prompt: |
      Evaluate retrieval quality.
      Be strict.
      Score semantic similarity.
```

---

## Tool Policies

```yaml
tool_policies:
  docs.search:
    approval: auto

  github.create_issue:
    approval: ask

  github.delete_repo:
    approval: always

  shell.exec:
    approval: deny
```

---

# Installation (Planned)

## Clone

```bash
git clone https://github.com/your-org/mcp-chat.git
cd mcp-chat
```

---

## Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure

Create:

```bash
.env
```

Example:

```env
OPENAI_API_KEY=...
DATABASE_URL=sqlite:///./app.db
APP_SECRET_KEY=change-me
```

---

## Run

```bash
uvicorn app.main:app --reload
```

or

```bash
chainlit run app/ui/chainlit_app.py
```

---

# Development Roadmap

## v1

Core MVP:

- [ ] chat UI
- [ ] OpenAI-compatible LLM provider
- [ ] MCP tools
- [ ] MCP resources
- [ ] MCP prompts
- [ ] SSE transport
- [ ] Streamable HTTP transport
- [ ] OAuth login
- [ ] prompt profiles
- [ ] tool approvals
- [ ] SQLite persistence

---

## v2

Production hardening:

- [ ] PostgreSQL
- [ ] Redis
- [ ] multi-user support
- [ ] role-based permissions
- [ ] audit dashboard
- [ ] observability
- [ ] custom frontend

---

## v3

Advanced workflows:

- [ ] MCP broker mode
- [ ] workflow chaining
- [ ] memory plugins
- [ ] document ingestion
- [ ] RAG integrations
- [ ] plugin marketplace

---

# Security Notes

MCP can execute external actions.

This project assumes:

- untrusted tools are dangerous
- destructive tools require approval
- credentials must be encrypted
- remote auth must be explicit

Security features:

- approval workflows
- policy engine
- audit logging
- OAuth token isolation

---

# Design Philosophy

This is intentionally **not**:

- LangChain-first
- agent-magic-heavy
- workflow-over-architecture
- black-box automation

This project prefers:

- explicit control
- protocol correctness
- composable architecture
- debuggability
- Python-native extensibility

---

# Intended Users

Best suited for:

- backend engineers
- platform engineers
- AI infrastructure teams
- self-hosters
- MCP server developers
- tool integration teams

---

# Contributing

Planned.

Likely contribution areas:

- provider adapters
- MCP auth integrations
- frontend UX
- prompt packs
- tool policy modules
- observability integrations

---

# License

TBD

Suggested:

MIT
or
Apache-2.0

---

# References

Model Context Protocol:

:contentReference[oaicite:1]{index=1}

Official Python SDK:

:contentReference[oaicite:2]{index=2}