# MCP Workbench

A self-hosted Python platform for interacting with LLMs through the Model Context Protocol (MCP).

MCP Workbench provides a modern chat interface with first-class support for:

- MCP tools
- MCP resources
- MCP prompts
- remote MCP servers
- Streamable HTTP
- SSE
- OAuth authentication
- configurable system prompts
- multiple LLM providers
- tool governance and approval workflows

Built for engineers who want full control over prompts, integrations, authentication, and tooling.

---

# Why MCP Workbench?

Most AI chat tools optimize for convenience.

MCP Workbench optimizes for:

- protocol correctness
- extensibility
- Python hackability
- secure remote MCP integration
- prompt control
- observability
- explicit tool safety

The design goal is simple:

> Build the MCP-native AI workbench engineers actually want.

---

# Features

## MCP Native

Full MCP client support:

- multiple simultaneous MCP servers
- capability discovery
- tools
- resources
- prompts
- metadata inspection

Supported transports:

- stdio
- SSE
- Streamable HTTP

Authentication:

- OAuth 2.0
- PKCE
- token refresh
- encrypted credential storage

---

## Prompt Control

Configurable prompting:

- system prompt profiles
- per-session overrides
- MCP prompt composition
- reusable personas

Examples:

- General assistant
- Senior backend engineer
- RAG evaluator
- DevOps operator
- Security reviewer

---

## LLM Provider Agnostic

Supports OpenAI-compatible APIs:

- OpenAI
- Ollama
- llama.cpp
- LM Studio
- vLLM
- LocalAI

Planned:

- Anthropic
- Gemini
- Azure OpenAI

---

## Tool Governance

Safety controls:

- approval workflows
- allowlists
- denylists
- destructive action confirmation
- execution policies
- audit logging

Example policies:

- auto-approve document search
- require approval for GitHub writes
- deny shell execution

---

## Chat Experience

- streaming responses
- session history
- model selection
- prompt profile switching
- MCP server enable / disable
- OAuth connect / disconnect
- transparent tool invocation

---

# Architecture

High-level design:

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

Storage:

- SQLite (v1)
- PostgreSQL (v2)

---

## UI

Initial:

- Chainlit

Future:

- React / Next.js frontend

---

## LLM Providers

OpenAI-compatible abstraction:

- OpenAI
- Ollama
- llama.cpp
- vLLM
- LocalAI

---

## Auth

- OAuth 2.0
- PKCE
- encrypted token persistence

---

## Observability

Planned:

- structured logging
- Prometheus
- OpenTelemetry

---

# Project Structure

```text
mcp-workbench/
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

# Configuration

## MCP Servers

Example:

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
      Use tools when useful.
```

---

# Phase 1 Bootstrap

## Run locally

1. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

2. Start the app:

```bash
uvicorn app.main:app --reload
```

3. Verify health:

```bash
curl http://127.0.0.1:8000/health
```

## Tests

```bash
python -m pip install -e .[test]
pytest
```

  coding:
    name: Senior Backend Engineer
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

# Installation

## Clone

```bash
git clone https://github.com/your-org/mcp-workbench.git
cd mcp-workbench
```

---

## Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Install

```bash
pip install -e .[test]
```

---

## Configure

Create:

```bash
.env
```

Example:

```env
MCP_WORKBENCH_LLM_API_KEY=your-key
MCP_WORKBENCH_LLM_BASE_URL=https://api.openai.com/v1
MCP_WORKBENCH_LLM_DEFAULT_MODEL=gpt-3.5-turbo
```

---

## Run Backend

```bash
uvicorn app.main:app --reload
```

---

## Run UI

```bash
chainlit run app/ui/chainlit_app.py
```

---

# Development Roadmap

## v1 — MVP

Core platform:

- [ ] chat UI
- [ ] OpenAI-compatible provider layer
- [ ] MCP tools
- [ ] MCP resources
- [ ] MCP prompts
- [ ] SSE support
- [ ] Streamable HTTP support
- [ ] OAuth authentication
- [ ] prompt profiles
- [ ] tool approvals
- [ ] SQLite persistence

---

## v2 — Production Readiness

- [ ] PostgreSQL
- [ ] Redis
- [ ] multi-user authentication
- [ ] RBAC
- [ ] audit dashboard
- [ ] observability
- [ ] custom frontend

---

## v3 — Advanced Features

- [ ] MCP broker mode
- [ ] workflow chaining
- [ ] plugin architecture
- [ ] memory modules
- [ ] document ingestion
- [ ] RAG integrations
- [ ] prompt marketplace

---

# Security Model

MCP tools can perform real external actions.

Security assumptions:

- tools are not automatically trusted
- destructive actions require approval
- credentials must be encrypted
- remote auth must be explicit

Protections:

- approval workflows
- execution policies
- audit logging
- OAuth isolation
- server-level access controls

---

# Design Principles

MCP Workbench prefers:

- explicit control
- protocol correctness
- composable architecture
- observability
- safe tool execution
- provider neutrality
- debuggability

This is intentionally not:

- a black-box autonomous agent
- a LangChain abstraction layer
- a workflow automation platform
- a "magic AI assistant"

---

# Intended Users

Best suited for:

- backend engineers
- platform engineers
- DevOps teams
- AI infrastructure teams
- MCP server developers
- self-hosters
- tool integration teams

---

# Contributing

Planned.

Potential contribution areas:

- provider adapters
- MCP integrations
- auth providers
- frontend UX
- prompt packs
- observability integrations
- governance modules

---

# License

TBD

Recommended:

- MIT
or
- Apache-2.0

---

# References

Model Context Protocol:

https://modelcontextprotocol.io

Python SDK:

https://github.com/modelcontextprotocol/python-sdk
