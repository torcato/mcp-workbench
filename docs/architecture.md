# MCP Workbench Architecture

This document describes the initial architecture for MCP Workbench phase 1.

## Goals

- bootstrap a clean FastAPI-based backend
- separate configuration, logging, and API routing
- keep the first phase minimal and runnable
- prepare the codebase for later MCP, authentication, and tool integration

## Package layout

- `app/`
  - `main.py` - FastAPI application factory and entrypoint
  - `config.py` - typed environment configuration with Pydantic
  - `logging.py` - structured logging setup
  - `api/` - HTTP route modules

## Runtime

- `uvicorn app.main:app --reload` starts the application
- `/health` exposes a simple health check
- configuration is loaded from environment variables and `.env`
- logging is emitted in structured JSON format
