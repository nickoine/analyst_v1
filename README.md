# AI Agent Orchestration Platform

A Python-based AI agent orchestration platform for software decommissioning analysis.

The project demonstrates how to build an LLM-powered backend around structured multi-agent execution,
typed inter-agent handoffs, prompt configuration, runtime boundaries, and cost observability.

## Overview

The system receives a multipart HTTP request and executes a three-stage LLM agent pipeline.

Each agent has a dedicated responsibility.
Its output becomes validated structured context for the next stage,
creating a controlled context-chaining pattern instead of passing loose text between agents.

Although the current use case is software decommissioning analysis,
the architecture represents a generic pattern for structured AI agent orchestration.

## Core Idea

> LLM agents should behave like structured pipeline stages, not isolated chat calls.

Each pipeline stage:

- receives defined input
- executes an agent with a dedicated profile
- produces structured output
- validates the output with Pydantic
- passes the validated result forward
- logs execution and token usage data

This makes the workflow easier to inspect, debug, and extend.

## Architecture

The system follows a hexagonal architecture pattern.
The Flask HTTP layer is intentionally thin and only handles transport concerns.

The application core coordinates the use case,
while the agent runtime and LLM client are implemented as infrastructure adapters behind explicit port contracts.

High-level structure:

```text
HTTP Request
    ↓
Flask / Flask-RESTful API Layer
    ↓
Application Facade
    ↓
Analysis Service
    ↓
Agent Gateway Port
    ↓
RuntimeAgentGateway
    ↓
Agent Pipeline
    ↓
LLM Client Port
    ↓
OpenAI-Compatible LLM Adapter
```

## Main Structural Units

The project is organized around six main structural units:

1. HTTP API layer mounted at `/context`
2. Application facade and analysis service
3. Port protocols for the agent gateway and LLM client
4. `RuntimeAgentGateway` as the concrete pipeline orchestrator
5. Three concrete agents backed by YAML-driven profiles
6. LLM infrastructure adapter wrapping the OpenAI Responses API

These components are assembled at startup through `build_application` in `app/bootstrap.py`.

## Dependency Wiring

The application dependency graph is created once during startup.

`build_application` wires the main runtime components:

- `PromptManager`
- `OAIClient`
- `AgentPipeline`
- `RuntimeAgentGateway`
- `AnalysisService`
- `Application`

The resulting dependency graph is held for the lifetime of the Flask process.

This keeps runtime construction separate from HTTP request handling
and avoids hardcoding infrastructure details inside endpoints.

## Agent Pipeline

The system executes three agents in a directed sequential pipeline.

Each agent receives context from the previous stage, performs its own analysis step,
and produces structured output for the next stage.

Per-agent execution state is maintained during a single request lifecycle.

Conceptually:

```text
Agent 1
  → structured output
    → Agent 2
      → structured output
        → Agent 3
          → final structured result
```

This makes the agent workflow explicit, traceable, and easier to reason about.

## Agent Configuration

Agent behavior is driven by `AgentProfile` definitions loaded by `PromptManager` from YAML files in the `prompts` directory.

A typical profile contains:

- `system_content` - agent role, responsibilities, and constraints
- `actions` - optional executable actions or tool-like capabilities
- `output_contract` - structured-output binding, schema format, and request binding strategy
- `knowledge`, `memory`, `evaluators`, `reasoners`, `planners`, `feedback` - optional runtime extension points
- `reasoner_effort` and `reasoner_summary` - reasoning-level configuration for the LLM call
- `prompt_cache` - cache strategy for stable prompt material

Each profile acts as a declarative runtime contract.

It defines:
- what the agent is,
- how it should reason,
- which boundaries it must respect,
- and what validated output shape it must produce.

Each profile also references two sibling XML files:

- instructions
- assistant prefill

This keeps prompt content outside Python code and makes agent behavior easier to inspect, version, and modify.

All prompts and evaluation assets are excluded as proprietary intellectual property.

## Structured Output Enforcement

The project uses Pydantic v2 to enforce structured output contracts across the full pipeline.
Every agent stage must return data matching its declared schema before the result can be passed forward.
This prevents downstream agents from receiving malformed or ambiguous context.
The goal is to replace fragile prompt-chain behavior with explicit runtime contracts.

## LLM Integration

The system uses the OpenAI Python SDK with the Responses API.
The LLM client is wrapped behind a port interface, so the application core does not depend directly on the OpenAI SDK.
LLM routing is configured through environment variables:

```text
OAI_API_KEY
MODEL
BASE_URL
MAX_OUTPUT_TOKENS
```

Because `BASE_URL` is configurable, the system can target OpenAI-compatible providers,
including OpenRouter and other compatible endpoints.

## Cost Observability

The platform logs token usage at two levels:
- individual agent stage
- full pipeline aggregate
This provides basic operational visibility into LLM cost and runtime behavior.

## Prompt Caching

The project includes a SHA256-keyed prompt cache mechanism.
The cache reduces redundant prompt preparation and supports reuse of stable prompt artifacts,
contributing to a more predictable and observable LLM pipeline.

## State Management

The system is intentionally stateless.
There is no database nor message broker. All execution state exists only inside a single request lifecycle.
This keeps the project focused on orchestration mechanics rather than persistence or distributed workflow processing.

## Technology Stack

- Python 3.12
- Flask
- Flask-RESTful
- OpenAI Python SDK
- OpenAI Responses API
- Pydantic v2
- PyYAML

## Key Engineering Decisions

### Thin HTTP Layer

The Flask layer handles only transport concerns.
Business logic and orchestration are delegated to the application service and runtime components.

### Ports and Adapters

The agent gateway and LLM client are defined as protocols.
Concrete implementations sit behind these contracts,
keeping the application core independent of specific infrastructure choices.

### YAML-Driven Agents

Agent behavior is configured through YAML profiles instead of being hardcoded directly into Python classes.
This makes agents easier to inspect, update, and compare.

### Typed Agent Handoffs

Pipeline stages communicate through validated structured outputs.
This improves reliability and makes the flow easier to reason about.

### Request-Scoped State

The system avoids persistence and focuses only on the lifecycle of a single analysis request.
This keeps the orchestration model simple and explicit.

## What This Project Demonstrates

The project demonstrates practical backend and AI engineering patterns for LLM systems:

- multi-agent pipeline orchestration
- structured inter-agent context propagation
- schema-validated LLM outputs
- prompt externalization and configuration
- hexagonal architecture
- OpenAI-compatible LLM adapter design
- cost and token usage observability
- explicit dependency wiring
- separation between HTTP, application logic, runtime, and infrastructure

## Non-Goal

This project does not aim to be a complete SaaS product or fully production decommissioning platform.
Its purpose is to demonstrate architecture and implementation quality in an AI backend system.

## Summary

This project is a structured LLM orchestration backend built around a three-stage agent pipeline.
Its value is not only in the decommissioning analysis use case, but in the engineering pattern behind it:

- clean boundaries,
- typed contracts,
- prompt-driven agents,
- validated handoffs,
- observable execution,
- and OpenAI-compatible runtime integration.

The project shows how an AI pipeline can be implemented as a real backend system
instead of a loose chain of prompts or isolated API calls.

## Author

Designed and implemented by [@nickoine](https://github.com/nickoine).  
LLM tools were used as an auxiliary assistant for review, refinement, and documentation polishing.
