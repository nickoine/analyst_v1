# Built-in
from __future__ import annotations

# Internal
from agents.base import BaseAgent


class AnalyzerAgent(BaseAgent):
    """Concrete analyzer agent."""

    _supports_actions = False
    _supports_knowledge = False
    _supports_memory = False

    _parallel_tool_calls = False
    _tool_choice = "auto"
    _temperature = 0
