"""Integration tests: Memory backend x Reasoning strategy matrix.

Verifies that each memory implementation works correctly with each
reasoning strategy, testing the full flow:
    memory.get_prompt_ready() -> reasoning prompt construction ->
    reasoning.plan() / aplan() -> memory.add_to_memory()

These tests use real memory instances (not mocks) combined with
mocked LLM responses to isolate integration issues without
requiring API keys.
"""

import json
from unittest.mock import Mock

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_mock_agent(memory_instance, *, step_prompt="You are an agent in a simulation"):
    """Build a mock agent wired to a *real* memory instance."""
    agent = Mock()
    agent.__class__.__name__ = "TestAgent"
    agent.unique_id = 1
    agent.model = Mock()
    agent.model.steps = 1
    agent.step_prompt = step_prompt
    agent.llm = Mock()
    agent.tool_manager = Mock()
    agent.tool_manager.get_all_tools_schema.return_value = {}
    agent._step_display_data = {}

    # Wire memory
    agent.memory = memory_instance
    memory_instance.agent = agent
    memory_instance.display = False  # suppress rich output in tests

    return agent


def make_llm_response(content="mock plan content"):
    """Create a minimal mock LLM response."""
    rsp = Mock()
    rsp.choices = [Mock()]
    rsp.choices[0].message = Mock()
    rsp.choices[0].message.content = content
    rsp.choices[0].message.tool_calls = None
    return rsp


def make_react_response():
    """Create a mock LLM response in ReActOutput JSON format."""
    content = json.dumps({"reasoning": "test reasoning", "action": "test action"})
    return make_llm_response(content)


def seed_memory(memory, agent, n=2):
    """Add n dummy entries to memory so get_prompt_ready has content."""
    for i in range(n):
        memory.add_to_memory(type="observation", content={"info": f"step {i}"})
        memory.process_step(pre_step=True)
        agent.model.steps = i + 1
        memory.process_step(pre_step=False)
