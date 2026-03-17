# tests/test_fallback_brain.py

import logging
import os
from unittest.mock import Mock, patch

import pytest
from mesa.model import Model

from mesa_llm.fallback_brain import FallbackBrain
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backup_llm(response="backup response"):
    """Return a mock ModuleLLM whose generate() returns *response*."""
    backup = Mock()
    backup.generate = Mock(return_value=response)
    return backup


def _make_agent(model=None, fallback_brain=None):
    """Create a minimal LLMAgent with mocked memory/LLM."""
    if model is None:
        model = Model(seed=42)

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        agents = LLMAgent.create_agents(
            model,
            n=1,
            reasoning=ReActReasoning,
            system_prompt="Test",
            fallback_brain=fallback_brain,
        )
    agent = agents.to_list()[0]
    # Silence memory calls so tests don't need a real LLM for memory ops.
    agent.memory.add_to_memory = Mock()
    agent.memory.process_step = Mock()
    return agent


# ---------------------------------------------------------------------------
# FallbackBrain unit tests
# ---------------------------------------------------------------------------


class TestFallbackBrainInit:
    def test_defaults(self):
        backup = _make_backup_llm()
        fb = FallbackBrain(backup_llm=backup)
        assert fb.backup_llm is backup
        assert fb.max_retries == 3
        assert fb.fallback_action is None
        assert fb.failure_counts == {}

    def test_custom_params(self):
        backup = _make_backup_llm()

        def wait_action(agent):
            return {"action": "wait"}

        fb = FallbackBrain(backup_llm=backup, max_retries=5, fallback_action=wait_action)
        assert fb.max_retries == 5
        assert fb.fallback_action is wait_action


class TestFallbackBrainHandleFailure:
    def _make_fb_and_agent(self, max_retries=3, fallback_action=None, backup_response="ok"):
        backup = _make_backup_llm(response=backup_response)
        fb = FallbackBrain(
            backup_llm=backup,
            max_retries=max_retries,
            fallback_action=fallback_action,
        )
        agent = Mock()
        agent.unique_id = 1
        agent.llm = Mock()
        return fb, agent

    def test_first_failure_calls_backup_llm(self):
        fb, agent = self._make_fb_and_agent(max_retries=3)
        result = fb.handle_failure(agent, "test prompt", RuntimeError("primary failed"))

        fb.backup_llm.generate.assert_called_once_with("test prompt")
        assert result == "ok"
        assert fb.failure_counts[1] == 1

    def test_failure_count_increments(self):
        fb, agent = self._make_fb_and_agent(max_retries=3)
        for i in range(1, 4):
            fb.handle_failure(agent, None, RuntimeError("err"))
            assert fb.failure_counts[1] == i

    def test_within_retries_uses_backup(self):
        fb, agent = self._make_fb_and_agent(max_retries=3)
        # Three failures - all within retry limit
        for _ in range(3):
            result = fb.handle_failure(agent, "prompt", RuntimeError("fail"))
        assert fb.backup_llm.generate.call_count == 3
        assert result == "ok"

    def test_exceeds_retries_degrades_agent(self):
        fb, agent = self._make_fb_and_agent(max_retries=1)
        # First failure - within limit
        fb.handle_failure(agent, None, RuntimeError("fail"))
        # Second failure - exceeds limit, should degrade
        result = fb.handle_failure(agent, None, RuntimeError("fail again"))

        # Backup LLM should only have been called once (during first failure)
        assert fb.backup_llm.generate.call_count == 1
        # Agent LLM nullified
        assert agent.llm is None
        # Default idle result
        assert result == {"action": "idle"}

    def test_exceeds_retries_calls_fallback_action(self):
        action_called_with = []

        def my_action(ag):
            action_called_with.append(ag)
            return {"action": "patrol"}

        fb, agent = self._make_fb_and_agent(max_retries=1, fallback_action=my_action)
        fb.handle_failure(agent, None, RuntimeError("first"))
        result = fb.handle_failure(agent, None, RuntimeError("second"))

        assert action_called_with == [agent]
        assert result == {"action": "patrol"}

    def test_backup_failure_falls_through_to_degrade(self):
        """If backup LLM also fails on every attempt, agent is eventually degraded."""
        backup = Mock()
        backup.generate = Mock(side_effect=RuntimeError("backup also broken"))
        fb = FallbackBrain(backup_llm=backup, max_retries=2)
        agent = Mock()
        agent.unique_id = 42
        agent.llm = Mock()

        # Three failures - first two within limit (but backup always raises)
        fb.handle_failure(agent, None, RuntimeError("1"))
        fb.handle_failure(agent, None, RuntimeError("2"))
        # Third failure - exceeds limit
        result = fb.handle_failure(agent, None, RuntimeError("3"))

        # Backup was called for the first two attempts only
        assert backup.generate.call_count == 2
        assert agent.llm is None
        assert result == {"action": "idle"}

    def test_tracks_failures_per_agent(self):
        fb = FallbackBrain(backup_llm=_make_backup_llm(), max_retries=10)
        agent_a = Mock()
        agent_a.unique_id = 1
        agent_a.llm = Mock()
        agent_b = Mock()
        agent_b.unique_id = 2
        agent_b.llm = Mock()

        fb.handle_failure(agent_a, None, RuntimeError("a"))
        fb.handle_failure(agent_a, None, RuntimeError("a"))
        fb.handle_failure(agent_b, None, RuntimeError("b"))

        assert fb.failure_counts[1] == 2
        assert fb.failure_counts[2] == 1

    def test_warning_logged_within_retries(self, caplog):
        fb, agent = self._make_fb_and_agent(max_retries=3)
        with caplog.at_level(logging.WARNING):
            fb.handle_failure(agent, None, RuntimeError("fail"))
        assert any("primary brain failed" in r.message for r in caplog.records)

    def test_error_logged_on_degradation(self, caplog):
        fb, agent = self._make_fb_and_agent(max_retries=0)
        with caplog.at_level(logging.ERROR):
            fb.handle_failure(agent, None, RuntimeError("fail"))
        assert any("Degrading" in r.message for r in caplog.records)


class TestDegradeToDeterministic:
    def test_sets_llm_to_none(self):
        fb = FallbackBrain(backup_llm=_make_backup_llm())
        agent = Mock()
        agent.llm = Mock()
        fb.degrade_to_deterministic(agent)
        assert agent.llm is None

    def test_returns_idle_when_no_action(self):
        fb = FallbackBrain(backup_llm=_make_backup_llm())
        agent = Mock()
        agent.llm = Mock()
        result = fb.degrade_to_deterministic(agent)
        assert result == {"action": "idle"}

    def test_calls_and_returns_fallback_action(self):
        called = []

        def action(ag):
            called.append(ag)
            return {"action": "random_walk"}

        fb = FallbackBrain(backup_llm=_make_backup_llm(), fallback_action=action)
        agent = Mock()
        agent.llm = Mock()
        result = fb.degrade_to_deterministic(agent)

        assert called == [agent]
        assert result == {"action": "random_walk"}


# ---------------------------------------------------------------------------
# LLMAgent integration tests
# ---------------------------------------------------------------------------


class TestLLMAgentFallbackBrainParameter:
    def test_fallback_brain_stored_on_agent(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            model = Model(seed=42)
            fb = FallbackBrain(backup_llm=_make_backup_llm())
            agent = _make_agent(model=model, fallback_brain=fb)
        assert agent.fallback_brain is fb

    def test_default_fallback_brain_is_none(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            model = Model(seed=42)
            agent = _make_agent(model=model)
        assert agent.fallback_brain is None


class TestLLMAgentStepWrapper:
    """Tests for __init_subclass__ wrapped step behaviour."""

    def _make_model_and_agent_class(self, fallback_brain=None):
        """Returns a (model, AgentClass) pair where AgentClass has a custom step."""
        model = Model(seed=42)

        class TestAgent(LLMAgent):
            def step(self):
                raise RuntimeError("primary LLM failed")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            agents = TestAgent.create_agents(
                model,
                n=1,
                reasoning=ReActReasoning,
                system_prompt="Test",
                fallback_brain=fallback_brain,
            )
        agent = agents.to_list()[0]
        agent.memory.add_to_memory = Mock()
        agent.memory.process_step = Mock()
        return model, agent

    def test_step_exception_routes_to_fallback_brain(self):
        backup = _make_backup_llm(response={"action": "wait"})
        fb = FallbackBrain(backup_llm=backup, max_retries=3)
        _, agent = self._make_model_and_agent_class(fallback_brain=fb)

        agent.step()

        assert fb.failure_counts[agent.unique_id] == 1
        backup.generate.assert_called_once()

    def test_step_exception_reraises_without_fallback_brain(self):
        _, agent = self._make_model_and_agent_class(fallback_brain=None)

        with pytest.raises(RuntimeError, match="primary LLM failed"):
            agent.step()

    def test_degraded_agent_skips_step_calls_fallback_action(self):
        """Once llm is None the wrapped step should invoke fallback_action directly."""
        calls = []

        def idle_action(ag):
            calls.append(ag)
            return {"action": "idle"}

        fb = FallbackBrain(backup_llm=_make_backup_llm(), fallback_action=idle_action)
        _, agent = self._make_model_and_agent_class(fallback_brain=fb)

        # Manually degrade the agent.
        agent.llm = None

        result = agent.step()

        assert calls == [agent]
        assert result == {"action": "idle"}

    def test_degraded_agent_without_fallback_action_returns_idle(self):
        fb = FallbackBrain(backup_llm=_make_backup_llm())
        _, agent = self._make_model_and_agent_class(fallback_brain=fb)

        agent.llm = None

        result = agent.step()

        assert result == {"action": "idle"}

    def test_degraded_agent_without_fallback_brain_returns_idle(self):
        _, agent = self._make_model_and_agent_class(fallback_brain=None)

        agent.llm = None

        result = agent.step()

        assert result == {"action": "idle"}
