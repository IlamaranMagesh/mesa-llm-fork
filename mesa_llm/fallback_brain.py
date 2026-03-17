from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent
    from mesa_llm.module_llm import ModuleLLM


class FallbackBrain:
    """
    Handles LLM failures by managing a backup ModuleLLM and gracefully
    degrading an agent to a rule-based state when all generative options fail.

    The FallbackBrain is fully decoupled from the primary LLM - the primary
    brain has no knowledge of its existence.

    Parameters:
        backup_llm (ModuleLLM): A secondary LLM instance used when the primary
            fails (e.g. a local Ollama/vLLM model).
        max_retries (int): Number of times to attempt the backup LLM before
            degrading the agent to a deterministic rule-based agent.
            Defaults to 3.
        fallback_action (Callable | None): Optional callable that accepts the
            degraded agent as its sole argument and returns a deterministic
            action.  Executed once the agent has been permanently degraded.

    Attributes:
        failure_counts (dict[int, int]): Maps each agent's ``unique_id`` to the
            cumulative number of primary-brain failures recorded for that agent.
    """

    def __init__(
        self,
        backup_llm: ModuleLLM,
        max_retries: int = 3,
        fallback_action: Callable[[LLMAgent], Any] | None = None,
    ) -> None:
        self.backup_llm = backup_llm
        self.max_retries = max_retries
        self.fallback_action = fallback_action
        # Tracks the cumulative number of primary-brain failures per agent ID.
        self.failure_counts: dict[int, int] = {}

    def handle_failure(
        self,
        agent: LLMAgent,
        prompt: str | None,
        exception: Exception,
    ) -> Any:
        """
        Handle a failure from the primary LLM.

        Phase 1 - Retry with backup LLM: Attempt up to ``max_retries`` times
        to generate a response using the backup LLM.

        Phase 2 - Degrade: If all backup attempts also fail, permanently
        convert the agent to a rule-based agent for the remainder of the
        simulation.

        Parameters:
            agent: The agent whose primary LLM step failed.
            prompt: The prompt that was being processed when the failure
                occurred.  May be ``None`` when no specific prompt is
                available.
            exception: The exception raised by the primary LLM step.

        Returns:
            The backup LLM response on success, or the result of
            :meth:`degrade_to_deterministic` when all options are exhausted.
        """
        agent_id = agent.unique_id
        self.failure_counts[agent_id] = self.failure_counts.get(agent_id, 0) + 1

        # Phase 1: Attempt generation with the backup LLM.
        if self.failure_counts[agent_id] <= self.max_retries:
            logging.warning(
                "Agent %s primary brain failed (%s). "
                "Using backup LLM (attempt %d/%d).",
                agent_id,
                exception,
                self.failure_counts[agent_id],
                self.max_retries,
            )
            try:
                return self.backup_llm.generate(prompt)
            except Exception as backup_e:
                logging.error(
                    "Backup LLM also failed for Agent %s: %s",
                    agent_id,
                    backup_e,
                )

        # Phase 2: Exhausted retries - degrade to a deterministic Mesa agent.
        logging.error(
            "Agent %s exhausted all %d retries. Degrading to non-LLM agent.",
            agent_id,
            self.max_retries,
        )
        return self.degrade_to_deterministic(agent)

    def degrade_to_deterministic(self, agent: LLMAgent) -> Any:
        """
        Permanently convert the agent to a rule-based agent.

        Removes the agent's LLM capability so that no further network or
        inference calls are made.  If a ``fallback_action`` was provided it is
        executed and its result returned; otherwise ``{"action": "idle"}`` is
        returned as a safe baseline.

        Parameters:
            agent: The agent to degrade.

        Returns:
            The result of ``fallback_action(agent)`` if one was provided,
            otherwise ``{"action": "idle"}``.
        """
        # Nullify the LLM to prevent further inference calls.
        agent.llm = None

        if self.fallback_action is not None:
            return self.fallback_action(agent)

        return {"action": "idle"}
