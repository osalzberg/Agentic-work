import asyncio
from typing import Any, Optional


def generate_kql(agent: Any, question: str, model: Optional[str] = None, system_prompt: Optional[str] = None):
    """Generate KQL using the agent with optional overrides.

    This helper mirrors the behavior previously implemented inline in
    `/api/generate-kql`: it sets model/system overrides if available,
    invokes `agent.process_natural_language(question)` and returns the
    raw result (string or structured value). It will execute coroutines
    synchronously when necessary.
    """
    # Simple generator: call agent and synchronously run coroutines if returned.
    result = agent.process_natural_language(question)
    try:
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
    except Exception:
        try:
            result = str(result)
        except Exception:
            result = None
    return result
