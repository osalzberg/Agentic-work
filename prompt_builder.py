"""Prompt Builder for AKS / Container Log Analytics

Implements the layered prompt strategy described in containers_capsule/prompt_guidelines.md (relocated from docs/containers_capsule/).

Layers:
  L0 System Core                -> prompts/system_base.txt (fallback inline default)
  L1 Domain Capsule             -> prompts/domain_capsule_containerlogs.txt (optional)
    L2 Function Index (names)     -> parsed from containers_capsule/kql_functions_containerlogs.kql (new location)
  L3 Dynamic Retrieval Addendum -> lightweight keyword heuristic snippets
  L4 Clarified User Query       -> minimal normalization
  L5 Output Directive           -> decides explanation vs KQL-only
  L6 Assembly / Version Tag     -> final formatted string

Usage:
  from prompt_builder import build_prompt
  full_prompt, meta = build_prompt("why so many errors in payments service last 2h?", {})

The function returns (prompt_text, metadata_dict) so callers can log metadata (hashes, version, token estimates).

NOTE: This is intentionally dependency-light and avoids external NLP libs; you can plug in a richer intent classifier later.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from typing import Dict, List, Optional, Tuple

PROMPT_SCHEMA_VERSION = 2
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# System prompt - read from environment variable with fallback default
_DEFAULT_SYSTEM_PROMPT = "Translate the user prompt to KQL (Kusto Query Language) to run on an Azure Log Analytics workspace. Return ONLY the KQL query code with no explanations, comments, or additional text. NEVER use the old Application Insights schema, only tables listed in this page: https://learn.microsoft.com/en-us/azure/azure-monitor/reference/tables-index#application-insights"
SYSTEM_PROMPT = os.environ.get("KQL_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

# ------------------------- File Loading Helpers ------------------------- #

def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _fallback_system_prompt() -> str:
    return textwrap.dedent(
        f"""# PromptSchemaVersion:{PROMPT_SCHEMA_VERSION}
ROLE: Azure Monitor KQL Assistant

Mission:
Translate natural language into accurate, efficient KQL for Azure Monitor logs.

Non-Negotiable Rules:
1. Never fabricate table or column names. If unknown, state limitation.
2. Default timeframe: last 1 hour when absent (announce assumption).
3. Optimize cost: time filter early; project only needed columns.
4. Provide counts + rates for comparisons when relevant.
5. Mask potential secrets (Bearer tokens, keys, PEM blocks) -> replace with [REDACTED].
6. Return ONLY KQL unless user explicitly asks for explanation.
7. Use appropriate table names based on the data source (App*, Container*, Kube*, etc.).
8. For Application Insights: use App* tables (AppRequests, AppExceptions, AppTraces, AppDependencies, etc.).
9. For errors: check relevant error/severity fields in the table.
10. If the user intent is ambiguous, prefer the most common interpretation and state assumption.

Output Mode:
- KQL-Only unless explicitly asked for explanation.
"""
    ).strip()


# ------------------------- Function Index Extraction ------------------------- #
FUNC_PATTERN = re.compile(r"^let\s+([A-Za-z0-9_]+)\s*=\s*\(")
# We'll implement more resilient extraction without brittle regex.

def extract_function_index(kql_text: str) -> List[str]:
    """Extract top-level 'let Name = (params){' patterns from the KQL helper file.
    We only keep the signature (name + parameter list) to minimize token usage.
    """
    lines = kql_text.splitlines()
    signatures: List[str] = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("let ") and "={" not in line_stripped:
            # Pattern: let FunctionName = (params){
            match = re.match(r"let\s+([A-Za-z0-9_]+)\s*=\s*\(([^)]*)\)\{", line_stripped)
            if match:
                name, params = match.groups()
                signatures.append(f"{name}({params.strip()})")
    return signatures


# ------------------------- Keyword Heuristics --------------------------- #
KEYWORD_CONTEXT_MAP = {
    "error": "Errors classified via LogLevel CRITICAL/ERROR or stderr stream.",
    "latency": "Latency extracted with regex latency[=:]([0-9]+)ms into LatencyMs.",
    "slow": "Latency extracted with regex latency[=:]([0-9]+)ms into LatencyMs.",
    "crash": "Crash loops require join with KubePodInventory restart counts.",
    "restart": "Crash loops require join with KubePodInventory restart counts.",
    "stack": "Stack traces detected if message contains Exception, Traceback, or ' at '.",
    "trace": "Stack traces detected if message contains Exception, Traceback, or ' at '.",
    "status": "If LogMessage dynamic has field 'status', filter with tostring(LogMessage.status).",
    "500": "Structured status filter example: where tostring(LogMessage.status)=='500'",
    "noisy": "Noisy container detection = count lines per container or workload.",
    "volume": "High volume logs -> aggregate by WorkloadName then count().",
}


def derive_context_addendum(user_query: str) -> str:
    q_lower = user_query.lower()
    hits = []
    for kw, snippet in KEYWORD_CONTEXT_MAP.items():
        if kw in q_lower:
            hits.append(snippet)
    hits = list(dict.fromkeys(hits))  # de-duplicate preserving order
    if not hits:
        return ""
    return "\n".join(f"- {h}" for h in hits)


# ------------------------- Clarification & Output Mode ------------------ #

def clarify_query(user_query: str) -> str:
    """Light normalization: trim, collapse whitespace. (Avoid semantic rewrite here)."""
    normalized = re.sub(r"\s+", " ", user_query.strip())
    return normalized


def decide_output_mode(clarified_query: str) -> Tuple[str, str]:
    """Return (mode, directive_text)."""
    ql = clarified_query.lower()
    needs_explanation = any(w in ql for w in ["why", "explain", "describe", "root cause", "reason"])  # simple heuristic
    if needs_explanation:
        return (
            "explanation+sql",
            "Output Mode: Provide a concise explanation paragraph first, then a KQL block."
        )
    return ("kql-only", "Output Mode: Return only the KQL query (no prose).")


# ------------------------- Secret Mask (future hook) -------------------- #
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS-style as generic example
    re.compile(r"Bearer\s+[A-Za-z0-9-_\.]+"),
    re.compile(r"-----BEGIN [A-Z ]+-----"),
]


def mask_secrets(text: str) -> str:
    masked = text
    for pat in SECRET_PATTERNS:
        masked = pat.sub("[REDACTED]", masked)
    return masked


# ------------------------- Hash Utility --------------------------------- #

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------- Assembly Dataclasses ------------------------- #
@dataclass
class PromptMetadata:
    schema_version: int
    system_hash: str
    capsule_included: bool
    function_index_hash: str
    retrieval_keywords: List[str]
    output_mode: str
    timestamp_utc: str


# ------------------------- Main Builder -------------------------------- #

def build_prompt(
    user_query: str,
    intent_meta: Optional[Dict] = None,
    *,
    include_capsule: bool = True,
    force_kql_only: bool = False,
) -> Tuple[str, Dict]:
    intent_meta = intent_meta or {}

    # Use the system prompt constant
    system_text = SYSTEM_PROMPT
    
    # Use the user query directly without modification
    full_prompt = system_text

    meta = PromptMetadata(
        schema_version=PROMPT_SCHEMA_VERSION,
        system_hash=stable_hash(system_text),
        capsule_included=False,
        function_index_hash="",
        retrieval_keywords=[],
        output_mode="kql-only",
        timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return full_prompt, asdict(meta)


# ------------------------- Demo / CLI ----------------------------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build an AKS Container Logs prompt")
    parser.add_argument("query", help="User natural language query")
    parser.add_argument("--no-capsule", action="store_true", help="Exclude domain capsule layer")
    args = parser.parse_args()

    prompt, metadata = build_prompt(args.query, include_capsule=not args.no_capsule)
    print("=== PROMPT BEGIN ===")
    print(prompt)
    print("=== PROMPT END ===\n")
    print("Metadata:\n" + json.dumps(metadata, indent=2))
