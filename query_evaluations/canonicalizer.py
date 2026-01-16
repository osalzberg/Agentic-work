from __future__ import annotations

import re
from typing import Dict

# Simple KQL canonicalizer. This is intentionally lightweight and deterministic.
# Future: expand to AST-level normalization.

KEYWORDS = [
    "where",
    "project",
    "summarize",
    "extend",
    "join",
    "order by",
    "take",
    "top",
    "union",
    "let",
    "parse",
    "parse_json",
    "make-series",
    "render",
    "bin",
    "by",
]

_keyword_re = re.compile(
    r"\\b(" + "|".join(re.escape(k) for k in KEYWORDS) + r")\\b", re.IGNORECASE
)


def canonicalize_kql(kql: str) -> str:
    # Normalize line endings and whitespace
    s = kql.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s).strip()

    # Lowercase keywords only, preserve identifiers/case elsewhere
    def _lower_kw(m):
        return m.group(0).lower()

    s = _keyword_re.sub(_lower_kw, s)
    # Normalize spacing around commas and pipes
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)
    # Normalize datetime literals format (basic)
    s = re.sub(
        r"datetime\(([^)]+)\)",
        lambda m: f"datetime({m.group(1).strip()})",
        s,
        flags=re.IGNORECASE,
    )
    return s


def diff_summary(expected: str, generated: str) -> Dict[str, int]:
    return {
        "len_expected": len(expected),
        "len_generated": len(generated),
        "levenshtein_approx": _levenshtein_approx(expected, generated),
    }


def _levenshtein_approx(a: str, b: str, max_len: int = 20000) -> int:
    a = a[:max_len]
    b = b[:max_len]
    # Very small DP to avoid dependency
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]
