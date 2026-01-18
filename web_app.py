#!/usr/bin/env python3
"""
Web Interface for Natural Language KQL Agent
A Flask web application that provides a user-friendly interface for the KQL agent
"""

import asyncio
import io
import os
import re
import sys
import threading
import time  # needed for docs enrichment timing budget
import traceback
from datetime import datetime, timedelta, timezone
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from flask import request  # type: ignore
from flask import (Flask, jsonify, render_template, send_file,
                   send_from_directory)
from werkzeug.utils import secure_filename  # type: ignore

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pandas and openpyxl for batch testing
try:
    import openpyxl  # type: ignore
    import pandas as pd  # type: ignore

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    openpyxl = None

# Import the KQL agent
from logs_agent import KQLAgent

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.monitor.query import LogsQueryClient  # type: ignore
except Exception:  # Library might not be installed yet; schema fetch will be skipped
    DefaultAzureCredential = None  # type: ignore
    LogsQueryClient = None  # type: ignore
from examples_loader import load_capsule_csv_queries  # CSV capsule queries
# from example_catalog import load_example_catalog  # Disabled
from schema_manager import get_workspace_schema
from utils.batch_runner import run_batch
from utils.report_builder import build_excel, build_json

app = Flask(__name__)
# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Ensure workspace_id is defined before any early cache load attempts to avoid NameError
workspace_id = None

# Helper function to extract user token from Azure AD authentication headers
def get_user_token():
    """Extract access token from Azure AD Easy Auth headers.
    
    When Azure App Service Easy Auth is enabled, it adds headers with user info:
    - X-MS-TOKEN-AAD-ACCESS-TOKEN: The user's access token for Azure AD resources
    
    Returns:
        str or None: The access token if available, None otherwise
    """
    # Azure App Service Easy Auth provides the token in this header
    token = request.headers.get('X-MS-TOKEN-AAD-ACCESS-TOKEN')
    if token:
        print(f"[Auth] ✅ Using authenticated user's token (length: {len(token)})")
    else:
        print(f"[Auth] ⚠️ No user token found in headers. Available headers: {list(request.headers.keys())}")
        print(f"[Auth] ⚠️ Azure AD auth might not be configured or user not logged in")
    return token

# Legacy compatibility globals (removed caching logic, retained empty structures for tests/UI expecting them)
_workspace_schema_cache = {}
_workspace_schema_refresh_flags = {}
_workspace_schema_refresh_errors = {}

# Workspace schema cache removed; all schema requests now perform a fresh fetch via get_workspace_schema().

# Optional simple persistence for workspace schema across debug reloads.
SCHEMA_CACHE_FILE = None  # caching disabled


def _persist_workspace_cache():
    return  # no-op (caching disabled)


def _load_workspace_cache():
    return  # no-op (caching disabled)


# Attempt load at module import (only effective on debug reloads)
_load_workspace_cache()

# Docs enrichment tuning (env configurable to avoid UI stalls from slow/blocked Microsoft Docs fetches)
DOCS_ENRICH_DISABLE = bool(
    os.environ.get("DOCS_ENRICH_DISABLE")
)  # "1" or any non-empty string disables
WORKSPACE_SCHEMA_SYNC_FETCH = (
    os.environ.get("WORKSPACE_SCHEMA_SYNC_FETCH", "0") == "1"
)  # allow reverting to old synchronous behavior
DOCS_META_MAX_SECONDS = float(
    os.environ.get("DOCS_META_MAX_SECONDS", "10")
)  # cumulative time budget for metadata enrichment (raised from 4s to 10s)
# Removed legacy refresh flags and lock (stateless schema fetch)
DOCS_ENRICH_MAX_TABLES = int(os.environ.get("DOCS_ENRICH_MAX_TABLES", "8"))  # cap number of unmatched tables to enrich per request
DOCS_ENRICH_MAX_SECONDS = float(os.environ.get("DOCS_ENRICH_MAX_SECONDS", "5"))  # cumulative time budget per request
DOCS_ENRICH_COLUMN_FETCH = bool(os.environ.get("DOCS_ENRICH_COLUMN_FETCH", "1"))  # allow disabling column scraping (heavier)
# Fix: Properly parse "1" as True
DISABLE_SCHEMA_FETCH = os.environ.get("DISABLE_SCHEMA_FETCH", "0") in ("1", "true", "True", "TRUE", "yes")
print(f"[Config] DISABLE_SCHEMA_FETCH={DISABLE_SCHEMA_FETCH} (raw value: {os.environ.get('DISABLE_SCHEMA_FETCH', 'not set')})")


# Generic examples fallback
GENERIC_EXAMPLES = {
    "Application Insights": [
        "Show me failed requests from the last hour",
        "Show me recent exceptions",
        "Show me recent trace logs",
        "Show me dependency failures",
        "Show me page views from the last hour",
        "Show me performance counters",
    ],
    "Usage Analytics": [
        "Show me user activity patterns",
        "Get daily active users",
        "Show me usage statistics by region",
        "Show me usage trends over time",
    ],
}


# New endpoint: Suggest example queries based on resource type (dynamic mapping)
@app.route("/api/resource-examples", methods=["POST"])
def resource_examples():
    """Suggest example queries for a given resource type.

    Strategy:
      1. Attempt to locate an *_kql_examples.md file matching the resource_type inside NGSchema.
      2. Fallback to app_insights_capsule/kql_examples directory.
      3. If none found, return generic examples if available.
    """
    try:
        import glob

        data = request.get_json(silent=True) or {}
        resource_type = data.get("resource_type", "").strip()
        if not resource_type:
            return jsonify({"success": False, "error": "Resource type is required"})

        def _normalize(s: str) -> str:
            return s.replace(" ", "").lower()

        example_file = None
        ngschema_dir = os.path.join(os.path.dirname(__file__), "NGSchema")
        if os.path.exists(ngschema_dir):
            for root, dirs, _ in os.walk(ngschema_dir):
                for d in dirs:
                    if _normalize(d) == _normalize(resource_type):
                        kql_files = glob.glob(
                            os.path.join(root, d, "*_kql_examples.md")
                        )
                        if kql_files:
                            example_file = kql_files[0]
                            break
                if example_file:
                    break
        if not example_file:
            capsule_dir = os.path.join(
                os.path.dirname(__file__), "app_insights_capsule", "kql_examples"
            )
            if os.path.exists(capsule_dir):
                kql_files = glob.glob(os.path.join(capsule_dir, "*_kql_examples.md"))
                for f in kql_files:
                    base = os.path.basename(f).lower()
                    if _normalize(resource_type) in base:
                        example_file = f
                        break

        examples = []
        if example_file:
            try:
                with open(example_file, "r", encoding="utf-8") as ef:
                    content = ef.read()
                # Simple heuristic: lines starting with "#" are titles, code blocks are fenced ``` lines
                current_title = None
                current_code = []
                for line in content.splitlines():
                    if line.startswith("#"):
                        if current_title and current_code:
                            examples.append(
                                {
                                    "title": current_title.strip("# ").strip(),
                                    "query": "\n".join(current_code).strip(),
                                }
                            )
                        current_title = line
                        current_code = []
                    elif line.startswith("```"):
                        # Toggle collection; naive approach: start capturing after opening fence until closing
                        if current_code and current_code[-1] == "__END_FENCE__":
                            current_code.pop()  # remove marker
                        else:
                            current_code.append("__END_FENCE__")
                    else:
                        if current_title:
                            current_code.append(line)
                if current_title and current_code:
                    if current_code and current_code[-1] == "__END_FENCE__":
                        current_code.pop()
                    examples.append(
                        {
                            "title": current_title.strip("# ").strip(),
                            "query": "\n".join(current_code).strip(),
                        }
                    )
            except Exception as ex:  # noqa: BLE001
                print(f"[Examples] Failed parsing examples file {example_file}: {ex}")

        # Fallback generic examples
        if not examples:
            generic = (
                GENERIC_EXAMPLES.get(resource_type)
                or GENERIC_EXAMPLES.get("Application Insights")
                or []
            )
            examples = [
                {"title": f"Example {i+1}", "query": q} for i, q in enumerate(generic)
            ]

        return jsonify(
            {
                "success": True,
                "resource_type": resource_type,
                "example_file": example_file,
                "examples": examples,
            }
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/fetch-workspace-schema", methods=["GET", "POST"])
def fetch_workspace_schema():
    """Direct, stateless workspace schema retrieval (lightweight summary).

    Returns JSON:
      success, workspace_id, table_count, retrieved_at, source, error (optional)
    """
    global workspace_id
    if not workspace_id:
        return (
            jsonify(
                {
                    "success": False,
                    "workspace_id": None,
                    "table_count": 0,
                    "error": "Workspace not initialized",
                }
            ),
            200,
        )
    if DISABLE_SCHEMA_FETCH:
        return (
            jsonify(
                {
                    "success": True,
                    "workspace_id": workspace_id,
                    "table_count": 0,
                    "retrieved_at": None,
                    "source": "disabled",
                    "disabled": True,
                }
            ),
            200,
        )
    try:
        result = get_workspace_schema(workspace_id)
        if result.get("error"):
            return (
                jsonify(
                    {
                        "success": False,
                        "workspace_id": workspace_id,
                        "table_count": 0,
                        "error": result.get("error"),
                    }
                ),
                200,
            )
        return jsonify(
            {
                "success": True,
                "workspace_id": workspace_id,
                "table_count": len(result.get("tables", [])),
                "retrieved_at": result.get("retrieved_at"),
                "source": result.get("source"),
            }
        )
    except Exception as e:  # noqa: BLE001
        return (
            jsonify(
                {
                    "success": False,
                    "workspace_id": workspace_id,
                    "table_count": 0,
                    "error": str(e),
                }
            ),
            200,
        )


# Global agent instance
agent = None
_workspace_resource_types_cache = (
    {}
)  # deprecated: manifest data now supplied via SchemaManager
_workspace_queries_cache = {}
_ms_docs_table_resource_type_cache = (
    {}
)  # table_name -> resource_type | 'unknown resource type'
_ms_docs_table_full_cache = (
    {}
)  # table_name -> { description, columns:[{name,type,description}], fetched_at }
_ms_docs_table_queries_cache = {}  # table_name -> [ { name, description } ]

# Shared Azure credential (created once to avoid multiple az login prompts)
_azure_credential = None
_credential_creation_lock = threading.Lock()

# Static fallback map for common Application Insights (Azure Monitor 'classic' AI) derived tables
_STATIC_FALLBACK_TABLE_RESOURCE_TYPES = {
    # App Insights standard tables
    "AppRequests": "microsoft.insights/components",
    "AppDependencies": "microsoft.insights/components",
    "AppTraces": "microsoft.insights/components",
    "AppExceptions": "microsoft.insights/components",
    "AppAvailabilityResults": "microsoft.insights/components",
    "AppPageViews": "microsoft.insights/components",
    "AppPerformanceCounters": "microsoft.insights/components",
    "AppBrowserTimings": "microsoft.insights/components",
    "AppCustomEvents": "microsoft.insights/components",
    "AppCustomMetrics": "microsoft.insights/components",
    "AppMetric": "microsoft.insights/components",
    "AppMetrics": "microsoft.insights/components",
    "AppSessions": "microsoft.insights/components",
    "AppEvents": "microsoft.insights/components",
    "AppPageViewPerformance": "microsoft.insights/components",
}


def _lookup_table_resource_type_doc(table_name: str, timeout: float = 4.0) -> str:
    """Best-effort lookup of resource type for a given table via public Microsoft Docs.

    Tries https://learn.microsoft.com/en-us/azure/azure-monitor/reference/tables/{table}
    Table pages typically include a "Resource type:" label followed by a provider/resourceType value.
    Returns discovered resource type string or 'unknown resource type'.
    Caches results per process. Keeps failures cached to avoid repeated outbound calls.
    """
    if DOCS_ENRICH_DISABLE:
        return "unknown resource type"
    if not table_name:
        return "unknown resource type"
    cache_key = table_name
    # Static fallback first (case-insensitive)
    for k, v in _STATIC_FALLBACK_TABLE_RESOURCE_TYPES.items():
        if k.lower() == table_name.lower():
            return v
    if cache_key in _ms_docs_table_resource_type_cache:
        return _ms_docs_table_resource_type_cache[cache_key]
    slug = table_name.lower()
    url = (
        f"https://learn.microsoft.com/en-us/azure/azure-monitor/reference/tables/{slug}"
    )
    try:
        req = urlrequest.Request(
            url,
            headers={"User-Agent": "AzMonLogsAgent/1.0 (+https://github.com/noakup)"},
        )
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                _ms_docs_table_resource_type_cache[cache_key] = "unknown resource type"
                return "unknown resource type"
            content = resp.read().decode("utf-8", errors="ignore")
            # Heuristic: look for 'Resource type' line then capture next provider/resourceType token
            # Common patterns: <strong>Resource type:</strong> Microsoft.Insights/components
            # or visible text 'Resource type: microsoft.operationalinsights/workspaces'
            m = re.search(
                r"Resource\s*type:?\s*</?strong>[^A-Za-z0-9/]*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
                content,
                re.IGNORECASE,
            )
            if not m:
                # Fallback: search anywhere for microsoft.<provider>/<resourcetype> preceded by 'Resource type'
                m = re.search(
                    r"Resource\s*type:?[^\n]{0,120}?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
                    content,
                    re.IGNORECASE,
                )
            if m:
                rtype = m.group(1)
                # Normalize provider casing (Microsoft.*) if missing capital M
                if rtype.lower().startswith("microsoft.") and not rtype.startswith(
                    "Microsoft."
                ):
                    rtype = "Microsoft." + rtype.split(".", 1)[1]
                _ms_docs_table_resource_type_cache[cache_key] = rtype
                return rtype
    except (
        HTTPError,
        URLError,
        TimeoutError,
        ValueError,
    ) as e:  # noqa: PERF203 - broad acceptable for network
        print(f"[Docs Enrichment] Failed to fetch {url}: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[Docs Enrichment] Unexpected error for {url}: {e}")
    _ms_docs_table_resource_type_cache[cache_key] = "unknown resource type"
    return "unknown resource type"


def _fetch_table_docs_full(table_name: str, timeout: float = 6.0) -> dict:
    """Fetch table description and columns from Microsoft Docs table reference page.

    Caches results in _ms_docs_table_full_cache. Returns a dict:
      { description:str, columns:[{name,type,description}], fetched_at:str }
    """
    if DOCS_ENRICH_DISABLE:
        return {}
    if not table_name:
        return {}
    if table_name in _ms_docs_table_full_cache:
        return _ms_docs_table_full_cache[table_name]
    slug = table_name.lower()
    url = f"https://learn.microsoft.com/azure/azure-monitor/reference/tables/{slug}"
    try:
        req = urlrequest.Request(
            url,
            headers={"User-Agent": "AzMonLogsAgent/1.0 (+https://github.com/noakup)"},
        )
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {}
            html = resp.read().decode("utf-8", errors="ignore")
        # Very lightweight parsing heuristics (avoid full HTML parser to keep deps minimal)
        # Description: first <p> after <h1> or meta description
        desc_match = re.search(
            r"<h1[^>]*>.*?</h1>\s*<p>(.*?)</p>", html, re.IGNORECASE | re.DOTALL
        )
        description = ""
        if desc_match:
            description = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()
        # Columns: look for markdown-like table rendered as HTML <table> with column headers Name, Type
        columns = []
        table_sections = re.findall(
            r"<table.*?>.*?</table>", html, re.IGNORECASE | re.DOTALL
        )
        if DOCS_ENRICH_COLUMN_FETCH:
            for sect in table_sections:
                if re.search(
                    r"<th[^>]*>\s*Name\s*</th>", sect, re.IGNORECASE
                ) and re.search(r"<th[^>]*>\s*Type\s*</th>", sect, re.IGNORECASE):
                    # Extract rows
                    rows = re.findall(
                        r"<tr>(.*?)</tr>", sect, re.IGNORECASE | re.DOTALL
                    )
                    for r in rows[1:]:  # skip header
                        cols = re.findall(
                            r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.IGNORECASE | re.DOTALL
                        )
                        if len(cols) >= 2:
                            cname = re.sub(r"<[^>]+>", "", cols[0]).strip()
                            ctype = re.sub(r"<[^>]+>", "", cols[1]).strip()
                            cdesc = ""
                            if len(cols) >= 3:
                                cdesc = re.sub(r"<[^>]+>", "", cols[2]).strip()
                            if cname:
                                columns.append(
                                    {"name": cname, "type": ctype, "description": cdesc}
                                )
                    if columns:
                        break
        record = {
            "description": description,
            "columns": columns,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _ms_docs_table_full_cache[table_name] = record
        return record
    except Exception as e:  # noqa: BLE001
        print(f"[Docs Enrichment] Failed full table fetch for {table_name}: {e}")
        return {}


def _fetch_table_docs_queries(table_name: str, timeout: float = 6.0) -> list:
    """Fetch example queries for a table from Microsoft Docs queries page.

    Heuristic extraction pattern:
      <h3>Query Title</h3> (sometimes h2)
      <p>Short description sentence.</p>
      <pre><code class="lang-kusto">KQL HERE</code></pre>

    Returns list of { name, description, code, source='docs' }.
    Caches results per table.
    """
    if DOCS_ENRICH_DISABLE:
        return []
    if not table_name:
        return []
    if table_name in _ms_docs_table_queries_cache:
        return _ms_docs_table_queries_cache[table_name]
    slug = table_name.lower()
    url = f"https://learn.microsoft.com/azure/azure-monitor/reference/queries/{slug}"
    try:
        req = urlrequest.Request(
            url,
            headers={"User-Agent": "AzMonLogsAgent/1.0 (+https://github.com/noakup)"},
        )
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return []
            html_text = resp.read().decode("utf-8", errors="ignore")
        import html as html_lib  # local import to avoid global namespace clutter

        queries = []
        # Find all h2/h3 headings as potential query titles
        heading_iter = list(
            re.finditer(
                r"<h[23][^>]*>(.*?)</h[23]>", html_text, re.IGNORECASE | re.DOTALL
            )
        )
        for idx, match in enumerate(heading_iter):
            raw_title = match.group(1)
            title_clean = re.sub(r"<[^>]+>", "", raw_title).strip()
            if not title_clean:
                continue
            # Slice segment until next heading or limited length
            start = match.end()
            end = (
                heading_iter[idx + 1].start()
                if idx + 1 < len(heading_iter)
                else len(html_text)
            )
            segment = html_text[start:end]
            # Description: first <p>...</p>
            desc_match = re.search(r"<p>(.*?)</p>", segment, re.IGNORECASE | re.DOTALL)
            desc_clean = ""
            if desc_match:
                desc_clean = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()
            # Code: prefer <pre><code ...>...</code></pre>
            code_match = re.search(
                r"<pre[^>]*>\s*<code[^>]*>([\s\S]*?)</code>\s*</pre>",
                segment,
                re.IGNORECASE,
            )
            if not code_match:
                # fallback single code tag
                code_match = re.search(
                    r"<code[^>]*>([\s\S]*?)</code>", segment, re.IGNORECASE
                )
            code_text = ""
            if code_match:
                code_text = code_match.group(1)
                # Remove HTML tags inside code (rare) and unescape entities
                code_text = re.sub(r"<[^>]+>", "", code_text)
                code_text = html_lib.unescape(code_text).strip()
            # Only record if we have at least a code block (to ensure it's an actual query example)
            if code_text:
                queries.append(
                    {
                        "name": title_clean,
                        "description": desc_clean,
                        "code": code_text,
                        "source": "docs",
                    }
                )
        _ms_docs_table_queries_cache[table_name] = queries
        return queries
    except Exception as e:  # noqa: BLE001
        print(f"[Docs Enrichment] Failed table queries fetch for {table_name}: {e}")
        return []

    # Legacy cache-based workspace schema block removed.

    for entry in workspace_tables:
        tname = entry.get("name")
        ws_rt = entry.get("resource_type") or "Unknown"
        manifest_rts = manifest_tables_index.get(tname, [])
        in_manifest = bool(manifest_rts)
        if in_manifest:
            matched += 1
            # Build subset mapping: include table under each manifest resource type it appears in
            for mrt in manifest_rts:
                resource_type_subset.setdefault(mrt, []).append(tname)
                provider = mrt.split("/")[0]
                provider_summary.setdefault(
                    provider, {"tables": 0, "resource_types": set()}
                )
                provider_summary[provider]["tables"] += 1
                provider_summary[provider]["resource_types"].add(mrt)
        else:
            unmatched.append(tname)
        enriched.append(
            {
                "name": tname,
                "workspace_resource_type": ws_rt,
                "in_manifest": in_manifest,
                "manifest_resource_type": manifest_rts[0] if manifest_rts else None,
                "manifest_resource_types": manifest_rts,
                "provider": (manifest_rts[0].split("/")[0] if manifest_rts else None),
            }
        )

    # NOTE: Synthetic fallback grouping for unmatched tables removed per user request.
    # Unmatched tables will no longer be assigned to a synthetic resource type; they remain
    # visible only in the 'unmatched_tables' list. This keeps the UI honest about mapping
    # completeness and allows the frontend to surface an explicit error state when
    # workspace tables exist but no resource types are mapped.

    # Enrich unmatched tables using Microsoft Docs table pages (best-effort)
    doc_enriched = 0
    # if unmatched and not DOCS_ENRICH_DISABLE:
    # # Bound number of tables & cumulative time to keep endpoint responsive
    # t_docs_start = time.time()
    # for table in unmatched[:DOCS_ENRICH_MAX_TABLES]:
    #     doc_rtype = _lookup_table_resource_type_doc(table)
    #     # Normalize to lowercase provider/resource type to be consistent (as per docs examples)
    #     if doc_rtype != 'unknown resource type':
    #         doc_rtype = doc_rtype.strip()
    #         # Ensure pattern provider/resourceType; if uppercase Microsoft.* keep as-is else lowercase
    #         if '/' in doc_rtype:
    #             parts = doc_rtype.split('/')
    #             if len(parts) == 2:
    #                 # Keep original provider case if startswith Microsoft., else lowercase both
    #                 if not parts[0].startswith('Microsoft.'):
    #                     doc_rtype = f"{parts[0].lower()}/{parts[1].lower()}"
    #         else:
    #             # Not a recognized pattern, mark unknown
    #             doc_rtype = 'unknown resource type'
    #     # Update enriched list entry
    #     for e in enriched:
    #         if e['name'] == table:
    #             e['doc_resource_type'] = doc_rtype
    #             if doc_rtype != 'unknown resource type' and not e.get('provider'):
    #                 e['provider'] = doc_rtype.split('/')[0]
    #             break
    #     # If a resource type was found (not unknown), include in subset; otherwise bucket under 'unknown resource type'
    #     target_rtype = doc_rtype if doc_rtype != 'unknown resource type' else 'unknown resource type'
    #     if target_rtype not in resource_type_subset:
    #         resource_type_subset[target_rtype] = []
    #     resource_type_subset[target_rtype].append(table)
    #     if target_rtype != 'unknown resource type':
    #         provider = target_rtype.split('/')[0]
    #     else:
    #         provider = 'unknown'
    #     provider_summary.setdefault(provider, {'tables': 0, 'resource_types': set()})
    #     provider_summary[provider]['tables'] += 1
    #     if target_rtype != 'unknown resource type':
    #         provider_summary[provider]['resource_types'].add(target_rtype)
    #     doc_enriched += 1
    #     if (time.time() - t_docs_start) > DOCS_ENRICH_MAX_SECONDS:
    #         print(f"[Docs Enrichment] Cumulative time budget exceeded ({DOCS_ENRICH_MAX_SECONDS}s); skipping remaining tables.")
    #         break

    # Normalize provider summary sets to counts (after enrichment)
    provider_summary_out = {
        prov: {
            "tables": info["tables"],
            "resource_types": (
                len(info["resource_types"])
                if isinstance(info["resource_types"], set)
                else info["resource_types"]
            ),
        }
        for prov, info in provider_summary.items()
    }

    # Build per-table queries mapping (manifest first, docs fallback)
    manifest_table_queries = manifest_data.get("table_queries", {}) or {}
    capsule_csv_queries = {}
    try:
        capsule_csv_queries = load_capsule_csv_queries()
        # Debug summary of capsule CSV ingestion
        total_capsule_query_count = sum(len(v) for v in capsule_csv_queries.values())
        print(
            f"[Capsule CSV] tables={list(capsule_csv_queries.keys())} total_queries={total_capsule_query_count}"
        )
        for tbl, qlist in capsule_csv_queries.items():
            preview = [q.get("name", "<noname>")[:60] for q in qlist[:5]]
            print(f"[Capsule CSV] table={tbl} count={len(qlist)} preview={preview}")
    except Exception as e:  # noqa: BLE001
        print(f"[Workspace Schema] Capsule CSV ingestion error: {e}")

    def _normalize_code(code: str) -> str:
        if not isinstance(code, str):
            return ""
        c = code.replace("\r\n", "\n").replace("\r", "\n").strip()
        while "\n\n" in c:
            c = c.replace("\n\n", "\n")
        return c

    workspace_table_queries = {}
    tables_with_manifest_queries = 0
    tables_with_capsule_queries = 0
    tables_with_docs_queries = 0
    for tinfo in enriched:
        tname = tinfo["name"]
        combined: list[dict] = []
        # 1. Manifest queries (highest precedence)
        m_list = manifest_table_queries.get(tname) or []
        if m_list:
            for q in m_list:
                # Ensure uniform shape; manifest queries already include name/description
                if q.get("name"):
                    combined.append(
                        {
                            "name": q.get("name"),
                            "description": q.get("description"),
                            "code": q.get("code") if q.get("code") else None,
                            "source": "manifest",
                        }
                    )
            tables_with_manifest_queries += 1
        # 2. Capsule CSV queries (second precedence)
        c_list = capsule_csv_queries.get(tname) or []
        if c_list:
            for q in c_list:
                if q.get("name") and q.get("code"):
                    norm_code = _normalize_code(q.get("code"))
                    if not any(
                        _normalize_code(existing.get("code")) == norm_code
                        for existing in combined
                        if existing.get("code")
                    ):
                        combined.append(
                            {
                                "name": q.get("name"),
                                "description": q.get("description") or "",
                                "code": q.get("code"),
                                "source": q.get("source") or "capsule-csv",
                                "file": q.get("file"),
                            }
                        )
            tables_with_capsule_queries += 1
        # 3. Docs queries (fallback only adds if new code)
        if not DOCS_ENRICH_DISABLE:
            docs_q = _fetch_table_docs_queries(tname)
            if docs_q:
                added_docs = 0
                for q in docs_q:
                    if q.get("name") and q.get("code"):
                        norm_code = _normalize_code(q.get("code"))
                        if not any(
                            _normalize_code(existing.get("code")) == norm_code
                            for existing in combined
                            if existing.get("code")
                        ):
                            combined.append(
                                {
                                    "name": q.get("name"),
                                    "description": q.get("description"),
                                    "code": q.get("code"),
                                    "source": "docs",
                                }
                            )
                            added_docs += 1
                if added_docs:
                    tables_with_docs_queries += 1
        if combined:
            workspace_table_queries[tname] = combined
            # Debug per-table merged result
            src_counts = {
                "manifest": sum(1 for q in combined if q.get("source") == "manifest"),
                "capsule-csv": sum(
                    1 for q in combined if q.get("source") == "capsule-csv"
                ),
                "docs": sum(1 for q in combined if q.get("source") == "docs"),
            }
            print(
                f"[QueryMerge] table={tname} total={len(combined)} breakdown={src_counts}"
            )
        else:
            print(
                f"[QueryMerge] table={tname} no queries merged (manifest={len(m_list)} capsule={len(capsule_csv_queries.get(tname, []))})"
            )

    # Final debug: any query objects missing required fields
    missing_fields_total = 0
    for tbl, qlist in workspace_table_queries.items():
        for q in qlist:
            if not q.get("name") or not q.get("code"):
                print(f"[QueryMerge][WARN] table={tbl} query missing name/code -> {q}")
                missing_fields_total += 1
    if missing_fields_total:
        print(f"[QueryMerge][WARN] total_incomplete_queries={missing_fields_total}")
    else:
        print("[QueryMerge] all merged queries have name & code")

    response = {
        "success": True,
        "status": "ready",
        "workspace_id": workspace_id,
        "counts": {
            "workspace_tables": len(workspace_tables),
            "manifest_tables": sum(
                len(v) for v in manifest_data.get("resource_type_tables", {}).values()
            ),
            "matched_tables": matched,
            "unmatched_tables": len(unmatched),
            "resource_types_with_data": len(resource_type_subset),
        },
        "tables": enriched,
        "resource_type_tables": resource_type_subset,
        "providers": provider_summary_out,
        "unmatched_tables": unmatched,
        "retrieved_at": ws_cache.get("retrieved_at"),
        "doc_enrichment": {
            "performed": bool(unmatched) and not DOCS_ENRICH_DISABLE,
            "enriched_tables": doc_enriched,
            "cache_size": len(_ms_docs_table_resource_type_cache),
            "disabled": DOCS_ENRICH_DISABLE,
            "limits": {
                "max_tables": DOCS_ENRICH_MAX_TABLES,
                "max_seconds": DOCS_ENRICH_MAX_SECONDS,
                "column_fetch": DOCS_ENRICH_COLUMN_FETCH,
            },
        },
        # Workspace scoped table metadata (manifest first, docs fallback)
        "table_metadata": {},
        "table_queries": workspace_table_queries,
        "query_enrichment": {
            "tables_with_manifest_queries": tables_with_manifest_queries,
            "tables_with_capsule_csv_queries": tables_with_capsule_queries,
            "tables_with_docs_queries": tables_with_docs_queries,
        },
    }
    try:
        manifest_meta = manifest_data.get("table_metadata", {}) or {}
        t_meta_start = time.time()
        for tinfo in enriched:
            if (time.time() - t_meta_start) > DOCS_META_MAX_SECONDS:
                print(
                    f"[Workspace Schema] Metadata enrichment time budget exceeded ({DOCS_META_MAX_SECONDS}s); truncating."
                )
                break
            tname = tinfo["name"]
            meta = manifest_meta.get(tname, {}).copy()
            if not DOCS_ENRICH_DISABLE and (
                not meta.get("description") or not meta.get("columns")
            ):
                docs_meta = _fetch_table_docs_full(tname)
                if docs_meta:
                    if not meta.get("description") and docs_meta.get("description"):
                        meta["description"] = docs_meta["description"]
                    if not meta.get("columns") and docs_meta.get("columns"):
                        meta["columns"] = docs_meta["columns"]
            response["table_metadata"][tname] = meta
    except Exception as e:  # noqa: BLE001
        print(f"[Workspace Schema] Metadata enrichment error: {e}")
    # Verbose schema dump removed per request to reduce console noise.
    # Persist enriched response back into cache (exclude success/status to keep shape stable for early return)
    # Caching disabled: do not persist enriched workspace schema response.
    print(
        f"[Workspace Schema] Enrichment complete (no persistence) tables={len(response.get('tables', []))} table_queries_total={sum(len(v) for v in response.get('table_queries', {}).values())}"
    )
    return jsonify(response)


@app.route("/api/workspace-schema-status", methods=["GET"])
def workspace_schema_status():
    """Stateless status endpoint (cache removed).

    Returns JSON:
      success: bool
      status: 'uninitialized' | 'ready' | 'empty' | 'disabled' | 'query-only'
      workspace_id: str|None
      table_count: int
      retrieved_at: str|None
      source: str|None
      message: str (optional, for empty/query-only status)
    """
    global workspace_id
    if not workspace_id:
        return jsonify(
            {
                "success": False,
                "status": "uninitialized",
                "workspace_id": None,
                "table_count": 0,
                "retrieved_at": None,
                "source": None,
            }
        )
    if DISABLE_SCHEMA_FETCH:
        return jsonify(
            {
                "success": True,
                "status": "disabled",
                "workspace_id": workspace_id,
                "table_count": 0,
                "retrieved_at": None,
                "source": "disabled",
                "disabled": True,
            }
        )
    result = get_workspace_schema(workspace_id)
    if result.get('error'):
        return jsonify({'success': False, 'status': 'error', 'workspace_id': workspace_id, 'error': result.get('error'), 'table_count': 0, 'retrieved_at': None, 'source': None})
    tables = result.get('tables', [])
    
    # If no tables found, return query-only status (app still works for queries)
    if not tables:
        return jsonify({
            'success': True,
            'status': 'query-only',
            'workspace_id': workspace_id,
            'table_count': 0,
            'retrieved_at': result.get('retrieved_at'),
            'source': result.get('source'),
            'message': 'Table discovery unavailable. You can still run KQL queries by typing them directly.'
        })
    
    return jsonify({
        'success': True,
        'status': 'ready',
        'workspace_id': workspace_id,
        'table_count': len(tables),
        'retrieved_at': result.get('retrieved_at'),
        'source': result.get('source')
    })

# ---------------------------------------------------------------------------
# Compatibility workspace schema endpoint (previously removed). Front-end and
# tests poll this path; we now serve immediate ready state with stateless fetch.
# ---------------------------------------------------------------------------
@app.route("/api/workspace-schema", methods=["GET"])
def workspace_schema():
    """Enriched workspace schema (stateless) with merged query suggestions.

    Returns:
      success: bool
      status: ready|empty|uninitialized|error|disabled
      workspace_id: str|None
      tables: [ { name, resource_type, manifest_resource_types? } ]
      counts: { workspace_tables, tables_with_manifest_queries, tables_with_capsule_csv_queries, tables_with_docs_queries }
      table_queries: { tableName: [ { name, description, code, source, file? } ] }
      table_metadata: { tableName: { description, columns:[{name,type,description}] } }
      doc_enrichment: { disabled: bool }
    """
    global workspace_id
    if not workspace_id:
        return (
            jsonify(
                {
                    "success": False,
                    "status": "uninitialized",
                    "workspace_id": None,
                    "tables": [],
                    "counts": {},
                    "table_queries": {},
                    "table_metadata": {},
                    "error": "Workspace not initialized",
                }
            ),
            400,
        )
    if DISABLE_SCHEMA_FETCH:
        return (
            jsonify(
                {
                    "success": True,
                    "status": "disabled",
                    "workspace_id": workspace_id,
                    "tables": [],
                    "counts": {},
                    "table_queries": {},
                    "table_metadata": {},
                    "disabled": True,
                }
            ),
            200,
        )
    try:
        # Fetch workspace tables
        ws_result = get_workspace_schema(workspace_id)
        if ws_result.get("error"):
            return (
                jsonify(
                    {
                        "success": False,
                        "status": "error",
                        "workspace_id": workspace_id,
                        "error": ws_result.get("error"),
                        "tables": [],
                        "table_queries": {},
                        "table_metadata": {},
                    }
                ),
                500,
            )
        workspace_tables = ws_result.get("tables", [])

        # Load manifest (resource types + manifest queries)
        manifest_tables_index = {}
        manifest_table_queries = {}
        manifest_table_metadata = {}
        try:
            from schema_manager import SchemaManager

            mgr = SchemaManager.get()
            mgr.load_manifest(force=False)
            manifest_cache = mgr._manifest_cache or {}
            manifest_tables_index = manifest_cache.get("resource_type_tables", {}) or {}
            # Build reverse index for resource types per table
            reverse_index = {}
            for rt, tbls in manifest_tables_index.items():
                for t in tbls:
                    reverse_index.setdefault(t, []).append(rt)
            manifest_table_queries = manifest_cache.get("table_queries", {}) or {}
            manifest_table_metadata = manifest_cache.get("table_metadata", {}) or {}
        except Exception as me:  # noqa: BLE001
            print(f"[WorkspaceSchema] Manifest load error: {me}")
            reverse_index = {}

        # Capsule CSV queries
        try:
            from examples_loader import load_capsule_csv_queries

            capsule_csv_queries = load_capsule_csv_queries()
        except Exception as ce:  # noqa: BLE001
            print(f"[WorkspaceSchema] Capsule CSV ingestion error: {ce}")
            capsule_csv_queries = {}

        # Docs queries (optional, only if not disabled)
        tables_with_docs_queries = 0

        def _normalize_code(code: str) -> str:
            if not isinstance(code, str):
                return ""
            c = code.replace("\r\n", "\n").replace("\r", "\n").strip()
            while "\n\n" in c:
                c = c.replace("\n\n", "\n")
            return c

        table_queries = {}
        tables_with_manifest_queries = 0
        tables_with_capsule_queries = 0
        for entry in workspace_tables:
            tname = entry.get("name")
            combined = []
            # Manifest queries first
            m_list = manifest_table_queries.get(tname) or []
            if m_list:
                for q in m_list:
                    if q.get("name"):
                        combined.append(
                            {
                                "name": q.get("name"),
                                "description": q.get("description"),
                                "code": q.get("code") if q.get("code") else None,
                                "source": "manifest",
                            }
                        )
                tables_with_manifest_queries += 1
            # Capsule CSV queries second
            c_list = capsule_csv_queries.get(tname) or []
            if c_list:
                for q in c_list:
                    if q.get("name") and q.get("code"):
                        norm_code = _normalize_code(q.get("code"))
                        if not any(
                            _normalize_code(existing.get("code")) == norm_code
                            for existing in combined
                            if existing.get("code")
                        ):
                            combined.append(
                                {
                                    "name": q.get("name"),
                                    "description": q.get("description") or "",
                                    "code": q.get("code"),
                                    "source": q.get("source") or "capsule-csv",
                                    "file": q.get("file"),
                                }
                            )
                tables_with_capsule_queries += 1
            # Docs queries third (only if enabled)
            if not DOCS_ENRICH_DISABLE:
                docs_q = _fetch_table_docs_queries(tname)
                if docs_q:
                    added_docs = 0
                    for q in docs_q:
                        if q.get("name") and q.get("code"):
                            norm_code = _normalize_code(q.get("code"))
                            if not any(
                                _normalize_code(existing.get("code")) == norm_code
                                for existing in combined
                                if existing.get("code")
                            ):
                                combined.append(
                                    {
                                        "name": q.get("name"),
                                        "description": q.get("description"),
                                        "code": q.get("code"),
                                        "source": "docs",
                                    }
                                )
                                added_docs += 1
                    if added_docs:
                        tables_with_docs_queries += 1
            if combined:
                table_queries[tname] = combined

        # Attach manifest resource type info to tables
        reverse_index = {}
        for rt, tbls in manifest_tables_index.items():
            for t in tbls:
                reverse_index.setdefault(t, []).append(rt)
        enriched_tables = []
        for entry in workspace_tables:
            tname = entry.get("name")
            enriched_tables.append(
                {
                    "name": tname,
                    "resource_type": entry.get("resource_type") or "Unknown",
                    "manifest_resource_types": reverse_index.get(tname, []),
                }
            )

        # Build minimal table metadata (manifest first, docs fallback if enabled)
        table_metadata_out = {}
        for tname in [e["name"] for e in enriched_tables]:
            meta = manifest_table_metadata.get(tname, {}).copy()
            if not DOCS_ENRICH_DISABLE and (
                not meta.get("description") or not meta.get("columns")
            ):
                docs_meta = _fetch_table_docs_full(tname)
                if docs_meta:
                    if not meta.get("description") and docs_meta.get("description"):
                        meta["description"] = docs_meta["description"]
                    if not meta.get("columns") and docs_meta.get("columns"):
                        meta["columns"] = docs_meta["columns"]
            table_metadata_out[tname] = meta

        status = "ready" if enriched_tables else "empty"
        return jsonify(
            {
                "success": True,
                "status": status,
                "workspace_id": workspace_id,
                "tables": enriched_tables,
                "counts": {
                    "workspace_tables": len(enriched_tables),
                    "tables_with_manifest_queries": tables_with_manifest_queries,
                    "tables_with_capsule_csv_queries": tables_with_capsule_queries,
                    "tables_with_docs_queries": tables_with_docs_queries,
                },
                "table_queries": table_queries,
                "table_metadata": table_metadata_out,
                "retrieved_at": ws_result.get("retrieved_at"),
                "source": ws_result.get("source"),
                "doc_enrichment": {"disabled": DOCS_ENRICH_DISABLE},
                "error": None,
            }
        )
    except Exception as e:  # noqa: BLE001
        return (
            jsonify(
                {
                    "success": False,
                    "status": "error",
                    "workspace_id": workspace_id,
                    "error": str(e),
                }
            ),
            500,
        )


@app.route("/api/refresh-manifest", methods=["POST"])
def refresh_manifest():
    """Force a rescan of NGSchema manifests and update persisted manifest cache.

    This endpoint is explicit and not part of normal UI flow. Returns summary counts.
    """
    try:
        from schema_manager import SchemaManager

        mgr = SchemaManager.get()
        mgr.load_manifest(force=True)
        manifest = mgr._manifest_cache
        return jsonify(
            {
                "success": True,
                "forced": True,
                "resource_type_count": len(manifest.get("resource_type_tables", {})),
                "total_table_mappings": sum(
                    len(v) for v in manifest.get("resource_type_tables", {}).values()
                ),
                "fetched_at": manifest.get("fetched_at"),
                "manifests_scanned": manifest.get("manifests_scanned"),
            }
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/resource-schema", methods=["GET"])
def resource_schema():
    """Return manifest-derived resource schema merged with any cached workspace tables.

    This endpoint restores the previous /api/resource-schema functionality (now missing) so
    UI callers expecting it no longer receive a 404. It focuses on resource types and
    manifest metadata while optionally enriching with currently cached workspace tables.

    Response JSON includes:
      success: bool
      status: 'uninitialized' | 'pending' | 'ready'
      workspace_id: str|None
      manifest: {
        resource_types: [str],
        providers: [str],
        counts: { resource_types:int, providers:int, tables:int },
        resource_type_tables: { resource_type: [tableName] },
        table_metadata: { tableName: { description:str, columns:[{name,type,descriptions:[str]}], resource_types:[str] } },
        queries: [ { name, description, resource_type, provider, table, manifest_file, related_tables:[...] } ]
      }
      workspace: {
        tables: [ { name, resource_type } ],
        retrieved_at: str|None,
        count: int,
        source: str|None
      }
      table_join: [ { name, workspace_resource_type, manifest_resource_types:[str] } ]
    """
    global workspace_id

    # Persistent manifest load via SchemaManager (decoupled from workspace fetch)
    try:
        from schema_manager import SchemaManager

        mgr = SchemaManager.get()
        mgr.load_manifest(force=False)
        manifest_data = mgr._manifest_cache
    except Exception as e:  # pragma: no cover
        print(f"[ResourceSchema] Manifest load error: {e}")
        manifest_data = {}

    # If no workspace selected yet, return manifest-only payload (no error)
    if not workspace_id:
        # Provide explicit top-level lists for frontend convenience (resource_types/providers)
        _rt_tables = manifest_data.get("resource_type_tables", {}) or {}
        _resource_types_list = sorted(_rt_tables.keys())
        _providers_list = sorted({rt.split("/")[0] for rt in _rt_tables})
        return jsonify(
            {
                "success": True,
                "status": "manifest-only",
                "workspace_id": None,
                "manifest": {
                    "resource_type_tables": _rt_tables,
                    "table_resource_types": manifest_data.get(
                        "table_resource_types", {}
                    ),
                    "resource_types": _resource_types_list,
                    "providers": _providers_list,
                    "fetched_at": manifest_data.get("fetched_at"),
                    "manifests_scanned": manifest_data.get("manifests_scanned"),
                    "counts": {
                        "resource_types": len(_resource_types_list),
                        "providers": len(_providers_list),
                        "tables": sum(len(v) for v in _rt_tables.values()),
                    },
                },
                "workspace": {
                    "tables": [],
                    "retrieved_at": None,
                    "count": 0,
                    "source": None,
                },
                "table_join": [],
            }
        )

    # Workspace tables (may be absent if not yet fetched)
    # Fresh fetch of workspace tables (cache removed)
    ws_result = get_workspace_schema(workspace_id)
    if ws_result.get("error"):
        print(f"[ResourceSchema] workspace fetch error: {ws_result.get('error')}")
        workspace_tables = []
        ws_retrieved = None
        ws_source = None
    else:
        workspace_tables = ws_result.get("tables", [])
        ws_retrieved = ws_result.get("retrieved_at")
        ws_source = ws_result.get("source")

    # workspace_tables already set above
    join = []
    manifest_tables_index = manifest_data.get("resource_type_tables", {})
    # Build reverse index table -> list(resource_types)
    reverse_index = {}
    for rt, tbls in manifest_tables_index.items():
        for t in tbls:
            reverse_index.setdefault(t, []).append(rt)
    for entry in workspace_tables:
        tname = entry.get("name")
        w_rt = entry.get("resource_type") or "Unknown"
        m_rts = reverse_index.get(tname, [])
        join.append(
            {
                "name": tname,
                "workspace_resource_type": w_rt,
                "manifest_resource_types": m_rts,
            }
        )

    _rt_tables = manifest_data.get("resource_type_tables", {}) or {}
    _resource_types_list = sorted(_rt_tables.keys())
    _providers_list = sorted({rt.split("/")[0] for rt in _rt_tables})
    return jsonify(
        {
            "success": True,
            "status": "ready",
            "workspace_id": workspace_id,
            "manifest": {
                "resource_type_tables": _rt_tables,
                "table_resource_types": manifest_data.get("table_resource_types", {}),
                "resource_types": _resource_types_list,
                "providers": _providers_list,
                "fetched_at": manifest_data.get("fetched_at"),
                "manifests_scanned": manifest_data.get("manifests_scanned"),
                "counts": {
                    "resource_types": len(_resource_types_list),
                    "providers": len(_providers_list),
                    "tables": sum(len(v) for v in _rt_tables.values()),
                },
            },
            "workspace": {
                "tables": [
                    {
                        "name": e.get("name"),
                        "resource_type": e.get("resource_type") or "Unknown",
                        "manifest_resource_types": e.get("manifest_resource_types", []),
                    }
                    for e in workspace_tables
                ],
                "retrieved_at": ws_retrieved,
                "count": len(workspace_tables),
                "source": ws_source,
            },
            "table_join": join,
        }
    )


def _scan_manifest_resource_types() -> dict:
    """Scan NGSchema manifest files to enumerate resource types/providers, map tables, and pick up query definitions.

    Returns a dict with keys:
        resource_types: List[str]
        providers: List[str]
        counts: { 'resource_types': int, 'providers': int, 'tables': int }
        resource_type_tables: { resource_type: [tableName, ...] }
        table_resource_type: { tableName: resource_type }
        queries: [ { 'resource_type': str, 'provider': str, 'name': str, 'description': str, 'table': str|None, 'path': str|None, 'manifest_file': str } ]
        queries_by_provider: { provider: [query, ...] }
        queries_by_resource_type: { resource_type: [query, ...] }
        retrieved_at: ISO timestamp

    Query extraction heuristics:
        - Look for top-level or nested keys named 'queries', 'sampleQueries', 'queryExamples'
        - Each item may include name/description/table/path or similar fields
        - Record path relative to repo if provided or attempt to infer when key 'file'/'path' present
    """
    if _workspace_resource_types_cache.get(
        "resource_types"
    ) and _workspace_resource_types_cache.get("resource_type_tables"):
        return _workspace_resource_types_cache

    base_dir = os.path.join(os.path.dirname(__file__), "NGSchema")
    if not os.path.isdir(base_dir):
        print("[Manifest Scan] NGSchema directory not found; skipping.")
        _workspace_resource_types_cache.update(
            {
                "resource_types": [],
                "providers": [],
                "counts": {"resource_types": 0, "providers": 0, "tables": 0},
                "resource_type_tables": {},
                "table_resource_type": {},
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return _workspace_resource_types_cache

    import json

    resource_types = set()
    resource_type_tables = {}
    table_resource_type = {}
    table_metadata = (
        {}
    )  # table_name -> { description, columns: [ {name,type,description?} ], resource_types:[...] }
    extracted_queries = []  # flat list of query metadata
    table_queries = (
        {}
    )  # table_name -> [ { name, description, resource_type, provider, manifest_file } ]
    manifest_files = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(".manifest.json"):
                manifest_files.append(os.path.join(root, fname))

    print(f"[Manifest Scan] Found {len(manifest_files)} manifest files. Parsing...")

    def _extract_types(obj, current_resource_type=None, manifest_path=None):
        if isinstance(obj, dict):
            tval = obj.get("type")
            if isinstance(tval, str) and "/" in tval:
                resource_types.add(tval)
                current_resource_type = tval
                resource_type_tables.setdefault(current_resource_type, [])
            # Strict query collection: ONLY process items under a 'queries' list.
            # Each query's related tables come EXCLUSIVELY from its 'relatedTables' array.
            if "queries" in obj and isinstance(obj["queries"], list):
                for q in obj["queries"]:
                    if not isinstance(q, dict):
                        continue
                    q_name = q.get("name") or q.get("title") or q.get("queryName")
                    if not q_name:
                        continue  # skip nameless entries
                    q_desc = q.get("description") or q.get("summary") or ""
                    q_path = (
                        q.get("path")
                        or q.get("file")
                        or q.get("kqlFile")
                        or q.get("kql_path")
                    )
                    # Only consider 'relatedTables' as authoritative mapping
                    related_list = q.get("relatedTables")
                    if not isinstance(related_list, list) or not related_list:
                        # If no relatedTables provided, we treat it as global (no table association)
                        related_tables = []
                    else:
                        # Filter to non-empty string table names
                        related_tables = [
                            t for t in related_list if isinstance(t, str) and t
                        ]
                    q_record = {
                        "resource_type": current_resource_type,
                        "provider": (
                            current_resource_type.split("/")[0]
                            if current_resource_type
                            else None
                        ),
                        "name": q_name,
                        "description": q_desc,
                        "path": q_path,
                        "manifest_file": manifest_path,
                        "related_tables": sorted(related_tables),
                    }
                    extracted_queries.append(q_record)
                    for rt_table in related_tables:
                        table_queries.setdefault(rt_table, []).append(
                            {
                                "name": q_name,
                                "description": q_desc,
                                "resource_type": current_resource_type,
                                "provider": (
                                    current_resource_type.split("/")[0]
                                    if current_resource_type
                                    else None
                                ),
                                "manifest_file": manifest_path,
                            }
                        )
            if "tables" in obj and isinstance(obj["tables"], list):
                for tbl in obj["tables"]:
                    if isinstance(tbl, dict):
                        tname = tbl.get("name") or tbl.get("tableName")
                        if tname and current_resource_type:
                            if tname not in resource_type_tables[current_resource_type]:
                                resource_type_tables[current_resource_type].append(
                                    tname
                                )
                                table_resource_type.setdefault(
                                    tname, current_resource_type
                                )
                            # Capture table metadata
                            tdesc = tbl.get("description") or tbl.get("summary") or ""
                            columns = []
                            raw_cols = tbl.get("columns") or tbl.get("schema") or []
                            if isinstance(raw_cols, list):
                                for c in raw_cols:
                                    if isinstance(c, dict):
                                        cname = c.get("name") or c.get("columnName")
                                        ctype = c.get("type") or c.get("dataType")
                                        cdesc = c.get("description") or ""
                                        if cname:
                                            columns.append(
                                                {
                                                    "name": cname,
                                                    "type": ctype,
                                                    "description": cdesc,
                                                }
                                            )
                            meta = table_metadata.setdefault(
                                tname,
                                {
                                    "descriptions": set(),
                                    "columns": {},
                                    "resource_types": set(),
                                },
                            )
                            if tdesc:
                                meta["descriptions"].add(tdesc)
                            meta["resource_types"].add(current_resource_type)
                            for col in columns:
                                existing = meta["columns"].get(col["name"])
                                if not existing:
                                    meta["columns"][col["name"]] = {
                                        "type": col["type"],
                                        "descriptions": set(),
                                    }
                                if (
                                    col.get("type")
                                    and not meta["columns"][col["name"]]["type"]
                                ):
                                    meta["columns"][col["name"]]["type"] = col["type"]
                                if col.get("description"):
                                    meta["columns"][col["name"]]["descriptions"].add(
                                        col["description"]
                                    )
            for v in obj.values():
                _extract_types(
                    v,
                    current_resource_type=current_resource_type,
                    manifest_path=manifest_path,
                )
        elif isinstance(obj, list):
            for item in obj:
                _extract_types(
                    item,
                    current_resource_type=current_resource_type,
                    manifest_path=manifest_path,
                )

    for mpath in manifest_files:
        try:
            with open(mpath, "r", encoding="utf-8") as mf:
                data = json.load(mf)
            _extract_types(data, manifest_path=mpath)
        except Exception as e:  # noqa: BLE001
            print(f"[Manifest Scan] Error parsing {mpath}: {e}")

    providers = {rt.split("/")[0] for rt in resource_types}
    resource_types_list = sorted(resource_types)
    providers_list = sorted(providers)

    # Organize queries by provider and resource type
    queries_by_provider = {}
    queries_by_resource_type = {}
    for q in extracted_queries:
        prov = q.get("provider") or "Unknown"
        rt = q.get("resource_type") or "Unknown"
        queries_by_provider.setdefault(prov, []).append(q)
        queries_by_resource_type.setdefault(rt, []).append(q)

    _workspace_resource_types_cache.update(
        {
            "resource_types": resource_types_list,
            "providers": providers_list,
            "counts": {
                "resource_types": len(resource_types_list),
                "providers": len(providers_list),
                "tables": sum(len(v) for v in resource_type_tables.values()),
            },
            "resource_type_tables": resource_type_tables,
            "table_resource_type": table_resource_type,
            "queries": extracted_queries,
            "queries_by_provider": queries_by_provider,
            "queries_by_resource_type": queries_by_resource_type,
            "table_queries": table_queries,
            "table_metadata": {
                t: {
                    "description": (
                        next(iter(m["descriptions"])) if m["descriptions"] else ""
                    ),
                    "all_descriptions": sorted(m["descriptions"]),
                    "columns": [
                        {
                            "name": cname,
                            "type": cinfo.get("type"),
                            "descriptions": sorted(cinfo.get("descriptions")),
                        }
                        for cname, cinfo in sorted(m["columns"].items())
                    ],
                    "resource_types": sorted(m["resource_types"]),
                }
                for t, m in table_metadata.items()
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # print('[Manifest Scan] Summary:')
    # print(f"  Resource Types: {len(resource_types_list)}")
    # print(f"  Providers: {len(providers_list)}")
    # total_tables = sum(len(v) for v in resource_type_tables.values())
    # print(f"  Tables (mapped): {total_tables}")
    # if extracted_queries:
    # print(f"  Queries extracted: {len(extracted_queries)}")
    # print(f"  Tables with query references: {len(table_queries)}")
    # if table_metadata:
    # print(f"  Tables with metadata: {len(table_metadata)}")
    # preview_types = resource_types_list[:20]
    # if preview_types:
    #     print('  Sample Resource Types:')
    #     for t in preview_types:
    #         print(f'    - {t}')
    #     if len(resource_types_list) > len(preview_types):
    #         print(f'    ... (+{len(resource_types_list) - len(preview_types)} more)')
    # print('  Providers: ' + ', '.join(providers_list[:15]) + (' ...' if len(providers_list) > 15 else ''))
    # print('  Sample Resource Type -> Tables:')
    # shown = 0
    # for rt in preview_types:
    #     tables_preview = resource_type_tables.get(rt, [])[:5]
    #     if tables_preview:
    #         print(f'    * {rt}: {", ".join(tables_preview)}' + (' ...' if len(resource_type_tables.get(rt, [])) > 5 else ''))
    #         shown += 1
    #     if shown >= 8:
    #         break

    return _workspace_resource_types_cache


def _get_azure_credential():
    """Compatibility shim retained for code paths expecting a credential.

    Prefer using `utils.kql_exec.get_logs_client()` and `execute_kql_query()` for
    data-plane operations. This function will attempt to return a DefaultAzureCredential
    when available for management-plane calls; otherwise returns None.
    """
    global _azure_credential
    if _azure_credential is not None:
        return _azure_credential

    with _credential_creation_lock:
        if _azure_credential is not None:
            return _azure_credential
        try:
            from utils.kql_exec import get_logs_client

            client = get_logs_client()
            # If we successfully got a LogsQueryClient, we don't need to create a separate credential
            if client is not None:
                _azure_credential = client
                return _azure_credential
        except Exception:
            pass

        # Fallback to legacy DefaultAzureCredential if present
        if DefaultAzureCredential is None:
            return None
        try:
            _azure_credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
            return _azure_credential
        except Exception:
            return None


def _fetch_workspace_tables(workspace: str):
    """Deprecated (cache removed). Kept for backward compatibility; now just performs a direct fetch and logs summary."""
    if not workspace:
        print("[Workspace Schema] No workspace ID provided.")
        return
    result = get_workspace_schema(workspace)
    if result.get("error"):
        print(f"[Workspace Schema] Error: {result['error']}")
        return
    tables = result.get("tables", [])
    print(
        f"[Workspace Schema] Direct fetch tables={len(tables)} source={result.get('source')} refreshed={result.get('refreshed')}"
    )


def _background_fetch_workspace_tables(workspace: str):
    """Deprecated shim retained for compatibility; just calls get_workspace_schema and logs summary."""
    if not workspace:
        return
    try:
        result = get_workspace_schema(workspace)
        if result.get("error"):
            print(
                f"[Workspace Schema] Background fetch error workspace={workspace}: {result['error']}"
            )
        else:
            print(
                f"[Workspace Schema] Background direct fetch tables={len(result.get('tables', []))} source={result.get('source')} refreshed={result.get('refreshed')}"
            )
    except Exception as e:  # noqa: BLE001
        print(
            f"[Workspace Schema] Background fetch unexpected error workspace={workspace}: {e}"
        )


@app.route("/")
def index():
    """Serve the main web interface"""
    return render_template("index.html")


@app.route("/api/setup", methods=["POST"])
def setup_workspace():
    """Setup the workspace ID for the agent"""
    global agent, workspace_id

    try:
        data = request.get_json()
        workspace_id = data.get("workspace_id", "").strip()

        if not workspace_id:
            return jsonify({'success': False, 'error': 'Workspace ID is required'})
        
        # Get user token from Azure AD authentication if available
        user_token = get_user_token()
        
        # Initialize agent (do not pass user_token to KQLAgent constructor)
        agent = KQLAgent(workspace_id)
        # Intentionally do NOT start schema fetch here to allow client to trigger and observe pending state
        print(
            "[Setup] Workspace initialized; schema fetch will start on first /api/workspace-schema request."
        )
        # Skip early persistence; we'll persist only once tables or enrichment are available.

        return jsonify(
            {
                "success": True,
                "message": f"Agent initialized for workspace: {workspace_id}",
                "schema_disabled": DISABLE_SCHEMA_FETCH,
            }
        )

    except Exception as e:
        return jsonify(
            {"success": False, "error": f"Failed to setup workspace: {str(e)}"}
        )


@app.route("/api/clear-workspace-cache", methods=["POST"])
def clear_workspace_cache():
    """Deprecated: cache removed. Always returns success without side effects."""
    global workspace_id
    if not workspace_id:
        return jsonify({"success": False, "error": "Workspace not initialized"}), 400
    return jsonify(
        {
            "success": True,
            "workspace_id": workspace_id,
            "cache_cleared": False,
            "message": "Schema caching removed; nothing to clear.",
        }
    )


@app.route("/api/generate-kql", methods=["POST"])
def process_query():
    """Process a natural language question and generate KQL query"""
    global agent

    try:
        if not agent:
            return jsonify(
                {
                    "success": False,
                    "error": "Agent not initialized. Please setup workspace first.",
                }
            )

        data = request.get_json()
        question = data.get("question", "").strip()
        model = data.get("model")  # Optional model override
        system_prompt = data.get("system_prompt")  # Optional system prompt override

        if not question:
            return jsonify({"success": False, "error": "Question is required"})

        # Use shared generator helper to ensure identical generation behavior
        from utils.generator import generate_kql

        result = generate_kql(agent, question, model=model, system_prompt=system_prompt)

        # If the helper returned a coroutine result that couldn't be run, ensure we handle it
        response_payload = {
            "success": True,
            "result": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Expose examples_error and candidate scores when container selection failed
        if (
            isinstance(result, str)
            and result.startswith("// Error")
            and "domain=containers" in result
            and "selected_example_count=0" in result
        ):
            try:
                from nl_to_kql import load_container_shots

                ctx = load_container_shots(question)
                response_payload["examples_error"] = ctx.get("examples_error")
                response_payload["top_candidate_scores"] = ctx.get(
                    "top_candidate_scores"
                )
            except Exception as expose_exc:
                response_payload["examples_error"] = f"expose_failed: {expose_exc}"
        return jsonify(response_payload)
        # cleanup is handled inside generate_kql

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        return jsonify(
            {"success": False, "error": error_msg, "traceback": traceback_str}
        )


@app.route("/api/explain", methods=["POST"])
def explain_query_result():
    """Explain previously returned query results.

    Expects JSON body:
      {
        "query_result": <result object returned by /api/query>,
        "original_question": "string"
      }

    Returns JSON:
      success: bool
      explanation: str (present on success)
      error: str (present on failure)
      timestamp: ISO8601
    """
    global agent
    try:
        if not agent:
            return jsonify({"success": False, "error": "Agent not initialized"}), 400
        payload = request.get_json(silent=True) or {}
        query_result = payload.get("query_result")
        original_question = payload.get("original_question", "")
        if not query_result:
            return jsonify({"success": False, "error": "query_result is required"}), 400
        # Ensure result structure minimally matches expectations
        if (
            not isinstance(query_result, dict)
            or query_result.get("type") != "query_success"
        ):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "query_result must be a successful query response",
                    }
                ),
                400,
            )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            explanation = loop.run_until_complete(
                agent.explain_results(query_result, original_question)
            )
        finally:
            loop.close()
        if isinstance(explanation, str) and explanation.startswith("❌"):
            return jsonify({"success": False, "error": explanation}), 200
        return jsonify(
            {
                "success": True,
                "explanation": explanation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Explain] Unexpected error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/refresh-workspace-schema", methods=["GET", "POST"])
def refresh_workspace_schema():
    """Compatibility refresh endpoint (stateless model).

    Accepts optional JSON {"refetch": bool}. Always returns 200 with summary.
    If workspace not initialized, returns success=False but does not error.
    """
    global workspace_id
    if not workspace_id:
        return (
            jsonify(
                {
                    "success": False,
                    "workspace_id": None,
                    "message": "Workspace not initialized",
                    "table_count": 0,
                }
            ),
            200,
        )
    try:
        payload = request.get_json(silent=True) or {}
        if not payload.get("refetch", True):
            return (
                jsonify(
                    {
                        "success": True,
                        "workspace_id": workspace_id,
                        "message": "No refetch requested",
                        "table_count": 0,
                    }
                ),
                200,
            )
        result = get_workspace_schema(workspace_id)
        if result.get("error"):
            return (
                jsonify(
                    {
                        "success": False,
                        "workspace_id": workspace_id,
                        "error": result.get("error"),
                        "table_count": 0,
                    }
                ),
                200,
            )
        tables = result.get("tables", [])
        return (
            jsonify(
                {
                    "success": True,
                    "workspace_id": workspace_id,
                    "refetched": True,
                    "table_count": len(tables),
                    "retrieved_at": result.get("retrieved_at"),
                    "source": result.get("source"),
                }
            ),
            200,
        )
    except Exception as e:  # noqa: BLE001
        return (
            jsonify(
                {
                    "success": False,
                    "workspace_id": workspace_id,
                    "error": str(e),
                    "table_count": 0,
                }
            ),
            200,
        )

        # Read and parse the example file
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()

        # Extract example queries (simple parsing - look for lines starting with specific patterns)
        examples = []
        lines = content.split("\n")
        current_example = None

        for line in lines:
            line = line.strip()
            if line.startswith("# ") or line.startswith("## "):
                if current_example:
                    examples.append(current_example)
                current_example = line.replace("#", "").strip()
            elif line.startswith("- ") and current_example:
                example_text = line.replace("- ", "").strip()
                if example_text:
                    examples.append(example_text)
                    current_example = None
        # If we have fewer than 5 examples, add some generic ones
        if len(examples) < 5:
            generic_examples = {
                "requests": [
                    "Show me failed requests from the last hour",
                    "What are the slowest requests in the last 24 hours?",
                    "Show me requests with response time > 5 seconds",
                    "Get the top 10 most frequent request URLs",
                    "Show me requests grouped by status code",
                ],
                "exceptions": [
                    "Show me recent exceptions",
                    "What are the most common exception types?",
                    "Show me exceptions from the last 6 hours",
                    "Get exception count by severity level",
                    "Show me exceptions grouped by operation name",
                ],
                "traces": [
                    "Show me recent trace logs",
                    "What are the most frequent trace messages?",
                    "Show me error traces from the last hour",
                    "Get traces with specific severity level",
                    "Show me traces grouped by source",
                ],
                "dependencies": [
                    "Show me dependency failures",
                    "What are the slowest dependencies?",
                    "Show me dependencies with high failure rate",
                    "Get dependency calls from the last hour",
                    "Show me dependencies grouped by type",
                ],
                "custom_events": [
                    "Show me recent custom events",
                    "What are the most frequent custom event types?",
                    "Show me custom events from the last hour",
                    "Get custom events grouped by name",
                    "Show me custom events with specific properties",
                ],
                "page_views": [
                    "Show me page views from the last hour",
                    "What are the most popular pages?",
                    "Show me page views grouped by browser",
                    "Get page load times by URL",
                    "Show me page views by geographic location",
                ],
                "performance": [
                    "Show me performance counters",
                    "What are the CPU usage trends?",
                    "Show me memory usage over time",
                    "Get performance metrics for the last hour",
                    "Show me performance counters by category",
                ],
                "usage": [
                    "Show me user activity patterns",
                    "What are the most popular features?",
                    "Show me usage statistics by region",
                    "Get daily active users",
                    "Show me usage trends over time",
                ],
            }

            if scenario in generic_examples:
                examples.extend(generic_examples[scenario][: 5 - len(examples)])

        # Limit to top 8 examples
        examples = examples[:8]

        return jsonify(
            {
                "success": True,
                "result": {
                    "type": "example_suggestions",
                    "scenario": scenario,
                    "suggestions": examples,
                    "count": len(examples),
                },
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/workspace-examples", methods=["POST"])
def discover_workspace_examples():
    """Discover workspace tables and map them to available example queries"""
    global agent, workspace_id

    try:
        # Allow workspace examples discovery even without agent initialization
        # since we're just showing available example files

        import glob
        import os

        # Define table mappings to example files
        table_examples_map = {
            "AppRequests": {
                "file": "app_insights_capsule/kql_examples/app_requests_kql_examples.md",
                "category": "Application Insights",
                "description": "HTTP requests to your application",
            },
            "AppExceptions": {
                "file": "app_insights_capsule/kql_examples/app_exceptions_kql_examples.md",
                "category": "Application Insights",
                "description": "Exceptions thrown by your application",
            },
            "AppTraces": {
                "file": "app_insights_capsule/kql_examples/app_traces_kql_examples.md",
                "category": "Application Insights",
                "description": "Custom trace logs from your application",
            },
            "AppDependencies": {
                "file": "app_insights_capsule/kql_examples/app_dependencies_kql_examples.md",
                "category": "Application Insights",
                "description": "External dependencies called by your application",
            },
            "AppPageViews": {
                "file": "app_insights_capsule/kql_examples/app_page_views_kql_examples.md",
                "category": "Application Insights",
                "description": "Page views in your web application",
            },
            "AppCustomEvents": {
                "file": "app_insights_capsule/kql_examples/app_custom_events_kql_examples.md",
                "category": "Application Insights",
                "description": "Custom events tracked by your application",
            },
            "AppPerformanceCounters": {
                "file": "app_insights_capsule/kql_examples/app_performance_kql_examples.md",
                "category": "Application Insights",
                "description": "Performance counters and metrics",
            },
            "Usage": {
                "file": "usage_kql_examples.md",
                "category": "Usage Analytics",
                "description": "User behavior and usage patterns",
            },
        }

        # Get available example files
        example_files = glob.glob("*_kql_examples.md")
        available_examples = {}

        for table, info in table_examples_map.items():
            if os.path.exists(info["file"]):
                available_examples[table] = info

        # Count examples by category
        example_categories = {}
        for table, info in available_examples.items():
            category = info["category"]
            example_categories[category] = example_categories.get(category, 0) + 1

        # Simulate discovered tables (in a real implementation, you'd query the workspace)
        discovered_tables = list(available_examples.keys())

        # Create summary
        summary = {
            "workspace_id": workspace_id or "Not configured",
            "total_tables": len(discovered_tables),
            "tables_with_examples": len(available_examples),
            "example_categories": example_categories,
        }

        # Create table details in the format expected by the frontend
        available_examples_formatted = {}
        for table in discovered_tables:
            if table in available_examples:
                info = available_examples[table]
                available_examples_formatted[table] = {
                    "table_info": {
                        "record_count": 10000,  # Simulated count, would be real in production
                        "category": info["category"],
                        "description": info["description"],
                    },
                    "examples": [
                        {
                            "source": "",
                            "description": "",  # Remove duplicate description (now shown in table header)
                            "query_count": 5,  # Simulated count
                        }
                    ],
                }

        return jsonify(
            {
                "success": True,
                "summary": summary,
                "available_examples": available_examples_formatted,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/examples/<scenario>", methods=["GET"])
def get_examples_for_scenario(scenario):
    """Get example queries for a specific scenario/table.

    Maps scenario names (like 'requests', 'exceptions') to table names
    and returns suggestions from the example files.
    """
    try:
        import os

        # Map scenario names to table names and example files
        scenario_to_table_map = {
            "requests": (
                "AppRequests",
                "app_insights_capsule/kql_examples/app_requests_kql_examples.md",
            ),
            "exceptions": (
                "AppExceptions",
                "app_insights_capsule/kql_examples/app_exceptions_kql_examples.md",
            ),
            "traces": (
                "AppTraces",
                "app_insights_capsule/kql_examples/app_traces_kql_examples.md",
            ),
            "dependencies": (
                "AppDependencies",
                "app_insights_capsule/kql_examples/app_dependencies_kql_examples.md",
            ),
            "page_views": (
                "AppPageViews",
                "app_insights_capsule/kql_examples/app_page_views_kql_examples.md",
            ),
            "custom_events": (
                "AppCustomEvents",
                "app_insights_capsule/kql_examples/app_custom_events_kql_examples.md",
            ),
            "performance": (
                "AppPerformanceCounters",
                "app_insights_capsule/kql_examples/app_performance_kql_examples.md",
            ),
            "usage": ("Usage", "usage_kql_examples.md"),
        }

        if scenario not in scenario_to_table_map:
            return jsonify({"success": False, "error": f"Unknown scenario: {scenario}"})

        table_name, example_file = scenario_to_table_map[scenario]

        # Check if file exists
        if not os.path.exists(example_file):
            return jsonify(
                {"success": True, "result": {"table": table_name, "suggestions": []}}
            )

        # Parse example file for suggestions
        suggestions = []
        try:
            with open(example_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract headers (lines starting with #) as suggestions
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("# ") and not line.startswith("## "):
                    # Remove markdown header syntax
                    suggestion = line.lstrip("#").strip()
                    if (
                        suggestion and len(suggestion) > 5
                    ):  # Filter out very short headers
                        suggestions.append(suggestion)
        except Exception as e:
            print(f"[Examples] Error parsing {example_file}: {e}")

        return jsonify(
            {
                "success": True,
                "result": {
                    "table": table_name,
                    "suggestions": suggestions[:10],  # Limit to 10 suggestions
                },
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/example-catalog", methods=["POST"])
def example_catalog():
    """Return unified example + (optional) live schema catalog.

    Expects JSON body: {"include_schema": bool, "force": bool}
    """
    # Example catalog disabled - return empty catalog
    return jsonify({"success": True, "catalog": {"tables": [], "examples": []}})


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_from_directory("static", filename)


@app.route("/api/score-query", methods=["POST"])
def score_query():
    """Score a generated query against an expected query."""
    try:
        from query_scorer import calculate_total_score

        data = request.get_json()

        # Required fields
        generated_kql = data.get("generated_kql", "")
        expected_kql = data.get("expected_kql", "")
        prompt = data.get("prompt", "")

        # Query execution results
        generated_columns = data.get("generated_columns", [])
        expected_columns = data.get("expected_columns", [])
        generated_results = data.get("generated_results", [])
        expected_results = data.get("expected_results", [])

        # Calculate score (uses same Azure OpenAI deployment as query generation)
        score_result = calculate_total_score(
            generated_kql=generated_kql,
            expected_kql=expected_kql,
            generated_columns=generated_columns,
            expected_columns=expected_columns,
            generated_results=generated_results,
            expected_results=expected_results,
            prompt=prompt,
        )

        return jsonify({"success": True, "score": score_result})

    except Exception as e:
        return (
            jsonify(
                {"success": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )


def _execute_kql_query(query: str):
    """Internal helper to execute a KQL query and return results.

    Returns:
        dict with 'success', 'tables' (if successful), or 'error' (if failed)
    """
    # Delegate to shared execution helper so server and CLI use identical execution logic
    try:
        from utils.kql_exec import execute_kql_query as _exec, normalize_status

        res = _exec(query)
        # Extract exec_stats if present and normalize
        raw_exec_stats = res.get("exec_stats", {}) if isinstance(res, dict) else {}
        status_candidate = raw_exec_stats.get("status") or raw_exec_stats.get("raw_status")
        normalized_status = normalize_status(status_candidate) or "UNKNOWN"

        # Derive ui_status
        if normalized_status in ("SUCCESS", "SUCCEEDED"):
            ui_status = "success"
        elif normalized_status in ("NO_DATA", "NODATA"):
            ui_status = "no_data"
        elif normalized_status in ("FAILED", "FAILURE", "UNKNOWN"):
            ui_status = "failed"
        else:
            ui_status = "error"

        exec_stats_out = {"status": normalized_status, "ui_status": ui_status}
        if "raw_status" in raw_exec_stats:
            try:
                exec_stats_out["raw_status"] = str(raw_exec_stats.get("raw_status"))
            except Exception:
                exec_stats_out["raw_status"] = "<unserializable>"

        # Keep legacy shape expected by callers: { success: bool, tables: [...] } or { success: False, error: str }
        if res.get("error"):
            return {"success": False, "error": res.get("error"), "exec_stats": exec_stats_out}
        # Convert rows (list of dicts) to tables shape if present
        rows = res.get("rows", [])
        if rows:
            # if rows are list of dicts, produce a single table with columns and rows
            if isinstance(rows[0], dict):
                cols = list(rows[0].keys())
                table_rows = [[r.get(c) for c in cols] for r in rows]
                table = {
                    "name": "table_0",
                    "columns": cols,
                    "rows": table_rows,
                    "row_count": len(table_rows),
                }
                return {"success": True, "tables": [table], "exec_stats": exec_stats_out}
            else:
                # assume rows already table-shaped
                return {"success": True, "tables": rows, "exec_stats": exec_stats_out}
        return {"success": True, "tables": [], "exec_stats": exec_stats_out}
    except Exception as e:
        from logs_agent import extract_innermost_error

        innermost_error = extract_innermost_error(str(e))
        return {"success": False, "error": innermost_error}


@app.route("/api/generate_and_evaluate_query", methods=["POST"])
def evaluate_query():
    """Evaluate a generated query against an expected query.

    This endpoint:
    1. Executes the generated query
    2. If successful, executes the expected query
    3. Compares results (all rows, not limited)
    4. Performs LLM scoring
    5. Returns complete evaluation

    Request body:
        prompt: str - The natural language prompt
        generated_query: str - The generated KQL query
        expected_query: str - The expected KQL query

    Response:
        success: bool
        execution_success: bool - Whether both queries executed successfully
        generated_error: str (optional) - Error from generated query
        expected_error: str (optional) - Error from expected query
        score: dict (optional) - Scoring results if both queries succeeded
    """
    global agent

    try:
        if not agent:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Agent not initialized. Please setup workspace first.",
                    }
                ),
                400,
            )

        data = request.get_json()
        prompt = (data.get("prompt") or "").strip()
        generated_query = data.get("generated_query")
        if isinstance(generated_query, str):
            generated_query = generated_query.strip()
        expected_query = (data.get("expected_query") or "").strip()
        model = data.get("model")  # Optional model override
        system_prompt = data.get("system_prompt")  # Optional system prompt override

        # Set overrides if provided (kept for backward compatibility; main orchestration
        # applies overrides too). We'll still set them here to ensure older code paths
        # are supported during transition.
        from azure_openai_utils import clear_model_override, set_model_override
        from prompt_builder import (clear_system_prompt_override,
                                    set_system_prompt_override)

        if model:
            set_model_override(model)
        if system_prompt:
            set_system_prompt_override(system_prompt)

        # If caller did not supply a generated query, let the shared orchestrator
        # perform generation so the server and run_batch use identical code paths.
        if not generated_query:
            generated_query = None
        if not expected_query:
            clear_model_override()
            return (
                jsonify({"success": False, "error": "Expected query is required"}),
                400,
            )

        # Use the shared helper to evaluate this single prompt case so behavior
        # is identical to `run_batch`. If `generated_query` was provided, pass it
        # through the case to avoid re-generating; otherwise let the helper
        # generate it.
        from utils.batch_runner import generate_and_evaluate_query

        case = {"id": None, "prompt": prompt, "expected_query": expected_query}
        if generated_query:
            case["generated_query"] = generated_query
        res = generate_and_evaluate_query(
            case, agent=agent, execute=True, model=model, system_prompt=system_prompt
        )

        # Map helper result to API shape. Do not infer execution success only from
        # helper `status` (which is a generic flow indicator). Instead inspect
        # returned exec stats and any `error` on the generated_query object so
        # the API accurately reflects query execution failures.
        gen_q = res.get("generated_query")
        gen_exec_stats = res.get("returned_exec_stats") or {}
        exp_exec_stats = res.get("expected_exec_stats") or {}

        # Detect generated execution error.
        # Priority: explicit exec_stats error (server/SDK) should take precedence.
        # Do NOT treat a benign 'message' string (e.g. 'Query executed successfully')
        # on the `generated_query` dict as an error unless the payload indicates an error.
        gen_err = None
        if isinstance(gen_exec_stats, dict):
            gen_err = gen_exec_stats.get("error") or gen_exec_stats.get("message")

        # If exec_stats did not indicate an error, inspect generated_query object
        if not gen_err and isinstance(gen_q, dict):
            # If the generator explicitly marked an error type, respect it
            qtype = (gen_q.get("type") or "").lower()
            if qtype == "query_error":
                gen_err = gen_q.get("error") or gen_q.get("message")
            else:
                # If generated_query is expected to be a string-like object, consider
                # absence of a kql_query a failure. Only treat 'message' as error when
                # it contains failure keywords.
                if not gen_q.get("kql_query"):
                    msg = gen_q.get("message") or ""
                    if any(k in msg.lower() for k in ("error", "failed", "could not", "no value")):
                        gen_err = msg

        # Detect expected execution error
        exp_err = None
        if isinstance(exp_exec_stats, dict):
            exp_err = exp_exec_stats.get("error") or exp_exec_stats.get("message")

        execution_success = gen_err is None and exp_err is None

        # Build a legacy-compatible `result` object for the UI templates.
        # Templates expect `{ type: 'query_success' | 'query_error', kql_query, data?, error?, message? }`.
        result_obj = None
        gen_q_val = None
        if isinstance(gen_q, dict):
            gen_q_val = gen_q.get("kql_query") or gen_q.get("kql") or gen_q.get("query")
        else:
            gen_q_val = gen_q

        if not execution_success:
            # Prefer generated error, fallback to expected error
            err_msg = gen_err or exp_err or "Query execution failed"
            result_obj = {
                "type": "query_error",
                "kql_query": gen_q_val,
                "error": err_msg,
                "message": f"Query execution failed: {err_msg}",
            }
        else:
            # Execution succeeded; we don't return full rows here, so surface a no-data placeholder
            result_obj = {
                "type": "query_success",
                "kql_query": gen_q_val,
                "data": {"type": "no_data", "message": "Results suppressed by server; see exec stats for counts."},
                "message": "Query executed successfully",
            }

        return jsonify(
            {
                "success": True,
                "execution_success": execution_success,
                "generated_error": gen_err,
                "expected_error": exp_err,
                "generated_query": res.get("generated_query"),
                "generated_exec_stats": gen_exec_stats,
                "expected_exec_stats": exp_exec_stats,
                "score": res.get("score"),
                "result": result_obj,
            }
        )

    except Exception as e:
        from azure_openai_utils import clear_model_override
        from prompt_builder import clear_system_prompt_override

        clear_model_override()  # Clean up on error
        clear_system_prompt_override()  # Clean up on error
        return (
            jsonify(
                {"success": False, "error": str(e), "traceback": traceback.format_exc()}
            ),
            500,
        )
    finally:
        from azure_openai_utils import clear_model_override
        from prompt_builder import clear_system_prompt_override

        clear_model_override()  # Always clean up
        clear_system_prompt_override()  # Always clean up


@app.route("/api/query-execute", methods=["POST"])
def query_execute():
    """Execute a KQL query and return results."""
    global agent

    try:
        if not agent:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Agent not initialized. Please setup workspace first.",
                    }
                ),
                400,
            )

        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400

        result = _execute_kql_query(query)
        return jsonify(result)

    except Exception as e:
        from logs_agent import extract_innermost_error

        innermost_error = extract_innermost_error(str(e))
        return jsonify({"success": False, "error": innermost_error})


@app.route("/api/batch-test/parse", methods=["POST"])
def batch_test_parse():
    """Parse Excel file and return list of prompts."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        return (
            jsonify(
                {"success": False, "error": "File must be .xlsx, .xls, or .csv format"}
            ),
            400,
        )

    try:
        from utils.file_parser import parse_prompts_from_file

        prompts, test_cases, df = parse_prompts_from_file(file)

        return jsonify({"success": True, "prompts": prompts, "test_cases": test_cases})
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"[BatchParse] Error parsing file: {error_details}")
        return (
            jsonify({"success": False, "error": f"Error parsing file: {str(e)}"}),
            500,
        )


@app.route("/api/batch-report/build", methods=["POST"])
def batch_report_build():
    """Build JSON + Excel report from a batch-run payload using shared report builder.

    Accepts payload: { results: [...], metadata: { ... } }
    Returns JSON containing base64-encoded Excel blob, json blob, file paths, and summary.
    """
    try:
        import base64
        import os
        from datetime import datetime

        data = request.json or {}
        results = data.get("results", [])
        metadata = data.get("metadata", {}) or {}

        if not results:
            return jsonify({"success": False, "error": "No results provided"}), 400

        # Augment metadata with model info if not provided
        model_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "Unknown")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "Unknown")
        from prompt_builder import SYSTEM_PROMPT

        system_prompt = SYSTEM_PROMPT

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Build artifacts using shared report builder
        report_json = build_json(results, metadata)
        try:
            excel_bytes = build_excel(results, metadata)
        except Exception:
            excel_bytes = None

        # Persist artifacts
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        base_filename = f"benchmark_{model_deployment}_{timestamp}"
        json_filename = f"{base_filename}.json"
        excel_filename = f"{base_filename}.xlsx"
        json_report_path = os.path.join(reports_dir, json_filename)
        excel_report_path = os.path.join(reports_dir, excel_filename)

        with open(json_report_path, "w", encoding="utf-8") as f:
            f.write(report_json)

        if excel_bytes:
            with open(excel_report_path, "wb") as f:
                f.write(excel_bytes)

        encoded_file = excel_bytes and base64.b64encode(excel_bytes).decode("utf-8")
        encoded_json = base64.b64encode(report_json.encode("utf-8")).decode("utf-8")

        response_summary = {
            "total": len(results),
            "successful": sum(
                1
                for r in results
                if r.get("execution_success") or r.get("status") == "success"
            ),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "wrong_query": sum(1 for r in results if r.get("status") == "wrong_query"),
            "total_returned_rows": sum(
                [r.get("returned_rows_count") or 0 for r in results]
            ),
        }

        return jsonify(
            {
                "success": True,
                "file_data": encoded_file,
                "filename": excel_filename,
                "json_file_data": encoded_json,
                "json_filename": json_filename,
                "json_report_path": json_report_path,
                "excel_report_path": excel_report_path,
                "model_info": {
                    "deployment": model_deployment,
                    "api_version": api_version,
                },
                "system_prompt": system_prompt,
                "summary": response_summary,
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "error": f"Error building file: {str(e)}"}),
            500,
        )


# Backwards-compatible alias for existing clients
@app.route("/api/batch-test/build", methods=["POST"])
def batch_test_build_alias():
    return batch_report_build()


@app.route("/api/batch-run", methods=["POST"])
def batch_run_endpoint():
    """Run server-side orchestration for a list of prompts using the shared runner.

    Expects JSON: { prompts: [{id, prompt, expected_query, ...}], execute: bool (optional), stop_on_critical_error: bool (optional), metadata: {} }
    Returns: { success: True, results: [...], summary: {...} }
    """
    try:
        data = request.get_json() or {}
        prompts = data.get("prompts") or data.get("results") or []
        execute_flag = bool(data.get("execute", False))
        stop_on_critical = bool(data.get("stop_on_critical_error", True))

        if not prompts:
            return jsonify({"success": False, "error": "No prompts provided"}), 400

        # Decide which agent to pass to the runner. We require an explicit
        # agent instance; if the global `agent` is not initialized (e.g. in
        # tests or lightweight scenarios), provide a minimal DummyAgent that
        # mirrors the previous fallback behavior but is explicit to this
        # endpoint.
        if agent:
            agent_to_use = agent
        else:
            class DummyAgentLocal:
                def process_natural_language(self, prompt: str):
                    return f"KQL: {prompt}"

            agent_to_use = DummyAgentLocal()

        # Run the batch using shared runner
        results = run_batch(
            prompts,
            agent=agent_to_use,
            execute=execute_flag,
            stop_on_critical_error=stop_on_critical,
        )

        summary = {
            "total": len(results),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "total_returned_rows": sum(
                [r.get("returned_rows_count", 0) or 0 for r in results]
            ),
        }

        return jsonify({"success": True, "results": results, "summary": summary})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/batch-test/upload", methods=["POST"])
def batch_test_upload():
    """Upload Excel file with prompts for batch testing"""
    global agent, workspace_id

    if not agent or not workspace_id:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Agent not initialized. Please setup workspace first.",
                }
            ),
            400,
        )

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        return (
            jsonify(
                {"success": False, "error": "File must be .xlsx, .xls, or .csv format"}
            ),
            400,
        )

    try:
        import pandas as pd

        # Read file based on extension
        if file.filename.endswith(".csv"):
            # Try flexible CSV parsing to handle mixed delimiters
            try:
                df = pd.read_csv(file)
            except Exception:
                # Reset file pointer and try with flexible parsing
                file.seek(0)
                try:
                    df = pd.read_csv(file, sep=None, engine="python")
                except Exception:
                    # Last resort: try with explicit quoting and skip bad lines
                    file.seek(0)
                    df = pd.read_csv(
                        file, quotechar='"', escapechar="\\", on_bad_lines="skip"
                    )
            # Ensure proper index and clean column names
            df = df.reset_index(drop=True)
            df.columns = df.columns.str.strip().str.replace("\t", "", regex=False)

            # Map common column name variations
            col_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ["question", "nl", "natural language", "ask"]:
                    col_map[col] = "Prompt"
                elif col_lower in [
                    "query",
                    "kql",
                    "expected",
                    "expected query",
                    "expected_query",
                ]:
                    col_map[col] = "Expected Query"
            if col_map:
                df = df.rename(columns=col_map)
        else:
            df = pd.read_excel(file)

        # Validate columns
        required_cols = ["Prompt"]
        if "Prompt" not in df.columns:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f'File must have a "Prompt" column. Found columns: {", ".join(df.columns)}',
                    }
                ),
                400,
            )

        # Add output columns if they don't exist
        if "Generated Query" not in df.columns:
            df["Generated Query"] = ""
        if "Reason" not in df.columns:
            df["Reason"] = ""

        # Process each row
        results = []
        for idx, row in df.iterrows():
            prompt = row["Prompt"]
            # Skip empty prompts silently (no logging, no display)
            if pd.isna(prompt) or not str(prompt).strip():
                continue
            prompt_str = str(prompt).strip()
            try:
                # Translate natural language to KQL
                result = asyncio.run(agent.process_natural_language(prompt_str))
                # Parse the result - it returns a formatted string, not a dict
                if isinstance(result, str):
                    # Check if result contains an error
                    if (
                        result.startswith("❌")
                        or "error" in result.lower()
                        or "failed" in result.lower()
                    ):
                        error_msg = result.replace("❌", "").strip()
                        df.at[idx, "Generated Query"] = ""
                        df.at[idx, "Reason"] = f"Error: {error_msg}"
                        results.append(
                            {
                                "row": idx + 1,
                                "prompt": prompt_str[:100],
                                "status": "error",
                                "reason": error_msg,
                                "score": None,
                            }
                        )
                    else:
                        # Extract KQL from the result (it's formatted with headers)
                        kql_query = result
                        # Try to extract just the query part if it has formatting
                        if "```kql" in result:
                            parts = result.split("```kql")
                            if len(parts) > 1:
                                kql_query = parts[1].split("```")[0].strip()
                        elif "📝 Generated KQL Query" in result:
                            parts = result.split("📝 Generated KQL Query")
                            if len(parts) > 1:
                                kql_query = parts[1].strip()
                        df.at[idx, "Generated Query"] = kql_query
                        df.at[idx, "Reason"] = "Successfully generated"
                        results.append(
                            {
                                "row": idx + 1,
                                "prompt": prompt_str[:100],
                                "status": "success",
                                "query_length": len(kql_query),
                                "score": None,  # Score calculation not available in this endpoint
                            }
                        )
                else:
                    df.at[idx, "Generated Query"] = ""
                    df.at[idx, "Reason"] = "Error: Unexpected result format"
                    results.append(
                        {
                            "row": idx + 1,
                            "prompt": prompt_str[:100],
                            "status": "error",
                            "reason": "Unexpected result format",
                            "score": None,
                        }
                    )
            except Exception as e:
                df.at[idx, "Generated Query"] = ""
                df.at[idx, "Reason"] = f"Exception: {str(e)}"
                results.append(
                    {
                        "row": idx + 1,
                        "prompt": prompt_str[:100],
                        "status": "error",
                        "reason": str(e),
                        "score": None,
                    }
                )

        # Build standardized Excel report for this run
        from utils.report_builder import build_excel, build_json

        metadata = {"source": "web-run"}
        try:
            excel_bytes = build_excel(
                results_for_report if "results_for_report" in locals() else results,
                metadata,
            )
        except Exception:
            excel_bytes = None

        # Return base64 encoded workbook as before
        import base64

        encoded_file = excel_bytes and base64.b64encode(excel_bytes).decode("utf-8")

        # Calculate summary
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")

        return jsonify(
            {
                "success": True,
                "file_data": encoded_file,
                "filename": f"results_{secure_filename(file.filename)}",
                "summary": {
                    "total": len(df),
                    "success": success_count,
                    "errors": error_count,
                    "skipped": skipped_count,
                },
                "results": results,
            }
        )

    except Exception as e:
        return (
            jsonify({"success": False, "error": f"Error processing file: {str(e)}"}),
            500,
        )


if __name__ == "__main__":
    print("Starting Natural Language KQL Agent Web Interface...")
    print("Features available:")
    print("   - Natural language to KQL translation")
    print("   - Interactive workspace setup")
    print("   - Query execution and results display")
    print("   - Example queries and suggestions")
    if not DISABLE_SCHEMA_FETCH:
        print("   - Workspace table discovery")
    else:
        print("   - Schema fetching: DISABLED (queries only)")
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"Starting server on http://localhost:8080 (debug={debug_mode})")
    try:
        app.run(debug=debug_mode, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n🛑 Web Interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        import traceback

        traceback.print_exc()
