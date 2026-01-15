# azure_agent/monitor_client.py
"""
This module provides a class for authenticating and querying Azure Monitor (Log Analytics) using the Azure SDK.
All methods are heavily commented for clarity and learning.
"""
from utils.kql_exec import get_logs_client, execute_kql_query


class AzureMonitorAgent:
    def __init__(self):
        """Light wrapper that delegates to `utils.kql_exec` for client and execution."""
        self.client = get_logs_client()

    def query_log_analytics(self, workspace_id, kql_query, timespan=None):
        try:
            exec_result = execute_kql_query(kql=kql_query, workspace_id=workspace_id, client=self.client, timespan=timespan)
            # Note: `exec_result['exec_stats']['status']` is normalized to an
            # upper-case string (e.g. "SUCCESS"). The SDK enum is preserved as
            # `exec_stats['raw_status']` when available. Consumers should prefer
            # `utils.kql_exec.is_success(...)` when checking success.
            if exec_result.get("tables") is not None:
                return {"tables": exec_result.get("tables")}
            return {"error": exec_result.get("exec_stats", {}).get("error")}
        except Exception as e:
            return {"error": str(e)}
