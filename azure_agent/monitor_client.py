# azure_agent/monitor_client.py
"""
This module provides a class for authenticating and querying Azure Monitor (Log Analytics) using the Azure SDK.
All methods are heavily commented for clarity and learning.
"""
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.core.credentials import AccessToken
from azure.monitor.query import LogsQueryStatus
import utils.kql_exec as kql_exec
from datetime import datetime, timedelta

class UserTokenCredential:
    """Credential that uses a user's access token from Azure AD authentication."""
    def __init__(self, access_token):
        self.access_token = access_token
    
    def get_token(self, *scopes, **kwargs):
        # Return token that expires in 1 hour (Azure AD tokens typically last 1 hour)
        expires_on = int((datetime.now() + timedelta(hours=1)).timestamp())
        return AccessToken(self.access_token, expires_on)

class AzureMonitorAgent:
    def __init__(self, user_token=None):
        """
        Initialize the AzureMonitorAgent.
        
        Args:
            user_token: Optional access token from authenticated user (Azure AD).
                       If provided, queries will run as that user.
                       If not provided, uses CLI credentials or managed identity.
        """
        if user_token:
            # Use the user's token from Azure AD authentication
            self.credential = UserTokenCredential(user_token)
        else:
            try:
                # Try Azure CLI credentials first
                self.credential = AzureCliCredential()
                # This will raise if az is not installed or not logged in
                _ = self.credential.get_token("https://management.azure.com/.default")
            except Exception:
                # Fallback to DefaultAzureCredential (env vars, managed identity, etc.)
                self.credential = DefaultAzureCredential()
        # For testability and optional SDK presence, use the shared get_logs_client helper
        try:
            self.client = kql_exec.get_logs_client(credential=self.credential)
        except Exception:
            self.client = None

    def query_log_analytics(self, workspace_id, kql_query, timespan=None):
        """
        Run a KQL query against a Log Analytics workspace.
        Args:
            workspace_id (str): The Log Analytics workspace ID (GUID).
            kql_query (str): The Kusto Query Language (KQL) query to run.
            timespan (str or tuple): ISO8601 duration or (start, end) tuple.
        Returns:
            dict: Query results or error message.
        """
        try:
            # Use the shared execution helper so behavior is consistent across
            # the codebase and so tests can monkeypatch `utils.kql_exec.execute_kql_query`.
            exec_result = kql_exec.execute_kql_query(kql=kql_query, workspace_id=workspace_id, client=self.client, timespan=timespan)
            # exec_result is a dict with keys: tables, returned_rows_count, exec_stats
            tables = exec_result.get("tables", [])
            if tables:
                return {"tables": tables}
            # If SDK returned an exec_stats error, surface it
            exec_stats = exec_result.get("exec_stats", {}) or {}
            if exec_stats.get("error"):
                return {"error": exec_stats.get("error")}
            # Otherwise, if there's a partial error or no tables, return a generic error
            return {"error": exec_result.get("error") or exec_stats.get("raw_status") or "Query failed"}
        except Exception as e:
            return {"error": str(e)}
