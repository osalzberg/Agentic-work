# azure_agent/monitor_client.py
"""
This module provides a class for authenticating and querying Azure Monitor (Log Analytics) using the Azure SDK.
All methods are heavily commented for clarity and learning.
"""
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.core.credentials import AccessToken
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
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
        self.client = LogsQueryClient(self.credential)

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
            response = self.client.query_workspace(
                workspace_id=workspace_id,
                query=kql_query,
                timespan=timespan
            )
            # Convert LogsTable objects to dicts manually
            if response.status == LogsQueryStatus.SUCCESS:
                tables = []
                for table in response.tables:
                    # Defensive: skip if table is not a LogsTable object
                    if not hasattr(table, 'name') or not hasattr(table, 'columns') or not hasattr(table, 'rows'):
                        continue
                    # Defensive: columns may be a list of dicts or strings, handle both
                    columns = []
                    for col in getattr(table, 'columns', []):
                        if hasattr(col, 'name'):
                            columns.append(col.name)
                        elif isinstance(col, dict) and 'name' in col:
                            columns.append(col['name'])
                        else:
                            columns.append(str(col))
                    table_dict = {
                        'name': getattr(table, 'name', ''),
                        'columns': columns,
                        'rows': getattr(table, 'rows', [])
                    }
                    tables.append(table_dict)
                return {"tables": tables}
            else:
                return {"error": getattr(response, 'partial_error', None)}
        except Exception as e:
            return {"error": str(e)}
