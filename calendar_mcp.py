import os
import asyncio
from typing import List, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

async def get_calendar_tools() -> List[BaseTool]:
    """
    Connects to the Google Calendar MCP server and retrieves available tools.
    """
    print("Connecting to Google Calendar MCP Server...")
    
    # Path to credentials.json in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(script_dir, "credentials.json")
    
    if not os.path.exists(creds_path):
        print(f"Warning: {creds_path} not found. Calendar tools will not be available.")
        return []

    # Path to local node_modules entry point
    node_entry = os.path.join(script_dir, "node_modules", "@cocal", "google-calendar-mcp", "build", "index.js")
    
    if not os.path.exists(node_entry):
        print(f"Warning: {node_entry} not found. Falling back to npx.")
        command = "npx.cmd"
        args = ["-y", "@cocal/google-calendar-mcp"]
    else:
        command = "node"
        args = [node_entry]

    client = MultiServerMCPClient(
        {
            "google-calendar": {
                "command": command,
                "args": args,
                "transport": "stdio",
                "env": {
                    "GOOGLE_OAUTH_CREDENTIALS": creds_path
                }
            }
        }
    )
    
    try:
        # MultiServerMCPClient.get_tools() returns a list of tools
        print(f"Requesting tools from MCP server at {creds_path}...")
        tools = await asyncio.wait_for(client.get_tools(), timeout=60.0)
        
        # Filter tools to keep JSON schema small and avoid Groq 413 Token Limits
        allowed_tools = {"list-calendars", "list-events", "create-event", "update-event", "delete-event"}
        filtered_tools = [t for t in tools if t.name in allowed_tools]
        
        print(f"Successfully loaded {len(filtered_tools)} tools from Google Calendar MCP (down from {len(tools)}).")
        return filtered_tools
    except asyncio.TimeoutError:
        print("Error: MCP connection timed out after 60 seconds.")
        return []
    except Exception as e:
        import traceback
        print(f"Error connecting to MCP: {str(e)}")
        traceback.print_exc()
        return []

# Tools are retrieved asynchronously via get_calendar_tools()

