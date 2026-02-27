import os
import asyncio
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

async def get_meet_tools() -> List[BaseTool]:
    """
    Connects to the Google Meet MCP server and retrieves available tools.
    """
    print("Connecting to Google Meet MCP Server...")
    
    # Path to credentials.json in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(script_dir, "credentials.json")
    
    if not os.path.exists(creds_path):
        print(f"Warning: {creds_path} not found. Meet tools will not be available.")
        return []

    # Path to local node_modules entry point (since we cloned it locally)
    node_entry = os.path.join(script_dir, "google-meet-mcp", "src", "index.js")
    
    if not os.path.exists(node_entry):
        print(f"Warning: {node_entry} not found.")
        return []

    command = "node"
    args = [node_entry]

    # Resolve token path exactly as google-calendar-mcp does
    token_path = os.path.join(os.path.expanduser("~"), ".config", "google-calendar-mcp", "tokens.json")

    client = MultiServerMCPClient(
        {
            "google-meet": {
                "command": command,
                "args": args,
                "transport": "stdio",
                "env": {
                    "GOOGLE_MEET_CREDENTIALS_PATH": creds_path,
                    "GOOGLE_MEET_TOKEN_PATH": token_path
                }
            }
        }
    )
    
    try:
        print(f"Requesting tools from Meet MCP server at {node_entry}...")
        tools = await asyncio.wait_for(client.get_tools(), timeout=60.0)
        
        allowed_tools = {"create_meeting", "list_meetings", "get_meeting"}
        filtered_tools = [t for t in tools if t.name in allowed_tools]
        
        print(f"Successfully loaded {len(filtered_tools)} tools from Google Meet MCP (down from {len(tools)}).")
        return filtered_tools
    except asyncio.TimeoutError:
        print("Error: Meet MCP connection timed out after 60 seconds.")
        return []
    except Exception as e:
        import traceback
        print(f"Error connecting to Meet MCP: {str(e)}")
        traceback.print_exc()
        return []

# Tools are retrieved asynchronously via get_meet_tools()
