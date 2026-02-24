import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_mcp():
    creds_path = os.path.abspath("credentials.json")
    print(f"Testing MCP with creds: {creds_path}")
    
    client = MultiServerMCPClient(
        {
            "google-calendar": {
                "command": "npx.cmd",
                "args": ["-y", "@cocal/google-calendar-mcp"],
                "transport": "stdio",
                "env": {
                    "GOOGLE_OAUTH_CREDENTIALS": creds_path,
                    "PATH": os.environ["PATH"]
                }
            }
        }
    )
    
    try:
        print("Fetching tools...")
        tools = await asyncio.wait_for(client.get_tools(), timeout=60.0)
        print(f"Success! Found {len(tools)} tools.")
        for t in tools:
            print(f" - {t.name}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp())
