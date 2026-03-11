from typing import List
from langchain_core.tools import BaseTool

async def get_calendar_tools() -> List[BaseTool]:
    """
    Previously used the @cocal/google-calendar-mcp Node.js MCP server.
    Calendar read tools (list-calendars, list-events) are no longer needed —
    conflict checking and booking are handled entirely by book_meeting_tool (pure Python).
    """
    return []

# Tools are retrieved asynchronously via get_calendar_tools()

