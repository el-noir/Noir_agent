import asyncio
from graph import create_portfolio_graph
from langchain_core.messages import HumanMessage

async def test_portfolio_query():
    print("\n--- Testing Portfolio Query ---")
    app = create_portfolio_graph()
    state = {"messages": [HumanMessage(content="What are your top React projects?")]}
    config = {"configurable": {"thread_id": "test_1"}}
    
    result = await app.ainvoke(state, config=config)
    print(f"AI Response: {result['messages'][-1].content}")
    print(f"Trace: {result['trace']}")

async def test_calendar_query():
    print("\n--- Testing Calendar Query ---")
    app = create_portfolio_graph()
    state = {"messages": [HumanMessage(content="What am I doing today?")]}
    config = {"configurable": {"thread_id": "test_2"}}
    
    result = await app.ainvoke(state, config=config)
    print(f"AI Response: {result['messages'][-1].content}")
    print(f"Trace: {result['trace']}")

if __name__ == "__main__":
    asyncio.run(test_portfolio_query())
    asyncio.run(test_calendar_query())
