from orchestrator import orchestrate_query
import json

try:
    print("Testing Orchestrator...")
    result = orchestrate_query("What projects have you worked on?")
    print("SUCCESS")
    print(json.dumps(result, indent=2))
except Exception as e:
    print("FAILED")
    import traceback
    traceback.print_exc()
