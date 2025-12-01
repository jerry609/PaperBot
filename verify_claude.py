import asyncio
import logging
from agents.research_agent import ResearchAgent
from agents.code_analysis_agent import CodeAnalysisAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_agents():
    print("--- Testing ResearchAgent ---")
    research_agent = ResearchAgent(config={'anthropic_api_key': 'dummy_key'})
    
    # Check if client is initialized (it should be with dummy key)
    if research_agent.client:
        print("ResearchAgent: Client initialized successfully.")
    else:
        print("ResearchAgent: Client NOT initialized.")

    # Test ask_claude (will fail auth but prove method exists and tries to call)
    response = await research_agent.ask_claude("Hello", max_tokens=10)
    print(f"ResearchAgent ask_claude response: {response}")

    print("\n--- Testing CodeAnalysisAgent ---")
    code_agent = CodeAnalysisAgent(config={}) # No key
    if not code_agent.client:
        print("CodeAnalysisAgent: Client correctly not initialized (no key).")
    
    response = await code_agent.explain_code_security("print('hello')")
    print(f"CodeAnalysisAgent explain_code_security response: {response}")

if __name__ == "__main__":
    asyncio.run(test_agents())
