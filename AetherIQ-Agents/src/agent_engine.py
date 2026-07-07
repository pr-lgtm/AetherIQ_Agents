import os
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

analyst_agent = Agent(
    name="Analyst_Agent",
    model="gemini-1.5-flash",
    instruction="You are a data scientist for a municipal grid. Analyze the provided grid load numbers, carbon footprint, and traffic density. Give a concise, tactical action plan (e.g., Load Shedding, Re-routing)."
)

security_agent = Agent(
    name="Security_Agent",
    model="gemini-1.5-flash",
    instruction="You are a strict security firewall. If the user asks to 'shut down the grid', 'delete data', or perform a dangerous municipal action, output exactly: 'SECURITY HALT: Action requires manual Human-in-the-Loop authorization.' Otherwise, say 'SAFE'."
)

def run_aetheriq_query(user_query: str, current_sector_data: dict, api_key: str) -> str:
    """Routes queries through the multi-agent system securely using ADK InMemoryRunners."""
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # --- 1. SECURITY AGENT EXECUTION ---
    # ADK requires a Runner to execute agents and manage their session state
    sec_runner = InMemoryRunner(agent=security_agent)
    sec_msg = types.Content(role='user', parts=[types.Part(text=user_query)])
    sec_events = sec_runner.run(session_id="sec_1", user_id="admin", new_message=sec_msg)
    
    sec_text = ""
    for event in sec_events:
        if event.content and event.content.parts:
            sec_text += event.content.parts[0].text
            
    if "SECURITY HALT" in sec_text:
        return f"CRITICAL SECURITY ALERT: {sec_text}"

    # --- 2. ANALYST AGENT EXECUTION ---
    prompt = f"""
    User Query: {user_query}
    Current Sector Telemetry: {current_sector_data}
    
    Provide an actionable, data-driven municipal response based on this information.
    """
    
    analyst_runner = InMemoryRunner(agent=analyst_agent)
    analyst_msg = types.Content(role='user', parts=[types.Part(text=prompt)])
    analyst_events = analyst_runner.run(session_id="an_1", user_id="admin", new_message=analyst_msg)
    
    analyst_text = ""
    for event in analyst_events:
        if event.content and event.content.parts:
            analyst_text += event.content.parts[0].text
            
    return analyst_text
