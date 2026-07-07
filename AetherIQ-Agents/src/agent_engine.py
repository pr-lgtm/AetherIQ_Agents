import os
from google.adk import Agent

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
    """Routes queries through the multi-agent system securely."""
    os.environ["GOOGLE_API_KEY"] = api_key
    
    security_check = security_agent.run(user_query)
    if "SECURITY HALT" in security_check.text:
        return f"CRITICAL SECURITY ALERT: {security_check.text}"

    prompt = f"""
    User Query: {user_query}
    Current Sector Telemetry: {current_sector_data}
    
    Provide an actionable, data-driven municipal response based on this information.
    """
    
    response = analyst_agent.run(prompt)
    return response.text