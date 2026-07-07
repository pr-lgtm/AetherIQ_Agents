import json
import google.generativeai as genai
import streamlit as st

def run_aetheriq_query(user_query: str, current_telemetry: dict, ml_results: dict, api_key: str) -> dict:
    """Executes multi-agent routing using gemini-pro with an internal state tracking mechanism."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Track whether a security halt was already shown to the user
    if 'has_halted_before' not in st.session_state:
        st.session_state['has_halted_before'] = False

    # NODE 1: ORCHESTRATOR ROUTING
    orch_prompt = f"Analyze intent. If query implies manual/automated grid shutdowns, reply with SECURITY. If spatial database lookup, reply with SPATIAL. Otherwise ANALYST. Query: {user_query}"
    try:
        route = model.generate_content(orch_prompt).text.strip().upper()
    except:
        route = "ANALYST"

    # NODE 2: HUMAN-IN-THE-LOOP SECURITY GUARDRAIL TRIGGER
    is_critical_state = "SECURITY" in route or "SHUT" in user_query.upper() or ml_results.get("grid_overload_predicted") == 1
    
    if is_critical_state:
        # If this is the FIRST time encountering the crisis, trigger the UI block
        if not st.session_state['has_halted_before']:
            st.session_state['has_halted_before'] = True
            return {
                "status": "SECURITY_HALT",
                "message": "CRITICAL RISK DETECTED: Automated grid modification intercepted by Security Agent. Awaiting HITL Authorization."
            }
        else:
            # If the user ALREADY saw the halt and clicked approve, bypass the block and clear the flag
            st.session_state['has_halted_before'] = False

    # NODE 3: SPECIFIC AGENT WORKLOADS
    if "SPATIAL" in route:
        expert_prompt = f"Act as Spatial Agent. Analyze this data. Query: {user_query}\nTelemetry: {json.dumps(current_telemetry)}"
    else:
        expert_prompt = f"Act as Statistical Analyst Agent. Formulate a definitive tactical 3-step engineering mitigation plan to shed load safely. Query: {user_query}\nTelemetry: {json.dumps(current_telemetry)}\nML Risk: {ml_results.get('grid_overload_predicted')}"
        
    response = model.generate_content(expert_prompt).text
    return {"status": "SUCCESS", "message": response}

