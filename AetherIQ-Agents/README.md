# AetherIQ: Geo-Spatial Decision Intelligence Platform for Urban Climate Resilience

AetherIQ is a multi-agent Decision Intelligence Platform engineered for municipal sustainability, environmental risk monitoring, and smart utility grid management. Built for the **Kaggle x Google 5-Day AI Agents Intensive Capstone**, the platform utilizes Google's Agent Development Kit (ADK 2.0) to process multivariate urban telemetry streams, execute predictive machine learning pipelines, and expose critical data insights through an autonomous, context-aware generative AI workflow.

**Primary Capstone Track:** 🌍 Agents for Good

## 1. The Core Problem Solved by AetherIQ
Modern smart cities are filled with IoT sensory networks capturing continuous environmental metrics such as traffic flows, electricity consumption, ambient temperatures, and air quality indexes. However, municipal stakeholders face a critical data bottleneck: these telemetry streams exist in isolated databases without integrated context. 

Because cities cannot parse these complex, intersecting variables in real time, they fail to achieve proactive operational awareness. This leads to unmanaged urban heat island effects, unexpected municipal utility grid failures during localized heatwaves, and an inability to map how traffic gridlock correlates to sudden atmospheric carbon pooling.

## 2. How AetherIQ Solves It (The Multi-Agent Approach)
AetherIQ bridges the gap between raw time-series data and tactical human decision-making by moving away from standard static dashboards and implementing a dynamic **Multi-Agent Architecture**:
* **The Orchestrator Agent:** Acts as the traffic controller, taking natural language queries from city planners and intelligently routing them to specialized sub-agents based on the required analytical task.
* **The Statistical Analyst Agent:** Equipped with tools to execute our trained Machine Learning models to compute structural infrastructure failure risks (grid overloads) and future localized environmental degradation.
* **The Geospatial Data Agent:** Interfaces securely with live city databases to pull real-time spatial records across diverse urban zones to answer contextual queries.

## 3. Capstone Technical Requirements Demonstrated
To achieve this autonomous functionality, AetherIQ implements the following core concepts from the AI Intensive course:

* **Agent / Multi-Agent System (ADK 2.0):** The core logic is built using a structured ADK 2.0 graph workflow. This ensures reliable execution logic where the LLM is guided through deterministic pathways to prevent hallucination during critical grid analysis.
* **Model Context Protocol (FastMCP) Server:** Instead of loading massive, static telemetry CSVs into the agent's context window, AetherIQ uses an MCP server. The Geospatial Data Agent queries this server dynamically to fetch exact climate variables or spatial coordinates on-demand.
* **Human-in-the-Loop Security Guardrails:** Because utility grids are critical infrastructure, AetherIQ implements a strict security node. If an agent determines a query requires executing a computationally expensive prediction or modifying a grid parameter, the workflow pauses and requests explicit Human-in-the-Loop confirmation via the UI before proceeding.
* **Agents CLI & Scaffold:** The project architecture was rapidly prototyped and initialized using the `uvx google-agents-cli setup` toolsets.

## 4. Programming Logic and Pipeline Architecture
The mathematical and predictive foundations of AetherIQ rely on robust data engineering:

* **Trigonometric Cyclic Time Encoding:** To ensure the ML tools understand temporal continuity (such as the chronological link between 11:59 PM and 12:00 AM), raw hours are mapped into continuous geometric space using:
  $$Hour_{sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
  $$Hour_{cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
* **Calibrated Grid Overload Classifier:** The Analyst Agent utilizes a logistic response curve to predict system failure risks and trigger predictive utility alerts:
  $$P(\text{Overload}) = \frac{1}{1 + e^{-\text{log-odds}}}$$

## 5. How to Run the Platform in GitHub Codespaces
Follow these exact steps inside your GitHub Codespaces terminal to deploy the application locally:

### Step 1: Initialize the Environment
Ensure your environment is isolated and install the required ADK dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install google-adk mcp fastmcp