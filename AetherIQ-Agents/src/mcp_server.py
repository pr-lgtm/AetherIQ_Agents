from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AetherIQ_Spatial_Data")

@mcp.tool()
def get_sector_metrics(sector_name: str) -> dict:
    """Fetch real-time urban telemetry and climate metrics for a specific city sector."""
    database = {
        "Downtown Core": {"traffic_density": 0.85, "ambient_temp": 38.5, "grid_load_mwh": 112.0, "aqi": 185},
        "Industrial District": {"traffic_density": 0.92, "ambient_temp": 40.1, "grid_load_mwh": 145.5, "aqi": 210},
        "Residential Suburbs": {"traffic_density": 0.45, "ambient_temp": 32.0, "grid_load_mwh": 65.2, "aqi": 95},
        "Green Belt Park": {"traffic_density": 0.20, "ambient_temp": 28.0, "grid_load_mwh": 15.0, "aqi": 45},
        "Commercial Hub": {"traffic_density": 0.75, "ambient_temp": 35.0, "grid_load_mwh": 95.0, "aqi": 140}
    }
    return database.get(sector_name, {"error": "Sector not found in municipal database."})

if __name__ == "__main__":
    print("AetherIQ MCP Server running on stdio...")
    mcp.run()