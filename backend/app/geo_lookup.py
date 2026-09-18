"""
Geo-coordinate lookup for Darukaa.Earth Biodiversity Intelligence Chatbot.
Maps latitude/longitude to climate zone and rainfall pattern using
a static Köppen-Geiger classification lookup table.
"""


# Simplified Köppen-Geiger climate classification based on lat/lon ranges
# This is a static approximation — a production system would use a proper
# climate raster dataset or API like Open-Meteo.

CLIMATE_ZONES = [
    # (lat_min, lat_max, lon_min, lon_max, climate_zone, rainfall_pattern)
    # South Asia
    (8, 15, 68, 88, "tropical", "high"),
    (15, 23.5, 68, 88, "subtropical", "moderate"),
    (23.5, 28, 68, 78, "semi_arid", "low"),
    (23.5, 28, 78, 92, "subtropical", "moderate"),
    (28, 35, 68, 80, "semi_arid", "low"),
    (28, 37, 72, 80, "temperate", "moderate"),

    # Sub-Saharan Africa — Sahel
    (10, 20, -18, 40, "semi_arid", "low"),
    (0, 10, -18, 40, "tropical", "high"),
    (-10, 0, 20, 45, "tropical", "high"),
    (-35, -10, 15, 40, "subtropical", "moderate"),

    # Europe — temperate
    (35, 55, -10, 30, "temperate", "moderate"),
    (55, 70, -10, 30, "boreal", "moderate"),

    # North America
    (25, 40, -125, -70, "temperate", "moderate"),
    (40, 55, -125, -70, "temperate", "moderate"),
    (55, 70, -170, -50, "boreal", "low"),
    (15, 25, -110, -85, "semi_arid", "low"),

    # South America
    (-10, 10, -80, -35, "tropical", "high"),
    (-35, -10, -75, -35, "subtropical", "moderate"),
    (-55, -35, -75, -60, "temperate", "moderate"),

    # Australia
    (-40, -30, 115, 155, "temperate", "moderate"),
    (-30, -20, 115, 155, "semi_arid", "low"),
    (-20, -10, 115, 155, "tropical", "high"),

    # Middle East
    (15, 35, 35, 60, "arid", "low"),

    # Central Asia
    (35, 50, 50, 90, "semi_arid", "low"),

    # Southeast Asia
    (-10, 20, 95, 140, "tropical", "high"),

    # East Asia
    (20, 40, 100, 145, "subtropical", "moderate"),
    (40, 55, 100, 145, "temperate", "moderate"),
]


def lookup_climate(lat: float, lon: float) -> dict:
    """
    Look up the climate zone and rainfall pattern for given coordinates.

    Returns:
        dict with 'region_climate_zone' and 'rainfall_pattern', or empty values
        if no match is found.
    """
    for lat_min, lat_max, lon_min, lon_max, climate, rainfall in CLIMATE_ZONES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return {
                "region_climate_zone": climate,
                "rainfall_pattern": rainfall
            }

    # Default fallback based on latitude bands
    abs_lat = abs(lat)
    if abs_lat < 10:
        return {"region_climate_zone": "tropical", "rainfall_pattern": "high"}
    elif abs_lat < 23.5:
        return {"region_climate_zone": "subtropical", "rainfall_pattern": "moderate"}
    elif abs_lat < 35:
        return {"region_climate_zone": "semi_arid", "rainfall_pattern": "low"}
    elif abs_lat < 55:
        return {"region_climate_zone": "temperate", "rainfall_pattern": "moderate"}
    else:
        return {"region_climate_zone": "boreal", "rainfall_pattern": "low"}


def classify_rainfall_mm(rainfall_mm: float) -> str:
    """
    Classify numeric rainfall in mm into low/moderate/high using
    empirical percentiles from Crop_recommendation.csv.
    """
    from backend.app.csv_knowledge import recalibrate_thresholds_from_crop_data
    t = recalibrate_thresholds_from_crop_data()
    low = t["rainfall_mm"]["low_cutoff"]
    high = t["rainfall_mm"]["high_cutoff"]
    if rainfall_mm < low:
        return "low"
    elif rainfall_mm <= high:
        return "moderate"
    else:
        return "high"


def classify_soil_ph(ph: float) -> str:
    """
    Classify soil pH into acidic/neutral/alkaline using
    empirical percentiles from Crop_recommendation.csv.
    """
    from backend.app.csv_knowledge import recalibrate_thresholds_from_crop_data
    t = recalibrate_thresholds_from_crop_data()
    acidic = t["soil_ph"]["acidic_cutoff"]
    alkaline = t["soil_ph"]["alkaline_cutoff"]
    if ph < acidic:
        return "acidic"
    elif ph <= alkaline:
        return "neutral"
    else:
        return "alkaline"
