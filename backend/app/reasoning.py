"""
Multi-metric reasoning engine for Darukaa.Earth Biodiversity Intelligence Chatbot.
Loads the structured relationship graph and ranks interventions based on
multi-variable precondition matching.
"""

import json
import os

from backend.app.csv_knowledge import recalibrate_thresholds_from_crop_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_PATH = os.path.join(BASE_DIR, "knowledge", "graph.json")

_graph = None


def _load_graph() -> list[dict]:
    """Lazy-load the intervention relationship graph."""
    global _graph
    if _graph is None:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _graph = data.get("interventions", [])
    return _graph


def categorize_variables(user_variables: dict) -> list[str]:
    """
    Convert raw user variable values into categorical tags for matching
    against intervention preconditions.

    Empirically grounded bucketing rules (recalibrated from Crop_recommendation.csv):
    - soil_organic_carbon_pct: <0.5 -> low, 0.5–1.5 -> moderate, >1.5 -> high
    - soil_ph: empirical 25th & 75th percentiles (approx <5.97 acidic, 5.97-6.92 neutral, >6.92 alkaline)
    - rainfall: empirical 25th & 75th percentiles (approx <65mm low, 65-125mm moderate, >125mm high)
    - soil_moisture: low/dry/bone_dry -> low, moderate -> moderate, high/wet/flood -> high
    - land_use_type: mapped to specific tags
    - region_climate_zone: mapped to specific tags
    """
    tags = []
    thresholds = recalibrate_thresholds_from_crop_data()
    ph_low = thresholds["soil_ph"]["acidic_cutoff"]
    ph_high = thresholds["soil_ph"]["alkaline_cutoff"]
    rain_low = thresholds["rainfall_mm"]["low_cutoff"]
    rain_high = thresholds["rainfall_mm"]["high_cutoff"]

    # Soil organic carbon
    soc = user_variables.get("soil_organic_carbon_pct")
    if soc is not None:
        try:
            soc = float(soc)
            if soc < 0.5:
                tags.append("low_soil_organic_carbon")
            elif soc <= 1.5:
                tags.append("moderate_soil_organic_carbon")
            else:
                tags.append("high_soil_organic_carbon")
        except (ValueError, TypeError):
            pass

    # Soil pH — calibrated against empirical agricultural data
    ph = user_variables.get("soil_ph")
    if ph is not None:
        try:
            ph = float(ph)
            if ph < ph_low:
                tags.append("acidic_soil")
            elif ph <= ph_high:
                tags.append("neutral_soil")
            else:
                tags.append("alkaline_soil")
        except (ValueError, TypeError):
            pass

    # Soil moisture
    moisture = user_variables.get("soil_moisture")
    if moisture and isinstance(moisture, str):
        moisture_lower = moisture.lower().strip()
        if any(w in moisture_lower for w in ["low", "dry", "bone dry", "arid", "parched"]):
            tags.append("low_soil_moisture")
        elif any(w in moisture_lower for w in ["moderate", "medium", "normal"]):
            tags.append("moderate_soil_moisture")
        elif any(w in moisture_lower for w in ["high", "wet", "saturated", "flood", "waterlog"]):
            tags.append("high_soil_moisture")

    # Rainfall pattern
    rainfall = user_variables.get("rainfall_pattern")
    if rainfall and isinstance(rainfall, str):
        rainfall_lower = rainfall.lower().strip()
        if any(w in rainfall_lower for w in ["low", "scarce", "minimal", "dry", "drought", "rainfed", "unreliable"]):
            tags.extend(["low_rainfall", "any_rainfall"])
        elif any(w in rainfall_lower for w in ["moderate", "medium", "seasonal", "average"]):
            tags.extend(["moderate_rainfall", "any_rainfall"])
        elif any(w in rainfall_lower for w in ["high", "heavy", "abundant", "monsoon", "flood"]):
            tags.extend(["high_rainfall", "any_rainfall"])

    # Land use type
    land_use = user_variables.get("land_use_type")
    if land_use and isinstance(land_use, str):
        land_lower = land_use.lower().strip().replace(" ", "_")
        if "monoculture" in land_lower or "mono" in land_lower or "wheat" in land_lower or "rice" in land_lower or "maize" in land_lower:
            tags.append("monoculture")
        if "pastoral" in land_lower or "grazing" in land_lower or "livestock" in land_lower or "rangeland" in land_lower:
            tags.append("pastoral")
            if "degraded" in land_lower or "overgraze" in land_lower:
                tags.append("degraded_rangeland")
        if "plantation" in land_lower:
            tags.append("plantation")
        if "mixed" in land_lower or "diverse" in land_lower or "polyculture" in land_lower or "intercrop" in land_lower:
            tags.append("mixed_farming")
        if "forest" in land_lower or "deforest" in land_lower:
            tags.append("deforested")
        if "degrade" in land_lower or "barren" in land_lower or "wasteland" in land_lower or "red_soil" in land_lower:
            tags.append("degraded_land")
        if "tillage" in land_lower or "plough" in land_lower or "plow" in land_lower:
            tags.append("conventional_tillage")
        if "wetland" in land_lower or "marsh" in land_lower or "peat" in land_lower:
            tags.append("wetland_adjacent")
            if "drain" in land_lower:
                tags.append("drained_wetland")

    # Climate zone
    climate = user_variables.get("region_climate_zone")
    if climate and isinstance(climate, str):
        climate_lower = climate.lower().strip()
        if "arid" in climate_lower and "semi" not in climate_lower and "sub" not in climate_lower:
            tags.extend(["arid", "semi_arid"])
        elif "semi_arid" in climate_lower or "semi-arid" in climate_lower or "semiarid" in climate_lower or "thar" in climate_lower or "rajasthan" in climate_lower:
            tags.append("semi_arid")
        if "tropical" in climate_lower:
            tags.append("tropical")
        if "subtropical" in climate_lower or "sub-tropical" in climate_lower:
            tags.append("subtropical")
        if "temperate" in climate_lower:
            tags.append("temperate")
        if "boreal" in climate_lower:
            tags.append("boreal")

    # Species richness
    species = user_variables.get("species_richness_observation")
    if species and isinstance(species, str):
        species_lower = species.lower().strip()
        if any(w in species_lower for w in ["low", "poor", "minimal", "depleted", "decline", "dead"]):
            tags.extend(["low_species_richness", "pollinator_decline"])
        elif any(w in species_lower for w in ["moderate", "medium"]):
            tags.append("moderate_species_richness")

    # Pollution / deforestation
    pollution = user_variables.get("pollution_or_deforestation_pressure")
    if pollution and isinstance(pollution, str):
        pollution_lower = pollution.lower().strip()
        if any(w in pollution_lower for w in ["high", "severe", "heavy", "chemical", "pesticide"]):
            tags.extend(["high_pollution", "fragmented_habitat", "high_pesticide_use", "degraded_water_quality"])
        elif any(w in pollution_lower for w in ["moderate", "medium"]):
            tags.extend(["moderate_pollution", "fragmented_habitat"])
        if "deforest" in pollution_lower:
            tags.append("deforested")

    return list(set(tags))


def _map_preconditions_to_user_vars(matched_preconditions: list[str], user_variables: dict) -> list[str]:
    """Map matched precondition tags back to original user variable keys."""
    precondition_to_slot = {
        "low_soil_organic_carbon": "soil_organic_carbon_pct",
        "moderate_soil_organic_carbon": "soil_organic_carbon_pct",
        "high_soil_organic_carbon": "soil_organic_carbon_pct",
        "acidic_soil": "soil_ph",
        "neutral_soil": "soil_ph",
        "alkaline_soil": "soil_ph",
        "low_soil_moisture": "soil_moisture",
        "moderate_soil_moisture": "soil_moisture",
        "high_soil_moisture": "soil_moisture",
        "low_rainfall": "rainfall_pattern",
        "moderate_rainfall": "rainfall_pattern",
        "high_rainfall": "rainfall_pattern",
        "any_rainfall": "rainfall_pattern",
        "monoculture": "land_use_type",
        "pastoral": "land_use_type",
        "degraded_rangeland": "land_use_type",
        "plantation": "land_use_type",
        "mixed_farming": "land_use_type",
        "deforested": "land_use_type",
        "degraded_land": "land_use_type",
        "conventional_tillage": "land_use_type",
        "wetland_adjacent": "land_use_type",
        "drained_wetland": "land_use_type",
        "arid": "region_climate_zone",
        "semi_arid": "region_climate_zone",
        "tropical": "region_climate_zone",
        "subtropical": "region_climate_zone",
        "temperate": "region_climate_zone",
        "boreal": "region_climate_zone",
        "low_species_richness": "species_richness_observation",
        "moderate_species_richness": "species_richness_observation",
        "pollinator_decline": "species_richness_observation",
        "high_pollution": "pollution_or_deforestation_pressure",
        "moderate_pollution": "pollution_or_deforestation_pressure",
        "fragmented_habitat": "pollution_or_deforestation_pressure",
        "high_pesticide_use": "pollution_or_deforestation_pressure",
        "degraded_water_quality": "pollution_or_deforestation_pressure",
    }

    matched_slots = set()
    for p in matched_preconditions:
        slot = precondition_to_slot.get(p)
        if slot and user_variables.get(slot) is not None:
            matched_slots.add(slot)

    # Ensure all user-provided variables that actually match are returned
    if not matched_slots:
        matched_slots = {k for k, v in user_variables.items() if v is not None and k not in ["geo_lat", "geo_lon"]}

    return list(matched_slots)


def rank_interventions(user_variables: dict, min_overlap: int = 2) -> list[dict]:
    """
    Score and rank interventions from the relationship graph based on
    how many of their preconditions match the user's derived variable tags.

    Enforces a minimum overlap of `min_overlap` preconditions (default 2)
    to ensure genuine multi-metric reasoning.

    Returns:
        List of intervention dicts sorted by score descending, each augmented
        with 'score', 'overlap_count', 'matched_preconditions', 'matched_variables',
        'co_benefits', and 'trade_offs'.
    """
    graph = _load_graph()
    user_tags = categorize_variables(user_variables)

    if not user_tags:
        return []

    scored = []

    for intervention in graph:
        preconditions = intervention.get("preconditions", [])
        affects_metrics = intervention.get("affects_metrics", [])

        # Count precondition matches
        matched_preconditions = [p for p in preconditions if p in user_tags]
        precondition_score = len(matched_preconditions)

        # Enforce minimum overlap
        if precondition_score < min_overlap:
            continue

        # Map to matched user variables
        matched_vars = _map_preconditions_to_user_vars(matched_preconditions, user_variables)

        # Count metric overlap with user-provided variables
        user_provided_var_names = [k for k, v in user_variables.items() if v is not None]
        var_to_metric = {
            "soil_organic_carbon_pct": "soil_organic_carbon",
            "soil_ph": "soil_ph",
            "soil_moisture": "soil_moisture",
            "rainfall_pattern": "water_retention",
            "species_richness_observation": "species_richness",
        }
        user_metrics = {var_to_metric.get(v, v) for v in user_provided_var_names}
        metric_overlap = len(set(affects_metrics) & user_metrics)

        # Build result with scoring info
        scored_intervention = {
            **intervention,
            "score": precondition_score,
            "overlap_count": precondition_score,
            "metric_overlap": metric_overlap,
            "matched_preconditions": matched_preconditions,
            "matched_variables": matched_vars,
            "co_benefits": intervention.get("co_benefits", []),
            "trade_offs": intervention.get("trade_offs", [])
        }
        scored.append(scored_intervention)

    # Sort by precondition score (primary), then metric overlap (secondary)
    scored.sort(key=lambda x: (x["score"], x["metric_overlap"]), reverse=True)

    return scored


def get_graph_stats() -> dict:
    """Return basic stats about the relationship graph."""
    graph = _load_graph()
    return {
        "total_interventions": len(graph),
        "intervention_names": [i["name"] for i in graph]
    }
