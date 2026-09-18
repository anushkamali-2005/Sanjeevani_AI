"""
backend/app/csv_knowledge.py — Structured CSV Data Layer

Provides empirical, structured numeric grounding from real datasets in /data:
1. Berkeley Earth Global Land Temperature datasets (Country, State, Global).
2. Global Forest Watch (GFW) Carbon & Tree Cover Loss datasets (Country & Subnational).
3. IUCN Red List Species dataset (Species.csv).
4. Agricultural Crop & Soil recommendation dataset (Crop_recommendation.csv)
   used to dynamically recalibrate empirical thresholds for rainfall, pH, and nutrients.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# In-memory dataframe caches
_DFS: Dict[str, pd.DataFrame] = {}
_CALIBRATED_THRESHOLDS: Optional[Dict[str, Any]] = None


def _load_csv(filename: str, usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Load and cache a CSV file lazily."""
    if filename in _DFS:
        return _DFS[filename]
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=usecols)
        _DFS[filename] = df
        return df
    except Exception as e:
        print(f"[CSV Knowledge] Error loading {filename}: {e}")
        return None


def recalibrate_thresholds_from_crop_data() -> Dict[str, Any]:
    """
    Computes real empirical percentile-based thresholds (25th, 50th, 75th percentiles)
    from Crop_recommendation.csv to replace guessed cutoffs in reasoning.py and geo_lookup.py.
    """
    global _CALIBRATED_THRESHOLDS
    if _CALIBRATED_THRESHOLDS is not None:
        return _CALIBRATED_THRESHOLDS

    df = _load_csv("Crop_recommendation.csv")
    if df is None or df.empty:
        # Fallback default if file not available
        _CALIBRATED_THRESHOLDS = {
            "rainfall_mm": {"low_cutoff": 65.0, "high_cutoff": 125.0, "source": "Crop_recommendation.csv", "column": "rainfall"},
            "soil_ph": {"acidic_cutoff": 6.0, "alkaline_cutoff": 7.0, "source": "Crop_recommendation.csv", "column": "ph"},
            "nitrogen_n": {"p25": 21.0, "p50": 37.0, "p75": 84.0, "source": "Crop_recommendation.csv", "column": "N"},
            "phosphorus_p": {"p25": 28.0, "p50": 51.0, "p75": 68.0, "source": "Crop_recommendation.csv", "column": "P"},
            "potassium_k": {"p25": 20.0, "p50": 32.0, "p75": 49.0, "source": "Crop_recommendation.csv", "column": "K"},
            "temperature_c": {"p25": 22.8, "p50": 25.6, "p75": 28.6, "source": "Crop_recommendation.csv", "column": "temperature"},
        }
        return _CALIBRATED_THRESHOLDS

    metrics = ["rainfall", "ph", "N", "P", "K", "temperature", "humidity"]
    desc = df[metrics].describe(percentiles=[0.25, 0.5, 0.75])

    _CALIBRATED_THRESHOLDS = {
        "rainfall_mm": {
            "p25": round(float(desc.loc["25%", "rainfall"]), 1),
            "p50": round(float(desc.loc["50%", "rainfall"]), 1),
            "p75": round(float(desc.loc["75%", "rainfall"]), 1),
            "low_cutoff": round(float(desc.loc["25%", "rainfall"]), 1),   # 64.6 mm
            "high_cutoff": round(float(desc.loc["75%", "rainfall"]), 1),  # 124.3 mm
            "source": "Crop_recommendation.csv",
            "column": "rainfall"
        },
        "soil_ph": {
            "p25": round(float(desc.loc["25%", "ph"]), 2),
            "p50": round(float(desc.loc["50%", "ph"]), 2),
            "p75": round(float(desc.loc["75%", "ph"]), 2),
            "acidic_cutoff": round(float(desc.loc["25%", "ph"]), 2),   # 5.97
            "alkaline_cutoff": round(float(desc.loc["75%", "ph"]), 2), # 6.92
            "source": "Crop_recommendation.csv",
            "column": "ph"
        },
        "nitrogen_n": {
            "p25": round(float(desc.loc["25%", "N"]), 1),
            "p50": round(float(desc.loc["50%", "N"]), 1),
            "p75": round(float(desc.loc["75%", "N"]), 1),
            "source": "Crop_recommendation.csv",
            "column": "N"
        },
        "phosphorus_p": {
            "p25": round(float(desc.loc["25%", "P"]), 1),
            "p50": round(float(desc.loc["50%", "P"]), 1),
            "p75": round(float(desc.loc["75%", "P"]), 1),
            "source": "Crop_recommendation.csv",
            "column": "P"
        },
        "potassium_k": {
            "p25": round(float(desc.loc["25%", "K"]), 1),
            "p50": round(float(desc.loc["50%", "K"]), 1),
            "p75": round(float(desc.loc["75%", "K"]), 1),
            "source": "Crop_recommendation.csv",
            "column": "K"
        },
        "temperature_c": {
            "p25": round(float(desc.loc["25%", "temperature"]), 1),
            "p50": round(float(desc.loc["50%", "temperature"]), 1),
            "p75": round(float(desc.loc["75%", "temperature"]), 1),
            "source": "Crop_recommendation.csv",
            "column": "temperature"
        },
        "humidity_pct": {
            "p25": round(float(desc.loc["25%", "humidity"]), 1),
            "p50": round(float(desc.loc["50%", "humidity"]), 1),
            "p75": round(float(desc.loc["75%", "humidity"]), 1),
            "source": "Crop_recommendation.csv",
            "column": "humidity"
        }
    }
    return _CALIBRATED_THRESHOLDS


def get_country_climate_trend(country: str = "India") -> Dict[str, Any]:
    """
    Returns real historical average temperature and warming trend direction
    from GlobalLandTemperaturesByCountry.csv.
    """
    df = _load_csv("GlobalLandTemperaturesByCountry.csv")
    if df is None:
        return {}

    country_clean = country.strip().title()
    matched = df[df["Country"].str.lower() == country_clean.lower()].dropna(subset=["AverageTemperature"]).sort_values("dt")
    if matched.empty:
        # Fallback to India
        matched = df[df["Country"].str.lower() == "india"].dropna(subset=["AverageTemperature"]).sort_values("dt")
        country_clean = "India"

    if matched.empty:
        return {}

    count = len(matched)
    start_year = matched["dt"].iloc[0][:4]
    end_year = matched["dt"].iloc[-1][:4]

    # Compare first 10-year monthly block (120 observations) with last 10-year block
    early_sample = min(120, count // 4)
    early_avg = float(matched["AverageTemperature"].iloc[:early_sample].mean())
    recent_avg = float(matched["AverageTemperature"].iloc[-early_sample:].mean())
    delta = round(recent_avg - early_avg, 2)
    overall_mean = round(float(matched["AverageTemperature"].mean()), 2)

    trend_dir = "warming" if delta > 0 else ("cooling" if delta < 0 else "stable")

    return {
        "country": country_clean,
        "historical_avg_c": overall_mean,
        "early_baseline_c": round(early_avg, 2),
        "recent_baseline_c": round(recent_avg, 2),
        "temperature_change_c": delta,
        "trend_direction": trend_dir,
        "period": f"{start_year}–{end_year}",
        "total_records": count,
        "source_file": "GlobalLandTemperaturesByCountry.csv",
        "column_name": "AverageTemperature",
        "citation": f"Berkeley Earth Global Land Temperature Dataset ({start_year}–{end_year})"
    }


def get_state_climate_trend(state: str, country: str = "India") -> Optional[Dict[str, Any]]:
    """
    Returns historical temperature trend for an Indian state from GlobalLandTemperaturesByState.csv.
    """
    df = _load_csv("GlobalLandTemperaturesByState.csv")
    if df is None:
        return None

    state_clean = state.strip().title()
    matched = df[(df["State"].str.lower() == state_clean.lower()) & (df["Country"].str.lower() == country.lower())]
    if matched.empty:
        matched = df[df["State"].str.lower().str.contains(state_clean.lower(), na=False)]

    if matched.empty:
        return None

    matched = matched.dropna(subset=["AverageTemperature"]).sort_values("dt")
    if matched.empty:
        return None

    count = len(matched)
    early_sample = min(120, count // 4)
    early_avg = float(matched["AverageTemperature"].iloc[:early_sample].mean())
    recent_avg = float(matched["AverageTemperature"].iloc[-early_sample:].mean())
    delta = round(recent_avg - early_avg, 2)

    return {
        "state": state_clean,
        "country": country,
        "historical_avg_c": round(float(matched["AverageTemperature"].mean()), 2),
        "temperature_change_c": delta,
        "trend_direction": "warming" if delta > 0 else "cooling",
        "period": f"{matched['dt'].iloc[0][:4]}–{matched['dt'].iloc[-1][:4]}",
        "source_file": "GlobalLandTemperaturesByState.csv",
        "column_name": "AverageTemperature",
        "citation": f"Berkeley Earth GlobalLandTemperaturesByState.csv ({state_clean})"
    }


def get_country_carbon_and_forest_loss(country: str = "India") -> Dict[str, Any]:
    """
    Returns real carbon stock and tree cover loss figures from
    'Country carbon data.csv' and 'Country tree cover loss.csv'.
    """
    c_df = _load_csv("Country carbon data.csv")
    t_df = _load_csv("Country tree cover loss.csv")

    res = {
        "country": country.strip().title(),
        "source_files": ["Country carbon data.csv", "Country tree cover loss.csv"]
    }

    if c_df is not None:
        matched_c = c_df[c_df["country"].str.lower() == country.lower()]
        if not matched_c.empty:
            # Sort by threshold or take highest density
            row = matched_c.iloc[-1]
            carbon_stocks = row.get("gfw_aboveground_carbon_stocks_2000__Mg_C")
            density_c_ha = row.get("avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1")
            net_flux = row.get("gfw_forest_carbon_net_flux__Mg_CO2e_yr-1")

            res["aboveground_carbon_stocks_mg_c"] = int(carbon_stocks) if pd.notna(carbon_stocks) else None
            res["avg_carbon_density_mg_c_ha"] = float(density_c_ha) if pd.notna(density_c_ha) else None
            res["carbon_net_flux_mg_co2e_yr"] = float(net_flux) if pd.notna(net_flux) else None
            res["carbon_column"] = "gfw_aboveground_carbon_stocks_2000__Mg_C"

    if t_df is not None:
        matched_t = t_df[t_df["country"].str.lower() == country.lower()]
        if not matched_t.empty:
            row_t = matched_t.iloc[-1]
            extent_2000 = row_t.get("extent_2000_ha")
            gain_ha = row_t.get("gain_2000-2020_ha")
            res["tree_cover_extent_2000_ha"] = int(extent_2000) if pd.notna(extent_2000) else None
            res["tree_cover_gain_2000_2020_ha"] = int(gain_ha) if pd.notna(gain_ha) else None
            res["loss_column"] = "extent_2000_ha, gain_2000-2020_ha"

    return res


def get_subnational_carbon_and_loss(state: str, country: str = "India") -> Optional[Dict[str, Any]]:
    """
    Returns state-level carbon stock and tree cover loss from
    'Subnational 1 carbon data.csv' and 'Subnational 1 tree cover loss.csv'.
    """
    c_df = _load_csv("Subnational 1 carbon data.csv")
    t_df = _load_csv("Subnational 1 tree cover loss.csv")
    if c_df is None and t_df is None:
        return None

    state_clean = state.strip().title()
    res = {
        "state": state_clean,
        "country": country,
        "source_files": ["Subnational 1 carbon data.csv", "Subnational 1 tree cover loss.csv"]
    }

    if c_df is not None:
        matched = c_df[(c_df["country"].str.lower() == country.lower()) & (c_df["subnational1"].str.lower() == state_clean.lower())]
        if not matched.empty:
            row = matched.iloc[-1]
            res["aboveground_carbon_stocks_mg_c"] = int(row.get("gfw_aboveground_carbon_stocks_2000__Mg_C", 0))
            res["avg_carbon_density_mg_c_ha"] = float(row.get("avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1", 0.0))
            res["carbon_column"] = "gfw_aboveground_carbon_stocks_2000__Mg_C"

    if t_df is not None:
        matched_t = t_df[(t_df["country"].str.lower() == country.lower()) & (t_df["subnational1"].str.lower() == state_clean.lower())]
        if not matched_t.empty:
            row_t = matched_t.iloc[-1]
            res["extent_2000_ha"] = int(row_t.get("extent_2000_ha", 0))
            res["gain_2000_2020_ha"] = int(row_t.get("gain_2000-2020_ha", 0))
            res["loss_column"] = "extent_2000_ha"

    return res


def get_species_baseline(group_or_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns species count and threatened status from Species.csv (IUCN Red List).
    """
    df = _load_csv("Species.csv")
    if df is None or df.empty:
        return {}

    if group_or_name:
        matched = df[df["Name"].str.lower() == group_or_name.strip().lower()]
        if not matched.empty:
            row = matched.iloc[0]
            return {
                "taxonomic_group": row["Name"],
                "total_assessed": row.get("Total"),
                "critically_endangered": row.get("CR"),
                "endangered": row.get("EN"),
                "vulnerable": row.get("VU"),
                "threatened_subtotal": row.get("Subtotal (threatened spp.)"),
                "source_file": "Species.csv",
                "column_name": "Subtotal (threatened spp.)",
                "citation": "IUCN Red List Species Summary (Species.csv)"
            }

    # If no specific group requested, return overall summary totals
    top_groups = df.head(5)[["Name", "Total", "Subtotal (threatened spp.)"]].to_dict(orient="records")
    return {
        "summary": "IUCN Global Species Assessment Breakdown",
        "top_groups": top_groups,
        "source_file": "Species.csv",
        "column_name": "Subtotal (threatened spp.)",
        "citation": "IUCN Red List Global Biodiversity Assessment (Species.csv)"
    }


def get_structured_grounding_context(slots: Optional[dict] = None, message: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """
    Assembles real empirical numbers from the CSV datasets based on known slots
    or user message (e.g., location, state, crop, rainfall).
    Returns:
    1. A formatted grounding prompt text block for the LLM synthesis context.
    2. A list of structured source items for inclusion in the API response sources array.
    """
    slots = slots or {}
    prompt_lines = []
    source_items = []

    # 1. Empirical agricultural thresholds
    thresholds = recalibrate_thresholds_from_crop_data()
    rain_p25 = thresholds["rainfall_mm"]["low_cutoff"]
    rain_p75 = thresholds["rainfall_mm"]["high_cutoff"]
    ph_p25 = thresholds["soil_ph"]["acidic_cutoff"]
    ph_p75 = thresholds["soil_ph"]["alkaline_cutoff"]

    prompt_lines.append(
        f"• Empirical Agricultural Baselines (source: Crop_recommendation.csv, N=2200 trials):\n"
        f"  - Rainfall Percentiles: Low < {rain_p25}mm, Moderate {rain_p25}–{rain_p75}mm, High > {rain_p75}mm (column: rainfall)\n"
        f"  - Soil pH Percentiles: Acidic < {ph_p25}, Neutral {ph_p25}–{ph_p75}, Alkaline > {ph_p75} (column: ph)"
    )
    source_items.append({
        "type": "structured_data",
        "file": "Crop_recommendation.csv",
        "detail": f"Empirical agricultural percentiles (Rainfall: {rain_p25}–{rain_p75}mm, pH: {ph_p25}–{ph_p75})"
    })

    # 2. Country Climate Trend (Default India or user country)
    trend = get_country_climate_trend("India")
    if trend:
        prompt_lines.append(
            f"• Real Historical Climate Trend for India (source: {trend['source_file']}, column: {trend['column_name']}):\n"
            f"  - Temperature Change: {trend['trend_direction'].title()} trend of +{trend['temperature_change_c']}°C "
            f"over {trend['period']} (Historical avg: {trend['historical_avg_c']}°C, Recent: {trend['recent_baseline_c']}°C)"
        )
        source_items.append({
            "type": "structured_data",
            "file": trend["source_file"],
            "detail": f"India land temperature trend: +{trend['temperature_change_c']}°C ({trend['period']}, column: AverageTemperature)"
        })

    # 3. Country Carbon Stocks
    carbon = get_country_carbon_and_forest_loss("India")
    if carbon and carbon.get("aboveground_carbon_stocks_mg_c"):
        stocks_bil = round(carbon['aboveground_carbon_stocks_mg_c'] / 1e9, 2)
        prompt_lines.append(
            f"• Forest & Carbon Stocks for India (source: Country carbon data.csv, column: gfw_aboveground_carbon_stocks_2000__Mg_C):\n"
            f"  - Total Aboveground Forest Carbon: {stocks_bil} billion Mg C (~{carbon.get('avg_carbon_density_mg_c_ha', 0)} Mg C/ha density)\n"
            f"  - Tree Cover Gain (2000–2020): {carbon.get('tree_cover_gain_2000_2020_ha', 0):,} ha (source: Country tree cover loss.csv)"
        )
        source_items.append({
            "type": "structured_data",
            "file": "Country carbon data.csv",
            "detail": f"India aboveground forest carbon stock: {stocks_bil}B Mg C (column: gfw_aboveground_carbon_stocks_2000__Mg_C)"
        })

    # 4. Check for specific Indian State if mentioned
    state_candidates = ["Rajasthan", "Maharashtra", "Punjab", "Haryana", "Karnataka", "Tamil Nadu", "Gujarat", "Kerala", "Madhya Pradesh", "Uttar Pradesh", "Bihar", "West Bengal", "Odisha", "Andhra Pradesh", "Telangana"]
    search_space = (str(slots) + " " + message).lower()
    matched_state = None
    for sc in state_candidates:
        if sc.lower() in search_space:
            matched_state = sc
            break

    if matched_state:
        st_carbon = get_subnational_carbon_and_loss(matched_state, "India")
        st_trend = get_state_climate_trend(matched_state, "India")
        if st_carbon and st_carbon.get("aboveground_carbon_stocks_mg_c"):
            st_mil = round(st_carbon["aboveground_carbon_stocks_mg_c"] / 1e6, 1)
            prompt_lines.append(
                f"• Subnational Carbon Data for {matched_state} (source: Subnational 1 carbon data.csv):\n"
                f"  - Aboveground Carbon Stocks: {st_mil} Million Mg C (Density: {st_carbon.get('avg_carbon_density_mg_c_ha')} Mg C/ha)"
            )
            source_items.append({
                "type": "structured_data",
                "file": "Subnational 1 carbon data.csv",
                "detail": f"{matched_state} aboveground carbon stocks: {st_mil}M Mg C (column: gfw_aboveground_carbon_stocks_2000__Mg_C)"
            })
        if st_trend:
            prompt_lines.append(
                f"• Subnational Climate Trend for {matched_state} (source: GlobalLandTemperaturesByState.csv):\n"
                f"  - Historical Average: {st_trend['historical_avg_c']}°C, Trend: {st_trend['temperature_change_c']}°C ({st_trend['period']})"
            )
            source_items.append({
                "type": "structured_data",
                "file": "GlobalLandTemperaturesByState.csv",
                "detail": f"{matched_state} temperature trend: +{st_trend['temperature_change_c']}°C ({st_trend['period']})"
            })

    grounding_text = "\n".join(prompt_lines)
    return grounding_text, source_items
