import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _safe_read_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def _normalize_cols(df, patterns):
    """Map column names that match any pattern to the canonical name."""
    if df is None:
        return {}
    mapping = {}
    for pat in patterns:
        for col in df.columns:
            if pat.lower() in col.lower():
                mapping[col] = pat
    return mapping

def load_macro():
    """Load vietnam_macro_2020_2025.csv with column auto-mapping."""
    df = _safe_read_csv("vietnam_macro_2020_2025.csv")
    if df is None:
        return None

    col_map = _normalize_cols(df, [
        "year", "GDP_trillion_VND", "GDP_billion_USD", "GDP_growth_pct",
        "GDP_per_capita_USD", "population_million", "agriculture_share_pct",
        "industry_share_pct", "services_share_pct", "taxes_share_pct",
        "agri_growth_pct", "industry_growth_pct", "services_growth_pct",
        "inflation_CPI_pct", "FDI_disbursed_billion_USD", "exports_billion_USD",
        "imports_billion_USD", "digital_economy_share_GDP_pct",
        "labor_productivity_million_VND",
    ])

    canon = {
        "year": "year",
        "GDP_trillion_VND": "GDP_trillion_VND",
        "GDP_billion_USD": "GDP_billion_USD",
        "GDP_growth_pct": "GDP_growth_pct",
        "GDP_per_capita_USD": "GDP_per_capita_USD",
        "population_million": "population_million",
        "agriculture_share_pct": "agriculture_share_pct",
        "industry_share_pct": "industry_share_pct",
        "services_share_pct": "services_share_pct",
        "taxes_share_pct": "taxes_share_pct",
        "agri_growth_pct": "agri_growth_pct",
        "industry_growth_pct": "industry_growth_pct",
        "services_growth_pct": "services_growth_pct",
        "inflation_CPI_pct": "inflation_CPI_pct",
        "FDI_disbursed_billion_USD": "FDI_disbursed_billion_USD",
        "exports_billion_USD": "exports_billion_USD",
        "imports_billion_USD": "imports_billion_USD",
        "digital_economy_share_GDP_pct": "digital_economy_share_GDP_pct",
        "labor_productivity_million_VND": "labor_productivity_million_VND",
    }

    for orig, canon_name in col_map.items():
        if orig != canon_name:
            df.rename(columns={orig: canon_name}, inplace=True)

    return df

def load_regions():
    """Load vietnam_regions_2024.csv with column auto-mapping."""
    df = _safe_read_csv("vietnam_regions_2024.csv")
    if df is None:
        return None

    col_map = _normalize_cols(df, [
        "region_id", "region_name_en", "population_million",
        "grdp_trillion_VND", "grdp_growth_pct", "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD", "exports_billion_USD",
        "digital_index_0_100", "ai_readiness_0_100",
        "trained_labor_pct", "gini_coef", "rd_intensity_pct",
        "internet_penetration_pct",
    ])

    canon = {
        "region_id": "region_id",
        "region_name_en": "region_name_en",
        "population_million": "population_million",
        "grdp_trillion_VND": "grdp_trillion_VND",
        "grdp_growth_pct": "grdp_growth_pct",
        "grdp_per_capita_million_VND": "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD": "fdi_registered_billion_USD",
        "exports_billion_USD": "exports_billion_USD",
        "digital_index_0_100": "digital_index_0_100",
        "ai_readiness_0_100": "ai_readiness_0_100",
        "trained_labor_pct": "trained_labor_pct",
        "gini_coef": "gini_coef",
        "rd_intensity_pct": "rd_intensity_pct",
        "internet_penetration_pct": "internet_penetration_pct",
    }

    for orig, canon_name in col_map.items():
        if orig != canon_name:
            df.rename(columns={orig: canon_name}, inplace=True)

    return df

def load_sectors():
    """Load vietnam_sectors_2024.csv with column auto-mapping."""
    df = _safe_read_csv("vietnam_sectors_2024.csv")
    if df is None:
        return None

    col_map = _normalize_cols(df, [
        "sector_id", "sector_name_en", "gdp_share_2024_pct",
        "growth_rate_2024_pct", "labor_million", "labor_share_pct",
        "export_billion_USD", "digital_index_0_100", "ai_readiness_0_100",
        "fdi_attraction_billion_USD", "spillover_coef_0_1",
        "automation_risk_pct", "rd_intensity_pct",
    ])

    canon = {
        "sector_id": "sector_id",
        "sector_name_en": "sector_name_en",
        "gdp_share_2024_pct": "gdp_share_2024_pct",
        "growth_rate_2024_pct": "growth_rate_2024_pct",
        "labor_million": "labor_million",
        "labor_share_pct": "labor_share_pct",
        "export_billion_USD": "export_billion_USD",
        "digital_index_0_100": "digital_index_0_100",
        "ai_readiness_0_100": "ai_readiness_0_100",
        "fdi_attraction_billion_USD": "fdi_attraction_billion_USD",
        "spillover_coef_0_1": "spillover_coef_0_1",
        "automation_risk_pct": "automation_risk_pct",
        "rd_intensity_pct": "rd_intensity_pct",
    }

    for orig, canon_name in col_map.items():
        if orig != canon_name:
            df.rename(columns={orig: canon_name}, inplace=True)

    return df

def get_dataset_info():
    info = {}
    df_m = load_macro()
    df_r = load_regions()
    df_s = load_sectors()
    info["macro"] = {
        "available": df_m is not None,
        "rows": len(df_m) if df_m is not None else 0,
        "cols": list(df_m.columns) if df_m is not None else [],
    }
    info["regions"] = {
        "available": df_r is not None,
        "rows": len(df_r) if df_r is not None else 0,
        "cols": list(df_r.columns) if df_r is not None else [],
    }
    info["sectors"] = {
        "available": df_s is not None,
        "rows": len(df_s) if df_s is not None else 0,
        "cols": list(df_s.columns) if df_s is not None else [],
    }
    return info
