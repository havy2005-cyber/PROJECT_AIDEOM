import numpy as np
import pandas as pd

def safe_float(val, default=0.0):
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default

def safe_int(val, default=0):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default

def mean_abs_pct_error(y_true, y_pred):
    """MAPE - handles zero values."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def normalize_minmax(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

def normalize_zscore(series):
    mn, mx = series.mean(), series.std()
    if mx == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mn) / mx

def weighted_score(df, criteria_weights):
    """Compute weighted sum score."""
    scores = pd.Series(0.0, index=df.index)
    for col, w in criteria_weights.items():
        if col in df.columns:
            scores += w * normalize_minmax(df[col])
    return scores

def topsis_matrix(df, criteria_cols, benefit_criteria, weights):
    """Compute TOPSIS scores and return DataFrame with scores."""
    mat = df[criteria_cols].copy().astype(float)
    for col in criteria_cols:
        if col not in benefit_criteria:
            mat[col] = -mat[col]

    norm = mat.div(np.sqrt((mat ** 2).sum(axis=0)), axis=1)
    weighted = norm * weights

    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)

    dist_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))

    closeness = dist_neg / (dist_pos + dist_neg)
    return closeness

def format_trillion(val):
    return f"{val:,.1f} T"

def format_billion(val):
    return f"{val:,.1f} B"

def format_percent(val):
    return f"{val:.2f}%"

def format_number(val, decimals=2):
    return f"{val:,.{decimals}f}"

def format_vnd(val):
    if val >= 1_000_000:
        return f"{val/1_000_000:,.1f}T VND"
    elif val >= 1_000:
        return f"{val/1_000:,.0f}M VND"
    return f"{val:,.0f} VND"

def currency_fmt(val):
    return f"${val:,.1f}B"

def generate_scenarios(base_values, deviations):
    """Generate 3 scenarios: pessimistic, base, optimistic."""
    return {
        "Bi quan (S1)": {k: v * (1 - d) for k, v, d in zip(base_values.keys(), base_values.values(), deviations)},
        "Co so (S2)": base_values,
        "Tuong lai (S3)": {k: v * (1 + d) for k, v, d in zip(base_values.keys(), base_values.values(), deviations)},
    }

def monte_carlo(func, params_dist, n_simulations=1000, seed=42):
    """Run Monte Carlo simulation."""
    np.random.seed(seed)
    results = []
    for _ in range(n_simulations):
        sample = {k: np.random.normal(mu, sigma) for k, (mu, sigma) in params_dist.items()}
        results.append(func(sample))
    return np.array(results)

def discretize_state(value, n_bins=5):
    """Discretize continuous value into state bins."""
    return min(int(value * n_bins), n_bins - 1)

def moving_average(series, window=3):
    return pd.Series(series).rolling(window=window, min_periods=1).mean()

def cagr(start, end, periods):
    if start <= 0 or periods <= 0:
        return 0.0
    return ((end / start) ** (1 / periods) - 1) * 100

def growth_rate(prev, curr):
    if prev == 0 or np.isnan(prev) or np.isnan(curr):
        return 0.0
    return ((curr - prev) / prev) * 100

def vietnamese_region_name(name_en):
    mapping = {
        "Northern Midlands and Mountains": "Trung du va mien nui phia Bac",
        "Red River Delta": "Dong Bang Song Hong",
        "North Central and South Central Coast": "Bac Trung Bo va duyen hai Trung Bo",
        "Central Highlands": "Tay Nguyen",
        "Southeast": "Dong Nam Bo",
        "Mekong Delta": "Dong Bang Song Cuu Long",
    }
    return mapping.get(name_en, name_en)

def vietnamese_sector_name(name_en):
    mapping = {
        "Agriculture-Forestry-Fishery": "Nong nghiep - Lam nghiep - Thuy san",
        "Manufacturing": "Cong nghiep - Che bien",
        "Construction": "Xay dung",
        "Mining": "Khai khoang",
        "Wholesale-Retail": "Ban buon - Ban le",
        "Finance-Banking-Insurance": "Tai chinh - Ngan hang - Bao hiem",
        "Logistics-Transport-Warehousing": "Logistics - Van tai - Kho tang",
        "Information-Communication-IT": "Thong tin - Truyen thong - CNTT",
        "Education-Training": "Giao duc - Dao tao",
        "Healthcare": "Y te - Suc khoe",
    }
    return mapping.get(name_en, name_en)


def hex_to_rgba(hex_color, alpha=0.15):
    """Convert hex color (with or without #) to rgba string. e.g. '#00D4FF' + 0.15 -> 'rgba(0,212,255,0.15)'."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color
