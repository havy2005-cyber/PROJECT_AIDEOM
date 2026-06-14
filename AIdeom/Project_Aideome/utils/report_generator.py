import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

def generate_text_report(title, sections):
    """Generate a structured text report."""
    lines = []
    lines.append("=" * 60)
    lines.append(title.center(60))
    lines.append(f"Ngay tao: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append("=" * 60)
    for section_title, section_content in sections:
        lines.append("")
        lines.append(f"--- {section_title} ---")
        if isinstance(section_content, pd.DataFrame):
            lines.append(section_content.to_string(index=False))
        elif isinstance(section_content, list):
            for item in section_content:
                lines.append(f"  - {item}")
        else:
            lines.append(str(section_content))
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)

def dataframe_to_html(df):
    """Convert DataFrame to styled HTML table string."""
    try:
        return df.to_html(index=False, escape=False, classes="dataframe")
    except Exception:
        return str(df)

def format_policy_recommendation(algorithm_name, results):
    """Format policy recommendation text from algorithm results."""
    lines = []
    lines.append(f"**Khuyen nghi chinh sach tu {algorithm_name}:**")
    if isinstance(results, dict):
        for key, val in results.items():
            lines.append(f"- {key}: {val}")
    elif isinstance(results, list):
        for item in results:
            lines.append(f"- {item}")
    return "\n".join(lines)

def summary_stats(df, numeric_only=True):
    """Generate summary statistics for a DataFrame."""
    if numeric_only:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df
    desc = numeric_df.describe().T
    desc["missing"] = df.isnull().sum()
    return desc

def correlation_insights(df, threshold=0.5):
    """Find strongly correlated pairs."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "var1": corr.columns[i],
                    "var2": corr.columns[j],
                    "correlation": val,
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

def export_results_csv(results_dict, filename="results.csv"):
    """Export results dict to CSV string."""
    rows = []
    for key, val in results_dict.items():
        if isinstance(val, (list, pd.Series)):
            for i, v in enumerate(val):
                rows.append({"Index": i, "Key": key, "Value": v})
        else:
            rows.append({"Index": 0, "Key": key, "Value": val})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def format_coefficients(coef_dict):
    """Format coefficient dictionary for display."""
    lines = []
    for name, val in coef_dict.items():
        sign = "+" if val >= 0 else ""
        lines.append(f"  {name}: {sign}{val:.4f}")
    return "\n".join(lines)

def latex_table(df, caption=""):
    """Convert DataFrame to LaTeX table string."""
    tex = []
    tex.append(f"\\begin{{table}}[h]")
    tex.append("\\centering")
    tex.append(f"\\caption{{{caption}}}")
    tex.append(f"\\begin{{tabular}}{{{'|'.join(['c'] * (len(df.columns) + 1))}|}}")
    tex.append("\\hline")
    tex.append(" & ".join([""] + [str(c) for c in df.columns]) + " \\\\")
    tex.append("\\hline")
    for _, row in df.iterrows():
        tex.append(" & ".join([str(v) for v in row]) + " \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    return "\n".join(tex)
