import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, PLOTLY_TEMPLATE

def _hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color

def _fig_layout(fig, title=None, height=400, xlabel=None, ylabel=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF", "family": "Arial"},
        xaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title=xlabel or ""),
        yaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title=ylabel or ""),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font={"color": "#B0C0E0"},
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}))
    return fig

def line_chart(x, y_dict, title=None, xlabel=None, ylabel=None, height=400):
    fig = go.Figure()
    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C"]
    for i, (name, vals) in enumerate(y_dict.items()):
        fig.add_trace(go.Scatter(
            x=list(x), y=list(vals), mode="lines+markers",
            name=name,
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=6, symbol="circle"),
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        ))
    return _fig_layout(fig, title, height, xlabel, ylabel)

def bar_chart(x, y_dict, title=None, xlabel=None, ylabel=None, height=400, orientation="v"):
    fig = go.Figure()
    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C", "#00BFFF"]
    for i, (name, vals) in enumerate(y_dict.items()):
        fig.add_trace(go.Bar(
            x=list(x) if orientation == "v" else list(vals),
            y=list(vals) if orientation == "v" else list(x),
            orientation=orientation,
            name=name,
            marker_color=colors[i % len(colors)],
            marker_opacity=0.85,
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group" if orientation == "v" else "group")
    return _fig_layout(fig, title, height, xlabel, ylabel)

def pie_chart(labels, values, title=None, height=400, colors=None):
    if colors is None:
        colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C", "#00BFFF", "#8B00FF", "#FF00AA", "#00FFAA"]
    fig = go.Figure(data=[go.Pie(
        labels=list(labels),
        values=list(values),
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside",
        hole=0.45,
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"},
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}, orientation="v"),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
    )
    return fig

def scatter_chart(x, y, labels=None, title=None, xlabel=None, ylabel=None, height=400, color=None, size=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y), mode="markers+text",
        marker=dict(color=color or NEON_BLUE, size=size or 10, opacity=0.8,
                    line=dict(color="#0A0E27", width=1)),
        text=list(labels) if labels is not None else None,
        textposition="top center",
        textfont={"color": "#E0E0FF", "size": 10},
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
    ))
    return _fig_layout(fig, title, height, xlabel, ylabel)

def area_chart(x, y_dict, title=None, xlabel=None, ylabel=None, height=400):
    fig = go.Figure()
    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700"]
    for i, (name, vals) in enumerate(y_dict.items()):
        fig.add_trace(go.Scatter(
            x=list(x), y=list(vals), mode="lines",
            name=name, fill="tonexty",
            line=dict(color=colors[i % len(colors)], width=1.5),
            fillcolor=f"rgba{tuple(list(int(colors[i % len(colors)].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.15])}",
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        ))
    return _fig_layout(fig, title, height, xlabel, ylabel)

def heatmap_chart(data, x_labels, y_labels, title=None, height=400, colorscale=None):
    if colorscale is None:
        colorscale = [
            [0.0, "#0A0E27"],
            [0.25, "#1A2A6A"],
            [0.5, "#00D4FF"],
            [0.75, "#BF00FF"],
            [1.0, "#FF00AA"],
        ]
    fig = go.Figure(data=go.Heatmap(
        z=data, x=x_labels, y=y_labels,
        colorscale=colorscale,
        hovertemplate="(%{x}, %{y})<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"},
        height=height,
        margin=dict(l=80, r=20, t=50, b=60),
        xaxis=dict(gridcolor="#1F2A5A"),
        yaxis=dict(gridcolor="#1F2A5A"),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    return fig

def pareto_chart(x, y1_label, y1_vals, y2_label, y2_vals, title=None, height=400):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x), y=list(y1_vals), mode="lines+markers",
        name=y1_label, yaxis="y",
        line=dict(color=NEON_BLUE, width=2.5),
        hovertemplate=f"{y1_label}: %{{y:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=list(x), y=list(y2_vals), name=y2_label, yaxis="y2",
        marker_color=NEON_PURPLE, opacity=0.5,
        hovertemplate=f"{y2_label}: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"},
        height=height, margin=dict(l=40, r=60, t=50, b=40),
        xaxis=dict(gridcolor="#1F2A5A", title=""),
        yaxis=dict(gridcolor="#1F2A5A", side="left", title=y1_label, title_font={"color": NEON_BLUE}),
        yaxis2=dict(gridcolor="#1F2A5A", side="right", overlaying="y", title=y2_label,
                    title_font={"color": NEON_PURPLE}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    return fig

def radar_chart(labels, values, title=None, height=450):
    fig = go.Figure()
    n = len(labels)
    fig.add_trace(go.Scatterpolar(
        r=list(values) + [values[0]],
        theta=list(labels) + [labels[0]],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.15)",
        line=dict(color=NEON_BLUE, width=2),
        marker=dict(size=5, color=NEON_BLUE),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"},
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", rotation=90,
                           tickfont={"color": "#B0C0E0"}),
            radialaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A",
                          tickfont={"color": "#B0C0E0"}, range=[0, max(values) * 1.2]),
        ),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
        showlegend=False,
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    return fig

def multi_radar_chart(labels_list, values_list, names, title=None, height=500):
    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C"]
    fig = go.Figure()
    for i, (labels, values) in enumerate(zip(labels_list, values_list)):
        n = len(labels)
        fig.add_trace(go.Scatterpolar(
            r=list(values) + [values[0]],
            theta=list(labels) + [labels[0]],
            fill="toself",
            fillcolor=_hex_to_rgba(colors[i % len(colors)], 0.15),
            line=dict(color=colors[i % len(colors)], width=2),
            name=names[i],
            hovertemplate=f"{names[i]}: %{{r:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"},
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", rotation=90,
                           tickfont={"color": "#B0C0E0"}),
            radialaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A",
                          tickfont={"color": "#B0C0E0"}),
        ),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    return fig

def pareto_front_chart(obj1, obj2, pareto_mask=None, labels=None, title=None, height=400):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(obj1), y=list(obj2), mode="markers",
        marker=dict(color="#2A3A5A", size=8, symbol="circle"),
        name="Cac diem thuong",
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
    ))
    if pareto_mask is not None:
        pareto_obj1 = [o1 for o1, m in zip(obj1, pareto_mask) if m]
        pareto_obj2 = [o2 for o2, m in zip(obj2, pareto_mask) if m]
        fig.add_trace(go.Scatter(
            x=pareto_obj1, y=pareto_obj2, mode="lines+markers",
            marker=dict(color=NEON_BLUE, size=10, symbol="star"),
            line=dict(color=NEON_BLUE, width=2),
            name="Pareto Front",
            hovertemplate="Pareto: (%{x:.2f}, %{y:.2f})<extra></extra>",
        ))
    elif labels is not None:
        fig.update_traces(text=list(labels), textposition="top center",
                         textfont={"color": "#E0E0FF", "size": 9})
    return _fig_layout(fig, title, height, "Muc tieu 1", "Muc tieu 2")

def stacked_bar_chart(x, y_dict, title=None, xlabel=None, ylabel=None, height=400):
    fig = go.Figure()
    colors = [NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C"]
    for i, (name, vals) in enumerate(y_dict.items()):
        fig.add_trace(go.Bar(
            x=list(x), y=list(vals), name=name,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=height,
        xaxis=dict(gridcolor="#1F2A5A"),
        yaxis=dict(gridcolor="#1F2A5A"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    return fig

def funnel_chart(stages, values, title=None, height=400):
    fig = go.Figure(go.Funnel(
        y=stages, x=values,
        marker=dict(color=[NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C"]),
        textinfo="value+percent",
        hovertemplate="%{y}<br>%{x}<br>%{percentInitial:.1%}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=height,
        margin=dict(l=80, r=20, t=50, b=40),
        title=dict(text=title, x=0.5, font={"color": NEON_BLUE, "size": 16}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    return fig

def reward_curve_chart(episodes, rewards, title=None, height=350):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(episodes), y=list(rewards), mode="lines",
        line=dict(color=NEON_GREEN, width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 136, 0.08)",
        hovertemplate="Episode %{x}: Reward %{y:.2f}<extra></extra>",
    ))
    return _fig_layout(fig, title, height, "Episode", "Reward")
