import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_macro, load_regions, load_sectors, get_dataset_info
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html


def run():
    st.markdown("""
    <div style="text-align:center; padding:0.8rem; margin:0 -2rem 1rem; 
        background:linear-gradient(90deg,#00D4FF,#BF00FF,#00FF88,#BF00FF,#00D4FF);
        border-top:1px solid rgba(0,212,255,0.2); border-bottom:1px solid rgba(0,212,255,0.2);">
        <div style="font-family:'Orbitron',monospace; font-size:2.5rem; font-weight:900;
            color:#00D4FF; letter-spacing:4px; text-shadow:0 0 20px rgba(0,212,255,0.5);">
            AIDEOM-VN
        </div>
        <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; color:#E0E0FF; 
            margin-top:0.4rem; font-weight:500; letter-spacing:1px;">
            AI-Driven Decision Optimization Model for Vietnam
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(glass_card("""
    <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:#B0C0E0; line-height:1.8;">
        He thong <b style="color:#00D4FF;">AIDEOM-VN</b> la cong cu phan tich va toi uu hoa quyet dinh phat trien
        kinh te Viet Nam, duoc xay dung tren nen tang <b style="color:#BF00FF;">Python, Streamlit, AI & Data Science</b>.
        He thong tich hop <b style="color:#00FF88;">12 mo hinh toi uu hoa</b> tu Ham San Xuat Cobb-Douglas,
        Quy Hoach Tuyen Tinh (LP), toi Q-Learning va NSGA-II, cung cap khuyen nghi chinh sach
        dua tren du lieu thuc te nam 2020-2025.
    </div>
    """), unsafe_allow_html=True)

    info = get_dataset_info()
    df_m = load_macro()
    df_r = load_regions()
    df_s = load_sectors()

    n_sectors = len(df_s) if df_s is not None else 0
    n_regions = len(df_r) if df_r is not None else 0
    n_datasets = sum(1 for v in info.values() if v["available"])

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_html(kpi_card("12", "Tong so mo hinh", "Bai 1 den Bai 12", NEON_BLUE))
    with kpi_cols[1]:
        render_html(kpi_card(f"{n_sectors}", "Tong so nganh", "Kinh te Viet Nam", NEON_GREEN))
    with kpi_cols[2]:
        render_html(kpi_card(f"{n_regions}", "Tong so vung", "6 Vung kinh te", NEON_PURPLE))
    with kpi_cols[3]:
        render_html(kpi_card(f"{n_datasets}", "Bo du lieu", "Du lieu thuc te", NEON_ORANGE))

    st.markdown(section_header("TRUY CAP NHANH", "blue"), unsafe_allow_html=True)
    quick_nav = [
        ("Bai 1", "Cobb Douglas + AI", "#00D4FF", "Du bao GDP & TFP"),
        ("Bai 2", "LP Ngansach so", "#BF00FF", "Phan bo ngan sach"),
        ("Bai 3", "Xep hang nganh", "#00FF88", "TOPSIS 10 nganh"),
        ("Bai 5", "MIP Du an", "#FF6B35", "Chon 15/22 du an"),
        ("Bai 7", "NSGA-II", "#FFD700", "Pareto Front"),
        ("Bai 8", "Mo hinh dong", "#FF4B8C", "Du bao 2026-35"),
        ("Bai 10", "Monte Carlo", "#00BFFF", "Stochastic GDP"),
        ("Bai 11", "Q-Learning", "#8B00FF", "Chinh sach dau tu"),
        ("Bai 12", "Dashboard", "#00FF88", "Tong hop ket qua"),
    ]

    nav_cols = st.columns(3)
    for idx, (num, title, color, desc) in enumerate(quick_nav):
        with nav_cols[idx % 3]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(10,14,40,0.85),rgba(15,25,65,0.7));
                border:1px solid rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4))+[0.3])};
                border-radius:12px; padding:1rem; margin-bottom:0.8rem;
                box-shadow:0 4px 20px rgba(0,0,0,0.4); transition:transform 0.2s;">
                <div style="font-family:Orbitron,monospace; font-size:1.1rem; font-weight:700;
                    color:{color}; text-shadow:0 0 10px {color}55;">{num}</div>
                <div style="font-family:Rajdhani,sans-serif; font-size:0.95rem; font-weight:600;
                    color:#E0E0FF; margin-top:0.3rem;">{title}</div>
                <div style="font-family:Rajdhani,sans-serif; font-size:0.78rem; color:#6070A0;
                    margin-top:0.2rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    if df_m is not None:
        st.markdown(section_header("GDP VIET NAM - XU HUONG 2020-2025", "purple"), unsafe_allow_html=True)
        years = df_m["year"].values.astype(int)
        gdp_t = df_m["GDP_trillion_VND"].values
        gdp_b = df_m["GDP_billion_USD"].values
        growth = df_m["GDP_growth_pct"].values
        fdi = df_m["FDI_disbursed_billion_USD"].values
        digital = df_m["digital_economy_share_GDP_pct"].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(years), y=list(gdp_t),
            mode="lines+markers+text", name="GDP (Ty VND)",
            line=dict(color=NEON_BLUE, width=3),
            marker=dict(size=10, symbol="circle"),
            text=[f"{v:.0f}T" for v in gdp_t],
            textposition="top center",
            textfont={"color": NEON_BLUE, "size": 10},
            hovertemplate="Nam %{x}: %{y:.1f}T VND<br>TT: %{customdata:.2f}%<extra></extra>",
            customdata=growth,
        ))
        fig.add_trace(go.Scatter(
            x=list(years), y=list(fdi * 300),
            mode="lines+markers", name="FDI (B USD, x300)",
            line=dict(color=NEON_GREEN, width=2, dash="dash"),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="Nam %{x}: FDI=$%{y:.1f}B<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"},
            height=380,
            xaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="Nam"),
            yaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="Trieu VND", side="left"),
            yaxis2=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", side="right",
                       overlaying="y", title="FDI (B USD, x300)", showgrid=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}, orientation="h", y=1.1),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig, use_container_width=True)

        kpi2_cols = st.columns(5)
        with kpi2_cols[0]:
            render_html(kpi_card(f"{gdp_t[-1]:.0f}T", "GDP 2025", "Trieu VND", NEON_BLUE))
        with kpi2_cols[1]:
            render_html(kpi_card(f"{growth[-1]:.2f}%", "Tang truong 2025", "GDP", NEON_GREEN))
        with kpi2_cols[2]:
            render_html(kpi_card(f"${fdi[-1]:.1f}B", "FDI 2025", "USD", NEON_PURPLE))
        with kpi2_cols[3]:
            render_html(kpi_card(f"${gdp_b[-1]:.0f}B", "GDP 2025", "USD", NEON_ORANGE))
        with kpi2_cols[4]:
            render_html(kpi_card(f"{digital[-1]:.1f}%", "Kinh te so", "GDP", "#FFD700"))

    st.markdown(section_header("12 MO HINH TRONG HE THONG", "green"), unsafe_allow_html=True)

    models = [
        ("Bai 1", "Cobb Douglas + AI", "Hoi quy", "GDP du bao, TFP, Growth Accounting"),
        ("Bai 2", "LP Ngansach", "Tuyen tinh", "Phan bo ngan sach 10 nganh"),
        ("Bai 3", "TOPSIS Nganh", "Da tieu chi", "Xep hang uu tien 10 nganh"),
        ("Bai 4", "LP Vung", "Tuyen tinh", "Phan bo nguon luc 6 vung"),
        ("Bai 5", "MIP Du an", "So nguyen hon hop", "Chon 15/22 du an toi uu"),
        ("Bai 6", "TOPSIS Vung", "Da tieu chi", "Xep hang 6 vung kinh te"),
        ("Bai 7", "NSGA-II", "Da muc tieu", "Pareto Front GDP-LD-AI"),
        ("Bai 8", "He thong dong", "Phuong trinh vi phan", "Du bao GDP 2026-2035"),
        ("Bai 9", "AI & LD", "Phan tich tac dong", "Thay the & tao viec lam AI"),
        ("Bai 10", "Monte Carlo", "Stochastic", "Khoang du bao GDP 90% CI"),
        ("Bai 11", "Q-Learning", "Hoc tang cuong", "Chinh sach dau tu vung"),
        ("Bai 12", "Dashboard", "Tong hop", "Tong hop 12 mo hinh"),
    ]

    model_df = pd.DataFrame(models, columns=["Bai", "Ten mo hinh", "Loai", "Ket qua"])
    st.dataframe(model_df, use_container_width=True)

    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 0.5rem; margin-top:1rem;
        border-top:1px solid rgba(0,212,255,0.15);">
        <div style="font-family:Orbitron,monospace; font-size:0.75rem; color:#4A5A8A; text-transform:uppercase;
            letter-spacing:2px;">
            AIDEOM-VN | AI-Driven Decision Optimization Model for Vietnam
        </div>
        <div style="font-family:Rajdhani,sans-serif; font-size:0.8rem; color:#3A4A7A; margin-top:0.3rem;">
            Xay dung tren Python, Streamlit, Scikit-learn, PuLP, NumPy, SciPy, Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)
