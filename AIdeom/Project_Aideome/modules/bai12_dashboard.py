import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_macro, load_regions, load_sectors, get_dataset_info
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.helpers import cagr

np.random.seed(42)

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _compute_ai_labor_totals(df_s, df_m):
    """Tinh total_displaced, total_labor tu bai 9."""
    if df_s is None or df_m is None:
        return 0.0, 0.0

    total_displaced = 0.0
    total_labor = 0.0
    for _, row in df_s.iterrows():
        labor = float(row["labor_million"])
        auto_risk = float(row["automation_risk_pct"])
        displaced = labor * (auto_risk / 100.0) * 0.7
        total_displaced += displaced
        total_labor += labor
    return total_displaced, total_labor


def _compute_stochastic_gdp_bounds(df, n_sim=1000, n_years=5, conf_level=0.90):
    """Tinh gdp_p5, gdp_p95 tu bai 10."""
    if df is None:
        return 0.0, 0.0

    col_growth = "GDP_growth_pct"
    gdp = df["GDP_trillion_VND"].values
    growth_hist = df[col_growth].values
    mu_g = np.mean(growth_hist)
    sigma_g = np.std(growth_hist)
    gdp_start = gdp[-1]

    sim_gdp = np.zeros((n_sim, n_years))
    for s in range(n_sim):
        gdp_t = gdp_start
        for t in range(n_years):
            g_new = np.random.normal(mu_g / 100.0, sigma_g / 100.0)
            gdp_t = gdp_t * (1 + g_new)
            sim_gdp[s, t] = gdp_t

    gdp_p5 = np.percentile(sim_gdp[:, -1], (1 - conf_level) / 2 * 100)
    gdp_p95 = np.percentile(sim_gdp[:, -1], (1 + conf_level) / 2 * 100)
    return float(gdp_p5), float(gdp_p95)


def _compute_budget_total():
    """Tra ve gia tri mac dinh cua total_budget tu bai 2."""
    return 10.0


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Tổng hợp kết quả từ 12 mô hình tối ưu hóa và dự báo.",
        "So sánh 5 kịch bản phát triển: S1-S5.",
        "Đưa ra khuyến nghị chính sách tổng hợp cho Việt Nam đến 2030.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)
    st.markdown("- Dữ liệu vĩ mô: GDP, FDI, xuất khẩu, kinh tế số 2020-2025 (6 năm)")
    st.markdown("- Dữ liệu 6 vùng kinh tế: GRDP, dân số, FDI, số hóa, AI Readiness")
    st.markdown("- Dữ liệu 10 ngành kinh tế: GDP, xuất khẩu, số hóa, AI Readiness, rủi ro tự động hóa")

    df_m = load_macro()
    df_r = load_regions()
    df_s = load_sectors()
    info = get_dataset_info()

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Shared macro data for all scenarios
    if df_m is not None:
        gdp_2025 = df_m["GDP_trillion_VND"].values[-1]
        gdp_growth_2025 = df_m["GDP_growth_pct"].values[-1]
        fdi_2025 = df_m["FDI_disbursed_billion_USD"].values[-1]
        exp_2025 = df_m["exports_billion_USD"].values[-1]
        digital_2025 = df_m["digital_economy_share_GDP_pct"].values[-1]

    # ---- 12.4.1 Kịch bản S1 ----
    st.markdown('<div class="section-header-purple">12.4.1 Kịch bản S1 - Bi quan</div>', unsafe_allow_html=True)

    if df_m is not None:
        s1_gdp_2030 = gdp_2025 * (1 + (gdp_growth_2025 * 0.7 / 100)) ** 5
        s1_fdi = fdi_2025 * 0.75
        s1_exp = exp_2025 * 0.80
        s1_digital = digital_2025 * 0.85

        s1_df = pd.DataFrame({
            "Chỉ tiêu": ["GDP 2030 (Tỷ VND)", "GDP 2030 (Tỷ USD)", "FDI 2030 (B USD)", "Xuất khẩu 2030 (B USD)", "Kinh tế số (% GDP)"],
            "Giá trị": [f"{s1_gdp_2030:.0f}", f"{s1_gdp_2030*1000/23807:.1f}", f"{s1_fdi:.1f}", f"{s1_exp:.1f}", f"{s1_digital:.1f}"],
            "Diễn giải": [
                "GDP tăng chậm 30% so với xu hướng",
                "GDP tính theo tỷ giá 23,807 VND/USD",
                "FDI giảm do suy thoái toàn cầu",
                "Xuất khẩu suy giảm 20%",
                "Chuyển đổi số chậm lại",
            ]
        })
        st.dataframe(s1_df, width='stretch')

    # ---- 12.4.2 Kịch bản S3 ----
    st.markdown(section_header("12.4.2 Kịch bản S3 - Lạc quan", "purple"), unsafe_allow_html=True)

    if df_m is not None:
        s3_gdp_2030 = gdp_2025 * (1 + (gdp_growth_2025 * 1.20 / 100)) ** 5
        s3_fdi = fdi_2025 * 1.25
        s3_exp = exp_2025 * 1.20
        s3_digital = min(digital_2025 * 1.30, 40.0)

        s3_df = pd.DataFrame({
            "Chỉ tiêu": ["GDP 2030 (Tỷ VND)", "GDP 2030 (Tỷ USD)", "FDI 2030 (B USD)", "Xuất khẩu 2030 (B USD)", "Kinh tế số (% GDP)"],
            "Giá trị": [f"{s3_gdp_2030:.0f}", f"{s3_gdp_2030*1000/23807:.1f}", f"{s3_fdi:.1f}", f"{s3_exp:.1f}", f"{s3_digital:.1f}"],
            "Diễn giải": [
                "GDP tăng nhanh 20% so với xu hướng",
                "GDP tính theo tỷ giá 23,807 VND/USD",
                "FDI tăng mạnh nhờ môi trường thuận lợi",
                "Xuất khẩu tăng 20%",
                "Kinh tế số đạt 40% GDP",
            ]
        })
        st.dataframe(s3_df, width='stretch')

    # ---- 12.4.3 Kịch bản S5 ----
    st.markdown(section_header("12.4.3 Kịch bản S5 - Chuyển đổi số quốc gia", "green"), unsafe_allow_html=True)

    if df_m is not None:
        s5_gdp_2030 = gdp_2025 * (1 + (gdp_growth_2025 * 1.10 / 100)) ** 5
        s5_fdi = fdi_2025 * 1.15
        s5_exp = exp_2025 * 1.15
        s5_digital = min(digital_2025 * 1.50, 45.0)

        s5_df = pd.DataFrame({
            "Chỉ tiêu": ["GDP 2030 (Tỷ VND)", "GDP 2030 (Tỷ USD)", "FDI 2030 (B USD)", "Xuất khẩu 2030 (B USD)", "Kinh tế số (% GDP)"],
            "Giá trị": [f"{s5_gdp_2030:.0f}", f"{s5_gdp_2030*1000/23807:.1f}", f"{s5_fdi:.1f}", f"{s5_exp:.1f}", f"{s5_digital:.1f}"],
            "Diễn giải": [
                "GDP tăng trưởng bền vững nhờ chuyển đổi số",
                "GDP tính theo tỷ giá 23,807 VND/USD",
                "FDI tăng 15% với chất lượng cao hơn",
                "Xuất khẩu tăng 15%",
                "Kinh tế số đạt 45% GDP — mục tiêu chiến lược",
            ]
        })
        st.dataframe(s5_df, width='stretch')

    # KPIs tổng quan
    kpi0_cols = st.columns(4)
    with kpi0_cols[0]:
        render_html(kpi_card("12", "Tổng số mô hình", "Bài 1 - Bài 12", NEON_BLUE))
    with kpi0_cols[1]:
        n_sectors = len(df_s) if df_s is not None else 0
        render_html(kpi_card(f"{n_sectors}", "Tổng số ngành", "Kinh tế Việt Nam", NEON_GREEN))
    with kpi0_cols[2]:
        n_regions = len(df_r) if df_r is not None else 0
        render_html(kpi_card(f"{n_regions}", "Tổng số vùng", "6 vùng kinh tế", NEON_PURPLE))
    with kpi0_cols[3]:
        n_datasets = sum(1 for v in info.values() if v["available"])
        render_html(kpi_card(f"{n_datasets}", "Bộ dữ liệu", "Dữ liệu thực tế", NEON_ORANGE))

    # GDP overview
    if df_m is not None:
        st.markdown(section_header("GDP & Kinh tế Việt Nam 2020-2025", "purple"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            years = df_m["year"].values.astype(int)
            gdp_t = df_m["GDP_trillion_VND"].values
            fig_gdp = go.Figure()
            fig_gdp.add_trace(go.Scatter(
                x=list(years), y=list(gdp_t),
                mode="lines+markers", name="GDP (Tỷ VND)",
                line=dict(color=NEON_BLUE, width=3),
                marker=dict(size=8),
                hovertemplate="Năm %{x}: %{y:.1f}T VND<extra></extra>",
            ))
            fig_gdp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=350,
                xaxis=dict(gridcolor="#1F2A5A", title="Năm"),
                yaxis=dict(gridcolor="#1F2A5A", title="Tỷ VND"),
                title=dict(text="GDP Việt Nam 2020-2025", x=0.5, font={"color": NEON_BLUE, "size": 14}),
                hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
            )
            st.plotly_chart(fig_gdp, width='stretch')

        with c2:
            fig_share = go.Figure()
            ag = df_m["agriculture_share_pct"].values
            ind = df_m["industry_share_pct"].values
            ser = df_m["services_share_pct"].values
            fig_share.add_trace(go.Bar(x=list(years), y=list(ag), name="Nông nghiệp", marker_color=NEON_GREEN, opacity=0.8))
            fig_share.add_trace(go.Bar(x=list(years), y=list(ind), name="Công nghiệp", marker_color=NEON_BLUE, opacity=0.8))
            fig_share.add_trace(go.Bar(x=list(years), y=list(ser), name="Dịch vụ", marker_color=NEON_PURPLE, opacity=0.8))
            fig_share.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=350,
                xaxis=dict(gridcolor="#1F2A5A", title="Năm"),
                yaxis=dict(gridcolor="#1F2A5A", title="Phần trăm (%)"),
                title=dict(text="Cơ cấu GDP 2020-2025", x=0.5, font={"color": NEON_BLUE, "size": 14}),
                legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
                hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
            )
            st.plotly_chart(fig_share, width='stretch')

        kpi1_cols = st.columns(5)
        gdp_growth_2025 = df_m["GDP_growth_pct"].values[-1]
        fdi_2025 = df_m["FDI_disbursed_billion_USD"].values[-1]
        exp_2025 = df_m["exports_billion_USD"].values[-1]
        digital_2025 = df_m["digital_economy_share_GDP_pct"].values[-1]
        with kpi1_cols[0]:
            render_html(kpi_card(f"{gdp_t[-1]:.0f}T", "GDP 2025", "Tỷ VND", NEON_BLUE))
        with kpi1_cols[1]:
            render_html(kpi_card(f"{gdp_growth_2025:.2f}%", "Tăng trưởng 2025", "GDP", NEON_GREEN))
        with kpi1_cols[2]:
            render_html(kpi_card(f"${fdi_2025:.1f}B", "FDI 2025", "USD", NEON_PURPLE))
        with kpi1_cols[3]:
            render_html(kpi_card(f"${exp_2025:.0f}B", "Xuất khẩu 2025", "USD", NEON_ORANGE))
        with kpi1_cols[4]:
            render_html(kpi_card(f"{digital_2025:.1f}%", "Kinh tế số 2025", "GDP", "#FFD700"))

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ TỔNG HỢP", "blue"), unsafe_allow_html=True)

    if df_r is not None:
        regions = df_r["region_name_en"].tolist()
        vn_names = {
            "Northern Midlands and Mountains": "Trung du mìên núi",
            "Red River Delta": "Đồng Bằng Sông Hồng",
            "North Central and South Central Coast": "Bắc Trung Bộ",
            "Central Highlands": "Tây Nguyên",
            "Southeast": "Đông Nam Bộ",
            "Mekong Delta": "Đồng Bằng Sông Cửu Long",
        }
        regions_vn = [vn_names.get(r, r) for r in regions]
        grdp = df_r["grdp_trillion_VND"].values

        rc1, rc2 = st.columns(2)
        with rc1:
            fig_r = go.Figure()
            sorted_idx = np.argsort(grdp)[::-1]
            colors = [NEON_BLUE, NEON_GREEN, NEON_PURPLE, NEON_ORANGE, "#FFD700", "#FF4B8C"]
            fig_r.add_trace(go.Bar(
                x=[regions_vn[i] for i in sorted_idx],
                y=[grdp[i] for i in sorted_idx],
                marker_color=[colors[j % len(colors)] for j in range(len(sorted_idx))],
                opacity=0.85,
                hovertemplate="Vùng: %{x}<br>GRDP: %{y:.1f}T<extra></extra>",
            ))
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=380,
                xaxis=dict(gridcolor="#1F2A5A", tickangle=20, title=""),
                yaxis=dict(gridcolor="#1F2A5A", title="GRDP (Tỷ VND)"),
                title=dict(text="GRDP 6 vùng kinh tế 2024", x=0.5, font={"color": NEON_BLUE, "size": 14}),
                hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
            )
            st.plotly_chart(fig_r, width='stretch')

        with rc2:
            fig_pie = go.Figure(data=[go.Pie(
                labels=regions_vn, values=grdp,
                marker_colors=colors,
                textinfo="label+percent", textposition="outside",
                hole=0.4,
                hovertemplate="%{label}: %{percent}<extra></extra>",
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=380,
                title=dict(text="Tỷ trọng GRDP theo vùng", x=0.5, font={"color": NEON_BLUE, "size": 14}),
                legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            )
            st.plotly_chart(fig_pie, width='stretch')

    # Model comparison
    st.markdown(section_header("So sánh các mô hình tối ưu", "purple"), unsafe_allow_html=True)
    model_comparison = pd.DataFrame({
        "Mô hình": ["Cobb-Douglas", "LP Ngân sách", "TOPSIS Ngành", "LP Vùng", "MIP Dự án", "TOPSIS Vùng", "NSGA-II", "Hệ thống động", "AI & LD", "Monte Carlo", "Q-Learning"],
        "Loại": ["Hồi quy", "Tuyến tính", "Đa tiêu chí", "Tuyến tính", "Số nguyên", "Đa tiêu chí", "Đa mục tiêu", "Động", "Phân tích", "Ngẫu nhiên", "Học tăng cường"],
        "Kết quả chính": ["GDP dự báo, TFP", "Phân bổ ngân sách", "Xếp hạng 10 ngành", "Phân bổ 6 vùng", "Chọn 15/22 DA", "Xếp hạng 6 vùng", "Pareto Front", "Dự báo 2026-35", "Thay thế/tạo LĐ", "Khoảng dự báo GDP", "Chính sách đầu tư"],
        "Trạng thái": [
            "Sẵn sàng" if HAS_SKLEARN else "Chưa sẵn sàng",
            "Sẵn sàng" if HAS_PULP else "Chưa sẵn sàng",
            "Sẵn sàng",
            "Sẵn sàng" if HAS_PULP else "Chưa sẵn sàng",
            "Sẵn sàng" if HAS_PULP else "Chưa sẵn sàng",
            "Sẵn sàng", "Sẵn sàng", "Sẵn sàng", "Sẵn sàng", "Sẵn sàng", "Sẵn sàng",
        ],
    })
    # show index starting at 1 instead of 0
    model_comparison.index = range(1, len(model_comparison) + 1)
    model_comparison.index.name = "STT"
    st.dataframe(model_comparison, width='stretch')

    # ===== TÍNH TOÁN BIẾN TỪ CÁC BÀI KHÁC =====
    total_displaced, total_labor = _compute_ai_labor_totals(df_s, df_m)
    gdp_p5, gdp_p95 = _compute_stochastic_gdp_bounds(df_m)
    total_budget = _compute_budget_total()

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    s3_digital = min(digital_2025 * 1.30, 40.0)
    s5_digital = min(digital_2025 * 1.50, 45.0)
    s3_gdp_vnd = gdp_2025 * (1 + (gdp_growth_2025 * 1.20 / 100)) ** 5
    s5_gdp_vnd = gdp_2025 * (1 + (gdp_growth_2025 * 1.10 / 100)) ** 5
    s1_gdp_vnd = gdp_2025 * (1 + (gdp_growth_2025 * 0.7 / 100)) ** 5
    s1_fdi = fdi_2025 * 0.75
    s3_fdi = fdi_2025 * 1.25
    s5_fdi = fdi_2025 * 1.15

    with st.expander("Kịch bản nào là tối ưu cho Việt Nam giai đoạn 2026-2030?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        So sánh 3 kịch bản chính:
        <br>- <b style="color:#FF6B35;">S1 (Bi quan):</b> GDP 2030 = {s1_gdp_vnd:.0f}T VND ({s1_gdp_vnd*1000/23807:.1f}T USD),
        FDI = ${s1_fdi:.1f}B, Kinh tế số = {digital_2025*0.85:.1f}% GDP.
        <br>- <b style="color:#00FF88;">S5 (Chuyển đổi số):</b> GDP 2030 = {s5_gdp_vnd:.0f}T VND ({s5_gdp_vnd*1000/23807:.1f}T USD),
        FDI = ${s5_fdi:.1f}B, Kinh tế số = {s5_digital:.0f}% GDP.
        <br>- <b style="color:#00D4FF;">S3 (Lạc quan):</b> GDP 2030 = {s3_gdp_vnd:.0f}T VND ({s3_gdp_vnd*1000/23807:.1f}T USD),
        FDI = ${s3_fdi:.1f}B, Kinh tế số = {s3_digital:.0f}% GDP.
        <br><b style="color:#FFD700;">S5 (Chuyển đổi số quốc gia)</b> là kịch bản được khuyến nghị
        vì cân bằng tốt nhất giữa tăng trưởng GDP ({s5_gdp_vnd*1000/23807:.1f}T USD) và
        kinh tế số ({s5_digital:.0f}% GDP), đồng thời duy trì thu hút FDI ổn định ({s5_fdi:.1f}B USD).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        S5 được khuyến nghị vì: (1) <b style="color:#00FF88;">Cân bằng tăng trưởng và bền vững</b> —
        Tăng trưởng GDP 10% trên xu hướng vừa đủ để đạt mục tiêu thoát bẫy thu nhập trung bình.
        (2) <b style="color:#FFD700;">Kinh tế số mục tiêu {s5_digital:.0f}% GDP</b> —
        Phù hợp với Chiến lược quốc gia về phát triển kinh tế số 2025-2030 của Việt Nam.
        (3) <b style="color:#00D4FF;">FDI tăng 15%</b> — Thu hút vốn có chất lượng, không quá phụ thuộc
        vào một thị trường. (4) <b style="color:#00FF88;">Khả thi về chính sách</b> —
        Mức tăng trưởng 10% có thể đạt được với đầu tư công và tư nhân phù hợp.
        Khác với S3 (lạc quan quá), S5 không đặt kỳ vọng quá cao vào điều kiện thuận lợi.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Kịch bản S5 đưa GDP 2030 lên <b style="color:#FFD700;">{s5_gdp_vnd*1000/23807:.1f} Tỷ USD</b>,
        kinh tế số đạt <b style="color:#00FF88;">{s5_digital:.0f}% GDP</b> —
        đưa Việt Nam vào nhóm dẫn đầu ASEAN về số hóa. GDP per capita ước tính
        khoảng {s5_gdp_vnd*1000/23807/n_datasets*1000:.0f} USD (với {n_datasets} triệu dân).
        Mức tăng trưởng này đủ để <b style="color:#00FF88;">thoát bẫy thu nhập trung bình</b>
        nếu duy trì đà tăng trưởng và đầu tư vào giáo dục, R&D.
        Khoảng cách với S3 chỉ {s3_gdp_vnd*1000/23807 - s5_gdp_vnd*1000/23807:.1f}T USD GDP,
        nhưng S5 an toàn hơn về rủi ro.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Áp dụng S5 làm kịch bản cơ sở</b> — Dùng S5 để xây dựng
        kế hoạch 5 năm và chiến lược phát triển 2026-2030. (2) <b style="color:#00D4FF;">Đầu tư vào chuyển đổi số</b> —
        Phân bổ ngân sách tập trung cho hạ tầng số, nền tảng AI, và đào tạo nhân lực số.
        (3) <b style="color:#00D4FF;">Giám sát và điều chỉnh</b> — Theo dõi chỉ số kinh tế số hàng quý,
        điều chỉnh chính sách nếu thực tế chệch khỏi S5. (4) <b style="color:#00D4FF;">Chuẩn bị phương án B</b> —
        Lập kế hoạch ứng phó nếu thực tế diễn ra theo S1 (bi quan).
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Việt Nam nên ưu tiên mục tiêu nào trong giai đoạn tới?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Từ kết quả 12 mô hình, có thể xác định 3 ưu tiên chiến lược cho Việt Nam:
        <br><b style="color:#00FF88;">Ưu tiên 1: Chuyển đổi số toàn diện</b> — Kết quả Cobb-Douglas
        (γ = cao), TOPSIS ngành (Top 3 đều có số hóa tốt), TOPSIS vùng (Đông Nam Bộ,
        ĐBSH dẫn đầu nhờ số hóa) đều cho thấy kinh tế số là <b style="color:#FFD700;">động lực tăng trưởng
        chính</b>. Mục tiêu {s5_digital:.0f}% GDP phù hợp với kết quả dự báo.
        <br><b style="color:#FFD700;">Ưu tiên 2: Nâng cao năng suất lao động</b> — AI & Lao động (Bài 9)
        cho thấy cần đào tạo {total_displaced*0.6:.1f}M lao động, đảm bảo hiệu suất chuyển đổi.
        <br><b style="color:#00D4FF;">Ưu tiên 3: Thu hút FDI chất lượng cao</b> — LP vùng, NSGA-II cho thấy
        FDI là yếu tố then chốt cho cả tăng trưởng GDP và chuyển giao công nghệ.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Ba ưu tiên này xuất phát từ: (1) <b style="color:#00FF88;">TFP và kinh tế số tương quan mạnh</b> —
        Kết quả Cobb-Douglas cho thấy γ cao → mỗi 1% tăng kinh tế số tạo ra
        nhiều GDP. (2) <b style="color:#FFD700;">Thị trường lao động cần chuyển đổi</b> —
        {total_displaced:.2f}M lao động bị thay thế bởi AI, cần chương trình đào tạo lại
        quy mô lớn. (3) <b style="color:#00D4FF;">FDI là động lực kép</b> — vừa tăng GDP trực tiếp,
        vừa chuyển giao công nghệ và nâng cao AI Readiness. (4) <b style="color:#00D4FF;">Ba yếu tố bổ trợ lẫn nhau</b> —
        chuyển đổi số tạo nhu cầu nhân lực mới → đào tạo nhân lực → thu hút FDI công nghệ cao →
        tăng trưởng kinh tế số.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Ba ưu tiên tạo thành <b style="color:#FFD700;">vòng xoắn tích cực</b>:
        Chuyển đổi số → tăng năng suất → thu hút FDI công nghệ cao →
        tạo việc làm chất lượng → tăng thu nhập → tăng tiêu dùng nội địa →
        tăng trưởng GDP → nhiều nguồn lực cho chuyển đổi số.
        Kết quả Monte Carlo (Bài 10) xác nhận ba ưu tiên này giúp
        tăng xác suất đạt mục tiêu +50% GDP lên <b style="color:#00FF88;">cao nhất</b>.
        Kết quả NSGA-II (Bài 7) cho thấy đầu tư vào ba ưu tiên này
        cân bằng tốt giữa GDP, việc làm và AI.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Ưu tiên 1: Chuyển đổi số toàn diện</b> — Ngân sách 15-20B USD/năm
        cho hạ tầng số hóa, AI, và nền tảng số quốc gia. (2) <b style="color:#00D4FF;">Ưu tiên 2: Nâng cao năng suất lao động</b> —
        Đào tạo lại {total_displaced*0.6:.1f}M lao động, nâng cao kỹ năng số cho {total_labor*0.3:.1f}M lao động.
        (3) <b style="color:#00D4FF;">Ưu tiên 3: Thu hút FDI công nghệ cao</b> —
        Chính sách ưu đãi cho dự án có R&D, chuyển giao công nghệ, vào khu công nghệ cao.
        (4) <b style="color:#00D4FF;">Phân bổ ngân sách LP tối ưu</b> — Áp dụng kết quả
        LP ngành (Bài 2) để phân bổ {total_budget:.0f}B USD cho 10 ngành.
        (5) <b style="color:#00D4FF;">Cân bằng vùng</b> — Áp dụng NSGA-II (Bài 7) để phân bổ
        nguồn lực liên vùng cân bằng GDP, việc làm, và AI.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Khuyến nghị chính sách tổng hợp đến năm 2030 từ 12 mô hình?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        12 mô hình trong AIDEOM cung cấp căn cứ khoa học cho 5 nhóm khuyến nghị chính sách:
        <br><b style="color:#00FF88;">1. Chiến lược kinh tế vĩ mô:</b> Theo kịch bản S5
        (chuyển đổi số quốc gia), GDP 2030 = {s5_gdp_vnd*1000/23807:.1f}T USD, kinh tế số = {s5_digital:.0f}% GDP.
        <br><b style="color:#FFD700;">2. Phân bổ ngành:</b> Áp dụng LP (Bài 2) với {total_budget:.0f}B USD,
        ưu tiên CNTT, Tài chính, Logistics chiếm >40% ngân sách.
        <br><b style="color:#00D4FF;">3. Phân bổ vùng:</b> Áp dụng NSGA-II (Bài 7) và Q-Learning (Bài 11),
        đầu tư mạnh vào Đông Nam Bộ, ĐBSH; đảm bảo ngưỡng tối thiểu cho vùng khó khăn.
        <br><b style="color:#BF00FF;">4. Nhân lực & AI:</b> Chương trình đào tạo lại {total_displaced*0.6:.1f}M lao động
        từ Bài 9, kết hợp Q-Learning để tối ưu hóa đầu tư nhân lực.
        <br><b style="color:#FF6B35;">5. Quản lý rủi ro:</b> Áp dụng Monte Carlo (Bài 10) để xây dựng
        khoảng tin cậy GDP 2030: [{gdp_p5:.0f} - {gdp_p95:.0f}] Tỷ VND.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        12 mô hình bổ trợ lẫn tạo thành hệ thống ra quyết định toàn diện:
        (1) <b style="color:#00FF88;">Cobb-Douglas (Bài 1)</b> xác định TFP và động lực tăng trưởng.
        (2) <b style="color:#FFD700;">LP & MIP (Bài 2, 4, 5)</b> tối ưu hóa phân bổ nguồn lực.
        (3) <b style="color:#00D4FF;">TOPSIS (Bài 3, 6)</b> xếp hạng ngành và vùng ưu tiên.
        (4) <b style="color:#BF00FF;">NSGA-II (Bài 7)</b> cân bằng đa mục tiêu.
        (5) <b style="color:#FF6B35;">Hệ thống động (Bài 8)</b> dự báo dài hạn.
        (6) <b style="color:#00D4FF;">AI & Lao động (Bài 9)</b> đánh giá tác động AI.
        (7) <b style="color:#FFD700;">Monte Carlo (Bài 10)</b> quản lý rủi ro.
        (8) <b style="color:#00FF88;">Q-Learning (Bài 11)</b> tối ưu chính sách đầu tư vùng động.
        (9) <b style="color:#00D4FF;">Dashboard (Bài 12)</b> tổng hợp toàn bộ kết quả.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        AIDEOM là hệ thống <b style="color:#FFD700;">ra quyết định dựa trên dữ liệu</b> (data-driven)
        cho phát triển kinh tế Việt Nam. Thay vì dựa vào kinh nghiệm chủ quan,
        nhà hoạch định chính sách có căn cứ định lượng từ 12 mô hình khác nhau.
        Mỗi mô hình đóng vai trò chuyên biệt, kết hợp lại tạo ra
        <b style="color:#00FF88;">bức tranh toàn cảnh và nhất quán</b> về chiến lược phát triển.
        Tất cả các mô hình đều hội tụ về 3 ưu tiên: chuyển đổi số, nâng cao năng suất,
        và thu hút FDI chất lượng — cho thấy đây là <b style="color:#FFD700;">định hướng chiến lược đúng đắn</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Xây dựng Ủy ban AIDEOM</b> — Tích hợp kết quả 12 mô hình
        vào quy trình hoạch định chính sách kinh tế hàng năm. (2) <b style="color:#00D4FF;">Cập nhật mô hình định kỳ</b> —
        Chạy lại tất cả 12 mô hình mỗi năm với dữ liệu mới nhất để điều chỉnh chính sách.
        (3) <b style="color:#00D4FF;">Kết nối các mô hình</b> — Dùng kết quả Bài 1-3 để phân bổ ngân sách,
        Bài 4-7 để phân bổ vùng, Bài 8-11 để dự báo và quản lý rủi ro. (4) <b style="color:#00D4FF;">Đào tạo nhân lực phân tích</b> —
        Xây dựng đội ngũ chuyên gia có thể vận hành và diễn giải kết quả AIDEOM.
        (5) <b style="color:#00D4FF;">Báo cáo tích hợp</b> — Tạo dashboard tổng hợp trình Chính phủ
        mỗi quý, cập nhật kịch bản S1/S5 và khuyến nghị chính sách cụ thể.
        </div>
        """, unsafe_allow_html=True)
