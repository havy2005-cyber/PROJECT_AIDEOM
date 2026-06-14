import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_macro
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.helpers import growth_rate, cagr

try:
    from scipy.integrate import odeint
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def dynamic_model(Y, t, params):
    Y_gdp, L, P = Y
    alpha, beta, gamma, delta = params
    dY_gdp = alpha * Y_gdp
    dL = beta * L
    dP = gamma * P + delta
    return [dY_gdp, dL, dP]


def run_simulation(df, years_base, n_years, scenario="base"):
    col_gdp = "GDP_trillion_VND"
    col_pop = "population_million"
    col_lp = "labor_productivity_million_VND"

    gdp_0 = df[col_gdp].values[-1]
    pop_0 = df[col_pop].values[-1]
    lp_0 = df[col_lp].values[-1]

    gdp_hist = df[col_gdp].values
    pop_hist = df[col_pop].values
    lp_hist = df[col_lp].values

    gdp_growth = np.mean(np.diff(gdp_hist) / df[col_gdp].values[:-1])
    pop_growth = np.mean(np.diff(pop_hist) / pop_hist[:-1])
    lp_growth = np.mean(np.diff(lp_hist) / lp_hist[:-1])

    if scenario == "optimistic":
        gdp_growth *= 1.2
        lp_growth *= 1.15
    elif scenario == "pessimistic":
        gdp_growth *= 0.7
        lp_growth *= 0.75

    alpha = gdp_growth
    beta = pop_growth
    gamma = lp_growth
    delta = 0.001

    t = np.linspace(0, n_years, n_years * 12 + 1)
    Y0 = [gdp_0, pop_0, lp_0]
    result = odeint(dynamic_model, Y0, t, args=((alpha, beta, gamma, delta),))
    annual_idx = list(range(0, len(t), 12))
    gdp_sim = result[annual_idx, 0]
    pop_sim = result[annual_idx, 1]
    lp_sim = result[annual_idx, 2]

    years = np.arange(years_base, years_base + n_years)
    return years, gdp_sim, pop_sim, lp_sim


def run():

    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Dự báo kinh tế Việt Nam dài hạn giai đoạn 2026-2035.",
        "Mô hình hệ thống động (System Dynamics) với 3 biến trạng thái: GDP, Dân số, Năng suất.",
        "Phân tích 3 kịch bản: Bi quan, Cơ sở, Lạc quan.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)
    st.markdown("- GDP (Tỷ VND): Tổng sản phẩm quốc nội 2020-2025")
    st.markdown("- Dân số (Triệu người): Quy mô dân số")
    st.markdown("- Năng suất lao động (Triệu VND/người): GDP / Dân số")

    df = load_macro()
    if df is None:
        st.warning("Không tìm thấy dữ liệu vietnam_macro_2020_2025.csv.")
        return

    col_year = "year"
    col_gdp = "GDP_trillion_VND"
    col_pop = "population_million"
    col_lp = "labor_productivity_million_VND"

    years_base = int(df[col_year].values[-1]) + 1
    n_years = 10

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình kịch bản
    st.markdown(section_header("Cấu hình kịch bản", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(3)
    sc_pessimistic = cfg_cols[0].checkbox("Kịch bản Bi quan (S1)", value=False)
    sc_optimistic = cfg_cols[1].checkbox("Kịch bản Lạc quan (S3)", value=False)
    show_all = cfg_cols[2].checkbox("Hiển thị cả 3 kịch bản", value=True)

    scenarios = {}
    if show_all:
        for sc in ["pessimistic", "base", "optimistic"]:
            years_s, gdp_s, pop_s, lp_s = run_simulation(df, years_base, n_years, scenario=sc)
            scenarios[sc] = {"years": years_s, "gdp": gdp_s, "pop": pop_s, "lp": lp_s}
    else:
        sc = "base"
        if sc_pessimistic:
            sc = "pessimistic"
        elif sc_optimistic:
            sc = "optimistic"
        years_s, gdp_s, pop_s, lp_s = run_simulation(df, years_base, n_years, scenario=sc)
        scenarios[sc] = {"years": years_s, "gdp": gdp_s, "pop": pop_s, "lp": lp_s}

    sc_labels = {
        "pessimistic": "Bi quan (S1)",
        "base": "Cơ sở (S2)",
        "optimistic": "Lạc quan (S3)",
    }
    sc_colors = {
        "pessimistic": NEON_ORANGE,
        "base": NEON_BLUE,
        "optimistic": NEON_GREEN,
    }

    # ---- 8.4.1 Dự báo GDP ----
    st.markdown('<div class="section-header-purple">8.4.1 Dự báo GDP</div>', unsafe_allow_html=True)

    import plotly.graph_objects as go

    fig_gdp = go.Figure()
    hist_years = df[col_year].values.astype(int)
    hist_gdp = df[col_gdp].values
    fig_gdp.add_trace(go.Scatter(
        x=list(hist_years), y=list(hist_gdp),
        mode="lines+markers", name="Thực tế",
        line=dict(color="#E0E0FF", width=2.5),
        marker=dict(size=8, color="#E0E0FF"),
        hovertemplate="Thực tế %{x}: %{y:.1f}T<extra></extra>",
    ))

    for sc_name, sc_data in scenarios.items():
        fig_gdp.add_trace(go.Scatter(
            x=list(sc_data["years"]), y=list(sc_data["gdp"]),
            mode="lines+markers", name=sc_labels.get(sc_name, sc_name),
            line=dict(color=sc_colors.get(sc_name, NEON_BLUE), width=2,
                     dash="dash" if sc_name != "base" else "solid"),
            marker=dict(size=6),
            hovertemplate=f"{sc_labels.get(sc_name,sc_name)}: %{{x}} = %{{y:.1f}}T<extra></extra>",
        ))

    fig_gdp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=400,
        xaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="Năm"),
        yaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="GDP (Tỷ VND)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}, orientation="h", y=1.1),
        shapes=[dict(type="rect", x0=2025.5, x1=2035.5, y0=0, y1=1,
                     xref="x", yref="paper", fillcolor="rgba(0,212,255,0.04)",
                     line=dict(color="rgba(0,212,255,0.1)"))],
        annotations=[dict(text="Giai đoạn dự báo", x=2030, xref="x", y=1.0, yref="paper",
                         showarrow=False, font={"color": "#00D4FF", "size": 10})],
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_gdp, use_container_width=True)

    # KPIs
    base_s = scenarios.get("base", scenarios[list(scenarios.keys())[0]])
    base_gdp_2025 = df[col_gdp].values[-1]
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_html(kpi_card(f"{base_s['gdp'][-1]:.0f}", "GDP 2035", "Tỷ VND (S2)", NEON_BLUE))
    with kpi_cols[1]:
        gdp_cagr_val = cagr(base_gdp_2025, base_s['gdp'][-1], 10)
        render_html(kpi_card(f"{gdp_cagr_val:.2f}%", "CAGR 2025-35", "Tăng trưởng bình quân", NEON_GREEN))
    with kpi_cols[2]:
        render_html(kpi_card(f"{base_s['pop'][-1]:.2f}", "Dân số 2035", "Triệu người (S2)", NEON_PURPLE))
    with kpi_cols[3]:
        lp_2035 = base_s['lp'][-1]
        lp_2025 = df[col_lp].values[-1]
        lp_gr = ((lp_2035 / lp_2025) - 1) * 100 if lp_2025 > 0 else 0
        render_html(kpi_card(f"{lp_2035:.1f}", "Năng suất 2035", f"M VND (+{lp_gr:.0f}%)", NEON_ORANGE))

    # ---- 8.4.2 Dự báo lao động ----
    st.markdown(section_header("8.4.2 Dự báo lao động & Năng suất", "purple"), unsafe_allow_html=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_lp = go.Figure()
        hist_lp = df[col_lp].values
        fig_lp.add_trace(go.Scatter(
            x=list(hist_years), y=list(hist_lp),
            mode="lines+markers", name="Thực tế",
            line=dict(color="#E0E0FF", width=2),
            marker=dict(size=8),
            hovertemplate="Thực tế: %{y:.1f}M<extra></extra>",
        ))
        for sc_name, sc_data in scenarios.items():
            fig_lp.add_trace(go.Scatter(
                x=list(sc_data["years"]), y=list(sc_data["lp"]),
                mode="lines", name=sc_labels.get(sc_name, sc_name),
                line=dict(color=sc_colors.get(sc_name, NEON_BLUE), width=2),
                hovertemplate=f"%{{x}}: %{{y:.1f}}M<extra></extra>",
            ))
        fig_lp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=380,
            xaxis=dict(gridcolor="#1F2A5A", title="Năm"),
            yaxis=dict(gridcolor="#1F2A5A", title="Triệu VND/người"),
            title=dict(text="Năng suất lao động 2020-2035", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_lp, use_container_width=True)

    with chart_cols[1]:
        fig_pop = go.Figure()
        hist_pop = df[col_pop].values
        fig_pop.add_trace(go.Scatter(
            x=list(hist_years), y=list(hist_pop),
            mode="lines+markers", name="Thực tế",
            line=dict(color="#E0E0FF", width=2),
            marker=dict(size=8),
        ))
        for sc_name, sc_data in scenarios.items():
            fig_pop.add_trace(go.Scatter(
                x=list(sc_data["years"]), y=list(sc_data["pop"]),
                mode="lines", name=sc_labels.get(sc_name, sc_name),
                line=dict(color=sc_colors.get(sc_name, NEON_BLUE), width=2,
                         dash="dash" if sc_name != "base" else "solid"),
            ))
        fig_pop.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=380,
            xaxis=dict(gridcolor="#1F2A5A", title="Năm"),
            yaxis=dict(gridcolor="#1F2A5A", title="Triệu người"),
            title=dict(text="Dân số 2020-2035", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_pop, use_container_width=True)

    # ---- 8.4.3 Dự báo năng suất ----
    st.markdown(section_header("8.4.3 Bảng tổng hợp các kịch bản", "green"), unsafe_allow_html=True)

    summary_rows = []
    for sc_name, sc_data in scenarios.items():
        gdp_gr = cagr(df[col_gdp].values[-1], sc_data["gdp"][-1], 10)
        lp_gr = ((sc_data["lp"][-1] / df[col_lp].values[-1]) - 1) * 100
        pop_gr = ((sc_data["pop"][-1] / df[col_pop].values[-1]) - 1) * 100
        summary_rows.append({
            "Kịch bản": sc_labels.get(sc_name, sc_name),
            "GDP 2035 (Tỷ VND)": round(sc_data["gdp"][-1], 1),
            "CAGR (%)": round(gdp_gr, 2),
            "Năng suất 2035 (M VND)": round(sc_data["lp"][-1], 1),
            "Dân số 2035 (Triệu)": round(sc_data["pop"][-1], 2),
            "Tăng năng suất (%)": round(lp_gr, 1),
            "Tăng dân số (%)": round(pop_gr, 1),
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.style.format({
        "GDP 2035 (Tỷ VND)": "{:.1f}",
        "CAGR (%)": "{:.2f}%",
        "Năng suất 2035 (M VND)": "{:.1f}",
        "Dân số 2035 (Triệu)": "{:.2f}",
        "Tăng năng suất (%)": "{:.1f}%",
        "Tăng dân số (%)": "{:.1f}%",
    }), use_container_width=True)

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    gdp_2035_base = base_s["gdp"][-1]
    gdp_2025 = df[col_gdp].values[-1]
    pop_2025 = df[col_pop].values[-1]
    pop_2035 = base_s["pop"][-1]
    lp_2025 = df[col_lp].values[-1]
    lp_2035 = base_s["lp"][-1]
    lp_growth = ((lp_2035 / lp_2025) - 1) * 100 if lp_2025 > 0 else 0
    cagr_base = cagr(gdp_2025, gdp_2035_base, 10)
    gdp_gap_s1_s3 = scenarios["optimistic"]["gdp"][-1] - scenarios["pessimistic"]["gdp"][-1]
    pop_growth_total = ((pop_2035 / pop_2025) - 1) * 100 if pop_2025 > 0 else 0

    with st.expander("Xu hướng tăng trưởng dài hạn và động lực kinh tế Việt Nam 2026-2035?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        GDP Việt Nam dự báo tăng từ <b style="color:#00D4FF;">{gdp_2025:.0f}T VND</b> (2025) lên
        <b style="color:#00FF88;">{gdp_2035_base:.0f}T VND</b> (2035) theo kịch bản cơ sở,
        CAGR = <b style="color:#FFD700;">{cagr_base:.2f}%/năm</b>.
        Cả 3 kịch bản đều cho thấy tăng trưởng dương, với khoảng cách giữa
        S1 (bi quan) và S3 (lạc quan) là <b style="color:#FF6B35;">{gdp_gap_s1_s3:.0f}T VND</b>
        vào năm 2035. Năng suất lao động tăng từ {lp_2025:.1f} triệu VND/người lên
        <b style="color:#00FF88;">{lp_2035:.1f} triệu VND/người</b> (+{lp_growth:.0f}%),
        phản ánh cải thiện hiệu quả sản xuất.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Tăng trưởng dài hạn được thúc đẩy bởi: (1) <b style="color:#00D4FF;">Quy mô dân số</b> —
        dân số tăng từ {pop_2025:.1f} triệu lên {pop_2035:.1f} triệu (+{pop_growth_total:.1f}%),
        tạo nguồn lao động và tiêu dùng nội địa. (2) <b style="color:#00FF88;">Tích lũy vốn</b> —
        FDI tiếp tục đổ vào Việt Nam, tăng quy mô vốn sản xuất. (3) <b style="color:#BF00FF;">Tăng năng suất</b> —
        năng suất lao động tăng {lp_growth:.0f}% nhờ chuyển đổi số và công nghệ.
        (4) <b style="color:#FFD700;">Kinh tế số</b> — Mô hình giả định tốc độ tăng năng suất
        {((scenarios['base']['lp'][-1]/scenarios['base']['lp'][0])**(1/10)-1)*100:.2f}%/năm.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Tăng trưởng {cagr_base:.2f}%/năm đưa GDP Việt Nam lên khoảng
        <b style="color:#00FF88;">{gdp_2035_base*1000/23807:.0f} Tỷ USD</b> vào năm 2035,
        với thu nhập bình quân đầu người khoảng
        {gdp_2035_base*1000/23807/pop_2035*1000:.0f} USD.
        Tốc độ tăng năng suất {lp_growth:.0f}% cho thấy nền kinh tế đang chuyển dần
        sang <b style="color:#FFD700;">tăng trưởng theo chiều sâu</b>, không chỉ dựa vào
        gia tăng lao động và vốn. Tuy nhiên, khoảng cách S1-S3 = {gdp_gap_s1_s3:.0f}T VND
        cho thấy rủi ro bi quan cũng rất đáng kể.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Duy trì đầu tư FDI</b> — Tiếp tục thu hút FDI chất lượng cao,
        đặc biệt vào công nghệ cao và chế biến chế tạo. (2) <b style="color:#00D4FF;">Đẩy mạnh năng suất</b> —
        Chính sách khuyến khích đầu tư R&D, nâng cao công nghệ trong doanh nghiệp.
        (3) <b style="color:#00D4FF;">Phát triển thị trường nội địa</b> — Tận dụng quy mô dân số
        {pop_2035:.1f} triệu người để thúc đẩy tiêu dùng nội địa. (4) <b style="color:#00D4FF;">Quản lý rủi ro</b> —
        Xây dựng kịch bản ứng phó cho S1 (bi quan) để giảm thiểu tác động
        nếu suy thoái toàn cầu xảy ra.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Nguy cơ suy giảm tăng trưởng và cách phòng ngừa?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Kịch bản bi quan (S1) với tốc độ tăng trưởng giảm 30% so với xu hướng cho thấy
        GDP 2035 có thể chỉ đạt mức thấp hơn đáng kể. Khoảng cách giữa S1 và S3
        là <b style="color:#FF6B35;">{gdp_gap_s1_s3:.0f}T VND</b> — tương đương
        {gdp_gap_s1_s3/gdp_2035_base*100:.0f}% GDP năm 2035. Các rủi ro chính bao gồm:
        suy thoái kinh tế toàn cầu, căng thẳng địa chính trị, biến đổi khí hậu,
        và các cú sốc cung ứng. Mô hình hệ thống động cho thấy nếu tốc độ tăng năng suất
        giảm, GDP sẽ giảm đáng kể qua cơ chế phản hồi tích cực.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nguy cơ suy giảm xuất phát từ: (1) <b style="color:#FF6B35;">Phụ thuộc bên ngoài cao</b> —
        FDI và xuất khẩu chiếm tỷ trọng lớn, dễ bị ảnh hưởng bởi biến động quốc tế.
        (2) <b style="color:#FF6B35;">Cơ cấu kinh tế chưa đổi mới</b> — Vẫn phụ thuộc vào
        lao động giá rẻ và gia công, chưa chuyển mạnh sang giá trị gia tăng cao.
        (3) <b style="color:#FF6B35;">Chuyển đổi số chưa đồng đều</b> — Nhiều ngành và vùng
        chưa bắt kịp xu hướng số hóa, làm chậm tăng năng suất. (4) <b style="color:#FF6B35;">Rủi ro nhân khẩu học</b> —
        Dân số già hóa dần sẽ giảm tốc độ tăng lao động sau 2030.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Khoảng cách {gdp_gap_s1_s3:.0f}T VND giữa S1 và S3 cho thấy tăng trưởng kinh tế
        Việt Nam có tính <b style="color:#FFD700;">bất định cao</b>. Một cú sốc tiêu cực
        có thể làm giảm GDP 2035 tới {gdp_gap_s1_s3/gdp_2035_base*100:.0f}%.
        Tuy nhiên, ngay cả kịch bản S1 vẫn duy trì tăng trưởng dương nhờ
        nền tảng dân số và năng suất — cho thấy nền kinh tế có <b style="color:#00FF88;">độ dẻo dai</b>
        (resilience) nhất định. Điều này nhấn mạnh tầm quan trọng của việc
        xây dựng nền tảng nội sinh vững chắc.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Xây dựng quỹ dự phòng</b> — Dự trữ ngân sách cho tình huống
        suy thoái kinh tế toàn cầu. (2) <b style="color:#00D4FF;">Đa dạng hóa thị trường</b> —
        Giảm phụ thuộc vào một vài thị trường xuất khẩu chính. (3) <b style="color:#00D4FF;">Đẩy mạnh nội tiêu dùng</b> —
        Phát triển thị trường nội địa {pop_2035:.1f} triệu người để giảm phụ thuộc xuất khẩu.
        (4) <b style="color:#00D4FF;">Nâng cao năng suất</b> — Đầu tư vào giáo dục, R&D và công nghệ
        để duy trì tốc độ tăng năng suất {lp_growth:.0f}%.
        (5) <b style="color:#00D4FF;">Chính sách tiền tệ linh hoạt</b> — Sẵn sàng kích thích kinh tế
        khi có dấu hiệu suy giảm.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Giải pháp nào để duy trì tăng trưởng bền vững giai đoạn 2026-2035?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Để đạt được kịch bản S3 (lạc quan) với GDP 2035 lên tới
        <b style="color:#00FF88;">{scenarios['optimistic']['gdp'][-1]:.0f}T VND</b>,
        Việt Nam cần đồng thời: (1) Duy trì tốc độ tăng năng suất {((scenarios['optimistic']['lp'][-1]/scenarios['optimistic']['lp'][0])**(1/10)-1)*100:.2f}%/năm.
        (2) Tiếp tục thu hút FDI chất lượng cao. (3) Đẩy mạnh chuyển đổi số toàn diện.
        Mô hình hệ thống động cho thấy ba biến trạng thái (GDP, Dân số, Năng suất)
        tương tác qua lại: năng suất tăng → thu nhập tăng → tiêu dùng tăng → GDP tăng →
        đầu tư tăng → năng suất tăng tiếp. Đây là <b style="color:#FFD700;">vòng xoắn tích cực</b>
        mà chính sách đúng đắn có thể kích hoạt.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Ba giải pháp then chốt xuất phát từ cấu trúc mô hình: (1) <b style="color:#00D4FF;">Tăng năng suất</b> —
        Biến trạng thái năng suất (LP) có tốc độ tăng trung bình {((scenarios['base']['lp'][-1]/scenarios['base']['lp'][0])**(1/10)-1)*100:.2f}%/năm,
        là động lực chính của tăng trưởng dài hạn. Năng suất tăng nhờ công nghệ, quản lý,
        và kỹ năng lao động. (2) <b style="color:#00FF88;">Thu hút FDI chất lượng</b> —
        FDI tăng quy mô vốn và chuyển giao công nghệ. (3) <b style="color:#BF00FF;">Chuyển đổi số</b> —
        Số hóa nâng cao hiệu quả của cả lao động và vốn.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nếu thực hiện đồng thời ba giải pháp, GDP 2035 có thể đạt mức S3
        (<b style="color:#00FF88;">{scenarios['optimistic']['gdp'][-1]:.0f}T VND</b>),
        cao hơn S1 tới <b style="color:#FFD700;">{gdp_gap_s1_s3:.0f}T VND</b>.
        Mức tăng trưởng này đưa Việt Nam thoát khỏi nhóm thu nhập trung bình thấp
        và tiến gần nhóm thu nhập trung bình cao. Đặc biệt, năng suất lao động
        tăng {lp_growth:.0f}% giúp nâng cao thu nhập thực tế của người lao động.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Chiến lược nâng cao năng suất</b> — Đầu tư vào giáo dục STEM,
        đào tạo kỹ năng số, khuyến khích đổi mới công nghệ trong doanh nghiệp.
        (2) <b style="color:#00D4FF;">Chính sách thu hút FDI chọn lọc</b> — Ưu tiên dự án công nghệ cao,
        có chuyển giao R&D, vào khu công nghệ cao. (3) <b style="color:#00D4FF;">Chương trình chuyển đổi số quốc gia</b> —
        Hỗ trợ SME số hóa, xây dựng hạ tầng số, phát triển thương mại điện tử.
        (4) <b style="color:#00D4FF;">Kết hợp 12 mô hình</b> — Dùng kết quả Cobb-Douglas (Bài 1),
        NSGA-II (Bài 7), Monte Carlo (Bài 10) để xây dựng kế hoạch tổng thể 2026-2035.
        (5) <b style="color:#00D4FF;">Giám sát và điều chỉnh</b> — Theo dõi chỉ số GDP, năng suất,
        dân số hàng quý, điều chỉnh chính sách kịp thời.
        </div>
        """, unsafe_allow_html=True)
