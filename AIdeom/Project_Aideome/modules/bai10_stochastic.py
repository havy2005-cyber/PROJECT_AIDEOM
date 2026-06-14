import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_macro
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.helpers import cagr, growth_rate

np.random.seed(42)


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Tối ưu trong điều kiện bất định sử dụng mô phỏng Monte Carlo.",
        "Xây dựng khoảng tin cậy 90% cho GDP 2030.",
        "So sánh 3 kịch bản: Bi quan, Cơ sở, Lạc quan.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)
    st.markdown("- GDP (Tỷ VND): Tổng sản phẩm quốc nội 2020-2025")
    st.markdown("- FDI (Tỷ USD): Vốn đầu tư trực tiếp nước ngoài")
    st.markdown("- Xuất khẩu (Tỷ USD): Giá trị xuất khẩu hàng hóa")
    st.markdown("- Kinh tế số (% GDP): Tỷ trọng kinh tế số")
    st.markdown("- Lạm phát (%): Chỉ số giá tiêu dùng CPI")

    df = load_macro()
    if df is None:
        st.warning("Không tìm thấy dữ liệu vietnam_macro_2020_2025.csv.")
        return

    col_gdp = "GDP_trillion_VND"
    col_growth = "GDP_growth_pct"
    col_fdi = "FDI_disbursed_billion_USD"
    col_digital = "digital_economy_share_GDP_pct"
    col_export = "exports_billion_USD"
    col_inflation = "inflation_CPI_pct"

    gdp = df[col_gdp].values
    growth_hist = df[col_growth].values
    fdi_hist = df[col_fdi].values
    digital_hist = df[col_digital].values
    export_hist = df[col_export].values
    inflation_hist = df[col_inflation].values

    mu_g = np.mean(growth_hist)
    sigma_g = np.std(growth_hist)
    mu_f = np.mean(fdi_hist)
    sigma_f = np.std(fdi_hist)
    mu_e = np.mean(export_hist)
    sigma_e = np.std(export_hist)
    mu_d = np.mean(digital_hist)
    sigma_d = np.std(digital_hist)
    mu_i = np.mean(inflation_hist)
    sigma_i = np.std(inflation_hist)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình
    st.markdown(section_header("Cấu hình kịch bản & Monte Carlo", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(3)
    n_sim = cfg_cols[0].slider("Số lần mô phỏng", 500, 5000, 1000, 100)
    n_years = cfg_cols[1].slider("Số năm dự báo", 3, 10, 5)
    conf_level = cfg_cols[2].slider("Mức độ tin cậy", 0.80, 0.99, 0.90, 0.01)

    # ---- 10.4.1 Kịch bản tốt ----
    st.markdown('<div class="section-header-purple">10.4.1 Kịch bản Tốt nhất (Top 5%)</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    sim_gdp = np.zeros((n_sim, n_years))
    sim_fdi = np.zeros((n_sim, n_years))
    sim_export = np.zeros((n_sim, n_years))
    sim_digital = np.zeros((n_sim, n_years))

    for s in range(n_sim):
        if s % 200 == 0:
            progress_bar.progress(int(s / n_sim * 100))
        gdp_t = gdp[-1]
        gdp_gr_mean = mu_g / 100.0
        for t in range(n_years):
            g_new = np.random.normal(gdp_gr_mean, sigma_g / 100.0)
            f_new = np.random.normal(mu_f, sigma_f)
            e_new = np.random.normal(mu_e, sigma_e)
            d_new = min(max(np.random.normal(mu_d, sigma_d), 5.0), 50.0)
            gdp_t = gdp_t * (1 + g_new)
            sim_gdp[s, t] = gdp_t
            sim_fdi[s, t] = max(f_new, 0.0)
            sim_export[s, t] = max(e_new, 0.0)
            sim_digital[s, t] = d_new

    progress_bar.progress(100)
    status_text.success(f"Hoàn thành {n_sim} lần mô phỏng!")

    years_sim = np.arange(2026, 2026 + n_years)
    gdp_mean = np.mean(sim_gdp, axis=0)
    gdp_p5 = np.percentile(sim_gdp, (1 - conf_level) / 2 * 100, axis=0)
    gdp_p95 = np.percentile(sim_gdp, (1 + conf_level) / 2 * 100, axis=0)
    gdp_p25 = np.percentile(sim_gdp, 25, axis=0)
    gdp_p75 = np.percentile(sim_gdp, 75, axis=0)
    fdi_mean = np.mean(sim_fdi, axis=0)
    exp_mean = np.mean(sim_export, axis=0)
    digital_mean = np.mean(sim_digital, axis=0)

    gdp_final_mean = gdp_mean[-1]
    gdp_cagr_mean = cagr(gdp[-1], gdp_final_mean, n_years)
    gdp_std = np.std(sim_gdp[:, -1])

    pct_reach = np.mean(sim_gdp[:, -1] > gdp[-1] * 1.5) * 100

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_html(kpi_card(f"{gdp_final_mean:.0f}T", "GDP TB 2030", "Tỷ VND", NEON_BLUE))
    with kpi_cols[1]:
        render_html(kpi_card(f"{gdp_cagr_mean:.2f}%", "CAGR TB", "Năm", NEON_GREEN))
    with kpi_cols[2]:
        render_html(kpi_card(f"{gdp_std:.0f}T", "Độ lệch chuẩn", "GDP cuối năm", NEON_ORANGE))
    with kpi_cols[3]:
        render_html(kpi_card(f"{pct_reach:.1f}%", "Tỷ lệ > 150%", "Mức tăng 50%", NEON_PURPLE))

    # Kịch bản tốt nhất (top 5%)
    top5_gdp = np.percentile(sim_gdp[:, -1], 95)
    top5_df = pd.DataFrame({
        "Năm": years_sim,
        "GDP TB (Tỷ VND)": np.round(gdp_mean, 1),
        f"P{int((1-conf_level)/2*100)} (Tỷ VND)": np.round(gdp_p5, 1),
        "P50 (Tỷ VND)": np.round(np.percentile(sim_gdp, 50, axis=0), 1),
        f"P{int((1+conf_level)/2*100)} (Tỷ VND)": np.round(gdp_p95, 1),
    })
    st.dataframe(top5_df.style.format({
        "GDP TB (Tỷ VND)": "{:.1f}",
        f"P{int((1-conf_level)/2*100)} (Tỷ VND)": "{:.1f}",
        "P50 (Tỷ VND)": "{:.1f}",
        f"P{int((1+conf_level)/2*100)} (Tỷ VND)": "{:.1f}",
    }), width='stretch')

    # ---- 10.4.2 Kịch bản Trung bình ----
    st.markdown(section_header("10.4.2 Kịch bản Trung bình (P50)", "purple"), unsafe_allow_html=True)

    scenario_df = pd.DataFrame({
        "Năm": years_sim,
        "GDP TB (Tỷ VND)": np.round(gdp_mean, 1),
        "GDP (Tỷ USD)": np.round(gdp_mean * 1000 / 23807, 1),
        "FDI TB (B USD)": np.round(fdi_mean, 1),
        "Xuất khẩu TB (B USD)": np.round(exp_mean, 1),
        "Kinh tế số TB (%)": np.round(digital_mean, 1),
    })
    st.dataframe(scenario_df.style.format({
        "GDP TB (Tỷ VND)": "{:.1f}",
        "GDP (Tỷ USD)": "{:.1f}",
        "FDI TB (B USD)": "{:.1f}",
        "Xuất khẩu TB (B USD)": "{:.1f}",
        "Kinh tế số TB (%)": "{:.1f}",
    }), width='stretch')

    # ---- 10.4.3 Kịch bản Xấu ----
    st.markdown(section_header("10.4.3 Kịch bản Xấu nhất (Bottom 5%)", "orange"), unsafe_allow_html=True)

    bot5_gdp = np.percentile(sim_gdp[:, -1], 5)
    stat_rows = []
    for t in range(n_years):
        stat_rows.append({
            "Năm": int(years_sim[t]),
            "Giá trị TB": round(gdp_mean[t], 1),
            "Độ lệch chuẩn": round(np.std(sim_gdp[:, t]), 1),
            "Min": round(np.min(sim_gdp[:, t]), 1),
            "Max": round(np.max(sim_gdp[:, t]), 1),
            f"P{int((1-conf_level)/2*100)}": round(np.percentile(sim_gdp[:, t], (1-conf_level)/2*100), 1),
            "P50": round(np.percentile(sim_gdp[:, t], 50), 1),
            f"P{int((1+conf_level)/2*100)}": round(np.percentile(sim_gdp[:, t], (1+conf_level)/2*100), 1),
        })
    stat_df = pd.DataFrame(stat_rows)
    st.dataframe(stat_df, width='stretch')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(years_sim), y=list(gdp_mean),
        mode="lines+markers", name="Trung bình",
        line=dict(color=NEON_BLUE, width=3),
        marker=dict(size=8),
        hovertemplate="TB: %{y:.1f}T<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(years_sim) + list(years_sim)[::-1],
        y=list(gdp_p5) + list(gdp_p95)[::-1],
        fill="toself", fillcolor="rgba(0,212,255,0.1)",
        line=dict(color="rgba(0,212,255,0.3)", width=1),
        name=f"Khoảng {int(conf_level*100)}%",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(years_sim) + list(years_sim)[::-1],
        y=list(gdp_p25) + list(gdp_p75)[::-1],
        fill="toself", fillcolor="rgba(0,212,255,0.2)",
        line=dict(color="rgba(0,212,255,0.5)", width=1),
        name="IQR 50%",
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=420,
        xaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="Năm"),
        yaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="GDP (Tỷ VND)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        shapes=[dict(type="line", x0=2025.5, x1=2025.5, y0=0, y1=1,
                     xref="x", yref="paper", line=dict(color="white", width=1, dash="dash"))],
        annotations=[dict(text="Thực tế", x=2025, xref="x", y=0.95, yref="paper",
                         showarrow=False, font={"color": "#8090C0", "size": 10})],
        title=dict(text=f"Monte Carlo: GDP dự báo (n={n_sim}, {int(conf_level*100)}% CI)", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig, width='stretch')

    # Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sim_gdp[:, -1],
        nbinsx=50,
        marker_color=NEON_BLUE,
        opacity=0.75,
        hovertemplate="GDP: %{x:.0f}T<br>Tần số: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=gdp_mean[-1], line_dash="dash", line_color=NEON_GREEN,
                       annotation_text=f"TB: {gdp_mean[-1]:.0f}T")
    fig_hist.add_vline(x=gdp_p5[-1], line_dash="dot", line_color=NEON_ORANGE,
                       annotation_text=f"P{int((1-conf_level)/2*100)}: {gdp_p5[-1]:.0f}T")
    fig_hist.add_vline(x=gdp_p95[-1], line_dash="dot", line_color=NEON_ORANGE,
                       annotation_text=f"P{int((1+conf_level)/2*100)}: {gdp_p95[-1]:.0f}T")
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=350,
        xaxis=dict(gridcolor="#1F2A5A", title="GDP cuối năm (Tỷ VND)"),
        yaxis=dict(gridcolor="#1F2A5A", title="Số lần mô phỏng"),
        title=dict(text="Phân bổ GDP cuối năm 2030", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_hist, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    p5_val = gdp_p5[-1]
    p95_val = gdp_p95[-1]
    conf_label = int(conf_level * 100)
    volatility_pct = (gdp_std / gdp_final_mean) * 100
    var_coef = gdp_std / gdp_final_mean
    prob_reach_target = np.mean(sim_gdp[:, -1] > gdp[-1] * 1.5) * 100

    with st.expander("GDP 2030 nằm trong khoảng nào và rủi ro biến động ra sao?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Với <b style="color:#00D4FF;">{n_sim} lần mô phỏng Monte Carlo</b>, GDP 2030 có phân bổ:
        <br>- <b style="color:#00FF88;">Giá trị trung bình:</b> <b style="color:#FFD700;">{gdp_final_mean:.0f} Tỷ VND</b>
        (<b style="color:#00D4FF;">{gdp_final_mean*1000/23807:.1f} Tỷ USD</b>)
        <br>- <b style="color:#00FF88;">Khoảng {conf_label}% tin cậy:</b> [{p5_val:.0f} - {p95_val:.0f}] Tỷ VND
        <br>- <b style="color:#00FF88;">Độ lệch chuẩn:</b> <b style="color:#FF6B35;">{gdp_std:.0f} Tỷ VND</b>
        <br>- <b style="color:#00FF88;">Hệ số biến thiên:</b> <b style="color:#FFD700;">{var_coef:.2%}</b>
        <br>- <b style="color:#00FF88;">Xác suất vượt mục tiêu +50%:</b> <b style="color:#00D4FF;">{prob_reach_target:.1f}%</b>
        <br>- <b style="color:#00FF88;">CAGR trung bình:</b> <b style="color:#00FF88;">{gdp_cagr_mean:.2f}%/năm</b>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Biến động GDP đến từ {n_sim} kịch bản ngẫu nhiên với các tham số:
        (1) <b style="color:#00D4FF;">Tăng trưởng GDP</b> — trung bình {mu_g:.2f}%/năm, độ lệch chuẩn {sigma_g:.2f}%.
        (2) <b style="color:#00FF88;">FDI</b> — trung bình {mu_f:.1f}B USD, biến động {sigma_f:.1f}B USD.
        (3) <b style="color:#BF00FF;">Xuất khẩu</b> — trung bình {mu_e:.1f}B USD, biến động {sigma_e:.1f}B USD.
        (4) <b style="color:#FFD700;">Kinh tế số</b> — trung bình {mu_d:.1f}%, biến động {sigma_d:.1f}%.
        Khoảng cách {p95_val - p5_val:.0f}T VND phản ánh mức độ bất định
        của nền kinh tế Việt Nam trong giai đoạn 2026-2030.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Khoảng tin cậy {conf_label}% = [{p5_val:.0f} - {p95_val:.0f}] Tỷ VND cho thấy
        <b style="color:#FFD700;">{conf_label}% khả năng</b> GDP 2030 sẽ nằm trong khoảng này.
        Hệ số biến thiên = {var_coef:.2%} phản ánh mức độ rủi ro:
        {"tương đối thấp" if var_coef < 0.15 else "trung bình" if var_coef < 0.25 else "cao"}.
        Xác suất vượt mục tiêu +50% GDP = <b style="color:#00FF88;">{prob_reach_target:.1f}%</b>.
        Kịch bản xấu nhất (P{int((1-conf_level)/2*100)}) có thể thấp hơn trung bình
        tới <b style="color:#FF6B35;">{(gdp_final_mean - p5_val)/gdp_final_mean*100:.0f}%</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Xây dựng kịch bản phòng trừ</b> — Lập kế hoạch ngân sách
        cho cả kịch bản P{int((1-conf_level)/2*100)} = {p5_val:.0f}T VND. (2) <b style="color:#00D4FF;">Quỹ dự phòng rủi ro</b> —
        Dự trữ 5-10% ngân sách cho tình huống xấu. (3) <b style="color:#00D4FF;">Đa dạng hóa đầu tư</b> —
        Giảm phụ thuộc vào một vài thị trường hoặc ngành duy nhất.
        (4) <b style="color:#00D4FF;">Cơ chế giám sát sớm</b> — Theo dõi chỉ số dẫn đầu
        (leading indicators) để phát hiện sớm dấu hiệu suy giảm.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Kịch bản xấu nhất (P{int((1-conf_level)/2*100)}) nguy hiểm đến mức nào?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Kịch bản bottom 5% (P{int((1-conf_level)/2*100)} = {p5_val:.0f}T VND) phản ánh
        GDP giảm tăng trưởng đáng kể so với trung bình. Chênh lệch so với kịch bản trung bình:
        <b style="color:#FF6B35;">-{gdp_final_mean - p5_val:.0f} Tỷ VND</b>
        (tương đương <b style="color:#FFD700;">-{(gdp_final_mean - p5_val)/gdp_final_mean*100:.1f}%</b> GDP).
        Đây là mức GDP tương đương với kịch bản S1 (bi quan) trong Bài 8,
        phản ánh tình huống suy thoái kinh tế toàn cầu hoặc cú sốc nội sinh nghiêm trọng.
        Tuy nhiên, ngay cả P{int((1-conf_level)/2*100)} vẫn cao hơn GDP 2025
        (<b style="color:#00FF88;">+{p5_val/gdp[-1]*100 - 100:.0f}%</b> so với năm 2025),
        cho thấy nền kinh tế có độ dẻo dai nhất định.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Kịch bản xấu nhất xảy ra khi: (1) <b style="color:#FF6B35;">Tăng trưởng GDP âm hoặc rất thấp</b> —
        do suy thoái kinh tế toàn cầu, giảm cầu xuất khẩu, và đầu tư FDI giảm mạnh.
        (2) <b style="color:#FF6B35;">FDI giảm sâu</b> — nhà đầu tư rút vốn hoặc không đầu tư mới
        do rủi ro địa chính trị hoặc suy thoái quốc tế. (3) <b style="color:#FF6B35;">Xuất khẩu suy giảm</b> —
        thị trường xuất khẩu chính (Mỹ, EU, Trung Quốc) suy giảm.
        (4) <b style="color:#FF6B35;">Kinh tế số chậm lại</b> — đầu tư số hóa giảm do thiếu vốn và niềm tin.
        Monte Carlo mô phỏng các biến này ngẫu nhiên theo phân bổ lịch sử.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        P{int((1-conf_level)/2*100)} = {p5_val:.0f}T VND cho thấy <b style="color:#FFD700;">{100-conf_label/10:.0f}% khả năng</b>
        nền kinh tế sẽ không giảm dưới mức này. Tuy nhiên, đây vẫn là mức
        <b style="color:#FF6B35;">thấp hơn kịch bản cơ sở tới {(gdp_final_mean - p5_val)/gdp_final_mean*100:.0f}%</b>.
        Về mặt thu nhập, GDP per capita tại P{int((1-conf_level)/2*100)} sẽ thấp hơn
        kỳ vọng, ảnh hưởng đời sống người dân và khả năng thoát nghèo.
        Đặc biệt, nếu GDP per capita tăng chậm, Việt Nam có nguy cơ
        <b style="color:#FF6B35;">mắc kẹt trong bẫy thu nhập trung bình</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Xây dựng kế hoạch dự phòng</b> — Lập ngân sách ứng phó
        cho GDP P{int((1-conf_level)/2*100)} = {p5_val:.0f}T VND. (2) <b style="color:#00D4FF;">Chính sách kích thích kinh tế</b> —
        Sẵn sàng gói kích thích tài khóa và tiền tệ nếu tăng trưởng giảm sâu.
        (3) <b style="color:#00D4FF;">Đa dạng hóa thị trường</b> — Giảm phụ thuộc vào
        Mỹ, EU, Trung Quốc bằng cách đẩy mạnh xuất khẩu sang ASEAN, Ấn Độ, Trung Đông.
        (4) <b style="color:#00D4FF;">Tăng cường dự trữ ngoại hối</b> — Đảm bảo khả năng
        ứng phó với cú sốc tài chính. (5) <b style="color:#00D4FF;">Phát triển thị trường nội địa</b> —
        Tăng tiêu dùng trong nước để giảm phụ thuộc xuất khẩu.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Chính sách nào giúp giảm rủi ro và tăng xác suất đạt mục tiêu?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Xác suất đạt mục tiêu +50% GDP (so với 2025) là <b style="color:#FFD700;">{prob_reach_target:.1f}%</b>,
        nghĩa là <b style="color:#FFD700;">khoảng {int(prob_reach_target/100*n_sim)}/{n_sim} lần mô phỏng</b>
        vượt mục tiêu. Mục tiêu +50% GDP tương đương GDP 2030 = {gdp[-1]*1.5:.0f}T VND.
        Để tăng xác suất này, cần tập trung vào các biến có biến động lớn nhất:
        tăng trưởng GDP (σ = {sigma_g:.2f}%), xuất khẩu (σ = {sigma_e:.1f}B USD),
        và kinh tế số. Các biến này phản ánh 3 động lực chính:
        tăng trưởng nội sinh, hội nhập quốc tế, và chuyển đổi số.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Xác suất đạt mục tiêu thấp đến từ: (1) <b style="color:#FF6B35;">Biến động tăng trưởng GDP</b> —
        σ = {sigma_g:.2f}% cho thấy tốc độ tăng trưởng có thể dao động đáng kể,
        ảnh hưởng trực tiếp đến quỹ đạo GDP. (2) <b style="color:#FF6B35;">FDI biến động mạnh</b> —
        σ = {sigma_f:.1f}B USD trên trung bình {mu_f:.1f}B USD cho thấy dòng vốn FDI
        không ổn định, phụ thuộc nhiều vào môi trường quốc tế. (3) <b style="color:#FF6B35;">Kinh tế số chưa đồng đều</b> —
        mức {mu_d:.1f}% GDP cho thấy vẫn còn dư địa tăng trưởng nhưng cần đầu tư mạnh.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Xác suất {prob_reach_target:.1f}% cho thấy mục tiêu +50% GDP là <b style="color:#FFD700;">khả thi
        nhưng không chắc chắn</b>. Để tăng xác suất này, cần đồng thời:
        (1) <b style="color:#00FF88;">Tăng trung bình tăng trưởng</b> — thúc đẩy năng suất, FDI, và xuất khẩu.
        (2) <b style="color:#00FF88;">Giảm biến động</b> — ổn định chính sách, đa dạng hóa thị trường.
        (3) <b style="color:#00FF88;">Phát triển nội sinh</b> — giảm phụ thuộc vào biến số bên ngoài.
        Mô hình Monte Carlo cho thấy rủi ro không chỉ đến từ quy mô mà còn từ
        <b style="color:#FFD700;">sự bất định</b> của các biến kinh tế.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Chiến lược tăng trưởng nội sinh</b> — Đầu tư vào giáo dục,
        R&D, và hạ tầng để tăng trung bình tăng trưởng. (2) <b style="color:#00D4FF;">Ổn định hóa chính sách</b> —
        Giảm biến động hành chính, tạo môi trường kinh doanh ổn định để giảm σ tăng trưởng.
        (3) <b style="color:#00D4FF;">Đa dạng hóa đối tác</b> — Giảm phụ thuộc vào một vài thị trường
        xuất khẩu hoặc nguồn FDI. (4) <b style="color:#00D4FF;">Phát triển kinh tế số</b> —
        Tăng từ {mu_d:.1f}% lên 35-40% GDP để tạo động lực tăng trưởng mới.
        (5) <b style="color:#00D4FF;">Chính sách đầu tư công chiến lược</b> — Đầu tư vào
        giao thông, năng lượng, số hóa để tạo nền tảng cho tăng trưởng bền vững.
        </div>
        """, unsafe_allow_html=True)
