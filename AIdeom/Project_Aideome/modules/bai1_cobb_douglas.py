import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_macro
from utils.helpers import safe_float, mean_abs_pct_error, growth_rate, cagr, format_percent
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.charts import line_chart, bar_chart, scatter_chart

try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
def cobb_douglas_fit(Y, L, K, D):
    if not HAS_SKLEARN:
        return None
    lnY = np.log(Y)
    lnL = np.log(L)
    lnK = np.log(K)
    lnD = np.log(D)
    t = np.arange(len(Y))
    X = np.column_stack([np.ones(len(Y)), lnL, lnK, lnD, t])
    model = LinearRegression(fit_intercept=False)
    model.fit(X, lnY)
    residuals = lnY - model.predict(X)
    coefs = model.coef_
    return {
        "A": np.exp(coefs[0]),
        "alpha": coefs[1],
        "beta": coefs[2],
        "gamma": coefs[3],
        "delta": coefs[4] * 100,
        "r_squared": model.score(X, lnY),
        "sigma": np.std(residuals),
        "y_pred": np.exp(model.predict(X)),
    }


def growth_accounting(Y, L, K, D, alpha, beta, gamma):
    gY = np.diff(np.log(Y))
    gL = np.diff(np.log(L))
    gK = np.diff(np.log(K))
    gD = np.diff(np.log(D))
    contrib_L = alpha * gL
    contrib_K = beta * gK
    contrib_D = gamma * gD
    tfp = gY - contrib_L - contrib_K - contrib_D
    return {
        "period": np.arange(1, len(Y)),
        "gY": gY * 100,
        "contrib_L": contrib_L * 100,
        "contrib_K": contrib_K * 100,
        "contrib_D": contrib_D * 100,
        "TFP": tfp * 100,
    }


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Ước lượng TFP (Năng suất nhân tố tổng hợp) từ hàm sản xuất Cobb-Douglas mở rộng.",
        "Phân rã tăng trưởng GDP thành các thành phần: Lao động, Vốn, Kinh tế số và TFP.",
        "Dự báo GDP giai đoạn 2026–2030 dựa trên mô hình đã ước lượng.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)

    df = load_macro()
    if df is None:
        st.warning("Không tìm thấy dữ liệu vietnam_macro_2020_2025.csv. Vui lòng kiểm tra thư mục data.")
        return

    data_info = [
        "GDP (Tỷ VND): Tổng sản phẩm quốc nội danh nghĩa",
        "Lao động (Triệu người): Dân số trong độ tuổi lao động",
        "FDI (Tỷ USD): Vốn đầu tư trực tiếp nước ngoài giải ngân",
        "Kinh tế số (% GDP): Tỷ trọng kinh tế số trong GDP",
        "Năng suất lao động (Triệu VND/người): Tính thông qua GDP / dân số",
    ]
    for d in data_info:
        st.markdown(f"- {d}")

    col_gdp = "GDP_trillion_VND"
    col_pop = "population_million"
    col_fdi = "FDI_disbursed_billion_USD"
    col_digital = "digital_economy_share_GDP_pct"
    col_lp = "labor_productivity_million_VND"

    gdp = df[col_gdp].values.astype(float)
    labor = df[col_pop].values.astype(float) * 1e6
    fdi = df[col_fdi].values.astype(float)
    digital = df[col_digital].values.astype(float)
    lp = df[col_lp].values.astype(float)
    years = df["year"].values.astype(int)
    n = len(years)

    capital_proxy = np.cumsum(fdi * 1e9 / (1e6 * df[col_pop].values))
    capital_proxy = capital_proxy / capital_proxy[0] * gdp[0] * 0.3

    result = cobb_douglas_fit(gdp, labor, capital_proxy, digital)
    if result is None:
        st.error("Không thể ước lượng mô hình. Vui lòng cài đặt scikit-learn.")
        return

    alpha = result["alpha"]
    beta = result["beta"]
    gamma = result["gamma"]
    A_param = result["A"]
    delta_param = result["delta"] / 100
    r2 = result["r_squared"]
    y_pred_fitted = result["y_pred"]
    mape_fit = mean_abs_pct_error(gdp, y_pred_fitted)

    ga = growth_accounting(gdp, labor, capital_proxy, digital, alpha, beta, gamma)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # ---- 1.4.1 Ước lượng TFP ----
    st.markdown('<div class="section-header-purple">1.4.1 Ước lượng TFP</div>', unsafe_allow_html=True)

    kpi_cols = st.columns(5)
    avg_tfp = np.mean(ga["TFP"]) if len(ga["TFP"]) > 0 else 0
    tfp_last = ga["TFP"][-1] if len(ga["TFP"]) > 0 else 0
    gdp_cagr = cagr(gdp[0], gdp[-1], n - 1)

    with kpi_cols[0]:
        render_html(kpi_card(f"{r2:.4f}", "R²", "Độ tin cậy mô hình", NEON_BLUE))
    with kpi_cols[1]:
        render_html(kpi_card(f"{mape_fit:.2f}%", "MAPE", "Sai số trung bình", NEON_PURPLE))
    with kpi_cols[2]:
        render_html(kpi_card(f"{avg_tfp:.2f}%", "TFP TB/Năm", "Năng suất toàn phần", NEON_GREEN))
    with kpi_cols[3]:
        render_html(kpi_card(f"{gdp_cagr:.2f}%", "CAGR GDP", f"{years[0]}–{years[-1]}", NEON_ORANGE))
    with kpi_cols[4]:
        render_html(kpi_card(f"{delta_param*100:.3f}%", "δ/Năm", "Hệ số công nghệ", NEON_BLUE))

    # Bảng hệ số
    coef_df = pd.DataFrame({
        "Hệ số": ["α (Lao động)", "β (Vốn)", "γ (Kinh tế số)", "δ (Công nghệ)", "A (Hằng số)", "R²", "MAPE"],
        "Giá trị": [f"{alpha:.4f}", f"{beta:.4f}", f"{gamma:.4f}", f"{delta_param*100:.4f}%/năm", f"{A_param:.6f}", f"{r2:.4f}", f"{mape_fit:.2f}%"],
        "Diễn giải": [
            "Mức độ sinh lời theo lao động",
            "Mức độ sinh lời theo vốn (FDI + đầu tư nội địa)",
            "Tác động của kinh tế số đến GDP",
            "Tăng trưởng công nghệ/năm",
            "Hệ số nhân tổng hợp",
            f"Mô hình giải thích {r2*100:.1f}% biến động GDP",
            "Sai số phần trăm tuyệt đối trung bình",
        ]
    })
    st.dataframe(coef_df, use_container_width=True)

    # Hàm Cobb-Douglas
    render_html(glass_card(f"""
    <div class="section-header-purple" style="margin-bottom:0.5rem;">Phương trình hàm sản xuất</div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; color:#E0E0FF; line-height:1.8; text-align:center;">
        <b>Y = A × L<sup>α</sup> × K<sup>β</sup> × D<sup>γ</sup> × e<sup>δt</sup></b><br>
        <span style="color:#8090C0; font-size:0.9rem;">
            Y = {A_param:.2f} × L<sup>{alpha:.3f}</sup> × K<sup>{beta:.3f}</sup> × D<sup>{gamma:.3f}</sup> × e<sup>{delta_param*100:.3f}%×t</sup>
        </span><br><br>
        <b>Điều kiện hiệu quả:</b> α + β + γ = {alpha+beta+gamma:.4f}
        {'(Gần quy mô không đổi ✓)' if abs(alpha+beta+gamma-1) < 0.15 else '(Thoáng giảm)'}
    </div>
    """))

    # ---- 1.4.2 Dự báo GDP & MAPE ----
    st.markdown(section_header("1.4.2 Dự báo GDP & MAPE", "purple"), unsafe_allow_html=True)

    forecast_years = np.arange(2026, 2031)
    n_forecast = len(forecast_years)

    labor_forecast = np.interp(forecast_years, years, df[col_pop].values.astype(float)) * (1 + 0.008) ** np.arange(1, n_forecast + 1)
    labor_all = np.concatenate([df[col_pop].values.astype(float) * 1e6, labor_forecast * 1e6])

    K_all = np.zeros(n + n_forecast)
    K_all[:n] = capital_proxy
    for i in range(n_forecast):
        K_all[n + i] = K_all[n + i - 1] * 1.08

    D_all = np.zeros(n + n_forecast)
    D_all[:n] = digital
    for i in range(n_forecast):
        D_all[n + i] = min(D_all[n + i - 1] * 1.05, 40.0)

    y_forecast = np.zeros(n + n_forecast)
    for t in range(n + n_forecast):
        y_forecast[t] = A_param * (labor_all[t] ** alpha) * (K_all[t] ** beta) * (D_all[t] ** gamma) * np.exp(delta_param * t)

    gdp_billion_actual = gdp * 1000 / 23807
    gdp_billion_fitted = y_pred_fitted * 1000 / 23807
    gdp_billion_forecast = y_forecast[n:] * 1000 / 23807

    forecast_df = pd.DataFrame({
        "Năm": forecast_years,
        "GDP dự báo (Tỷ VND)": np.round(y_forecast[n:], 1),
        "GDP (Tỷ USD)": np.round(gdp_billion_forecast, 1),
        "Tăng trưởng (%)": [growth_rate(y_forecast[n + i - 1], y_forecast[n + i]) if i > 0 else 0 for i in range(n_forecast)],
    })

    all_years = np.concatenate([years, forecast_years])
    all_gdp = np.concatenate([y_pred_fitted, y_forecast[n:]])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=list(years), y=list(gdp),
        mode="lines+markers+text", name="GDP thực tế",
        line=dict(color="#E0E0FF", width=3),
        marker=dict(size=10, symbol="circle"),
        text=[f"{v:.0f}T" for v in gdp],
        textposition="top center", textfont={"color": "#E0E0FF", "size": 10},
        hovertemplate="Thực tế %{x}: %{y:.1f}T<br>%{customdata:.2f}%<extra></extra>",
        customdata=df["GDP_growth_pct"].values,
    ))
    fig1.add_trace(go.Scatter(
        x=list(years), y=list(y_pred_fitted),
        mode="lines", name="GDP ước lượng",
        line=dict(color=NEON_PURPLE, width=2, dash="dot"),
        hovertemplate="Ước lượng %{x}: %{y:.1f}T<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=list(forecast_years), y=list(y_forecast[n:]),
        mode="lines+markers+text", name="GDP dự báo",
        line=dict(color=NEON_GREEN, width=2.5, dash="dash"),
        marker=dict(size=10, color=NEON_GREEN),
        text=[f"{v:.0f}T" for v in y_forecast[n:]],
        textposition="top center", textfont={"color": NEON_GREEN, "size": 10},
        hovertemplate="Dự báo %{x}: %{y:.1f}T<extra></extra>",
    ))
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=380,
        xaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A"),
        yaxis=dict(gridcolor="#1F2A5A", linecolor="#2A3A8A", title="GDP (Tỷ VND)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}, orientation="h", y=1.1),
        shapes=[dict(type="rect", x0=2025.5, x1=2030.5, y0=0, y1=1,
                     xref="x", yref="paper", fillcolor="rgba(0,255,136,0.05)",
                     line=dict(color="rgba(0,255,136,0.2)"))],
        annotations=[dict(text="Giai đoạn dự báo", x=2028, xref="x", y=1.0, yref="paper",
                         showarrow=False, font={"color": NEON_GREEN, "size": 10})],
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig1, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(forecast_df.style.format({
            "GDP dự báo (Tỷ VND)": "{:.1f}",
            "GDP (Tỷ USD)": "{:.1f}",
            "Tăng trưởng (%)": "{:.2f}%",
        }), width='stretch')
    with c2:
        mape_fc = mean_abs_pct_error(np.full(n_forecast, gdp[-1]), y_forecast[n:])
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:#B0C0E0; line-height:1.8;">
            <b style="color:#00D4FF;">MAPE giai đoạn ước lượng:</b> {mape_fit:.2f}%<br>
            <b style="color:#00D4FF;">GDP 2025:</b> {gdp[-1]:.1f} Tỷ VND<br>
            <b style="color:#00D4FF;">GDP 2030 dự báo:</b> {y_forecast[-1]:.1f} Tỷ VND<br>
            <b style="color:#00D4FF;">CAGR 2025-2030:</b> {cagr(gdp[-1], y_forecast[-1], n_forecast):.2f}%/năm<br>
            <b style="color:#00D4FF;">GDP 2030 (USD):</b> {y_forecast[-1]*1000/23807:.1f} Tỷ USD<br>
            <b style="color:#00D4FF;">R² mô hình:</b> {r2:.4f} ({'Rất tốt' if r2 > 0.95 else 'Tốt' if r2 > 0.85 else 'Trung bình'})
        </div>
        """, unsafe_allow_html=True)

    # ---- 1.4.3 Growth Accounting ----
    st.markdown(section_header("1.4.3 Phân rã tăng trưởng GDP (Growth Accounting)", "green"), unsafe_allow_html=True)

    contrib_df = pd.DataFrame({
        "Năm": [int(y) for y in ga["period"] + years[0]],
        "Tăng trưởng GDP (%)": np.round(ga["gY"], 2),
        "Lao động (%)": np.round(ga["contrib_L"], 2),
        "Vốn (%)": np.round(ga["contrib_K"], 2),
        "Kinh tế số (%)": np.round(ga["contrib_D"], 2),
        "TFP (%)": np.round(ga["TFP"], 2),
    })
    st.dataframe(contrib_df.style.format({
        "Tăng trưởng GDP (%)": "{:.2f}",
        "Lao động (%)": "{:.2f}",
        "Vốn (%)": "{:.2f}",
        "Kinh tế số (%)": "{:.2f}",
        "TFP (%)": "{:.2f}",
    }), width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[int(y) for y in ga["period"] + years[0]],
        y=list(ga["gY"]),
        name="Tăng trưởng GDP",
        marker_color=NEON_BLUE, opacity=0.6,
        hovertemplate="Tăng trưởng %{x}: %{y:.2f}%<extra></extra>",
    ))
    layers = [
        ("Lao động", ga["contrib_L"], NEON_GREEN),
        ("Vốn", ga["contrib_K"], NEON_PURPLE),
        ("Kinh tế số", ga["contrib_D"], NEON_ORANGE),
        ("TFP", ga["TFP"], "#FFD700"),
    ]
    for name, vals, color in layers:
        fig2.add_trace(go.Scatter(
            x=[int(y) for y in ga["period"] + years[0]],
            y=list(vals),
            mode="lines+markers", name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=6),
            hovertemplate=f"{name} %{{x}}: %{{y:.2f}}%<extra></extra>",
        ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=380,
        xaxis=dict(gridcolor="#1F2A5A", title="Năm"),
        yaxis=dict(gridcolor="#1F2A5A", title="Đóng góp (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}, orientation="h", y=1.15),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig2, width='stretch')

    # ---- 1.4.4 Dự báo GDP 2030 ----
    st.markdown(section_header("1.4.4 Dự báo GDP 2030", "orange"), unsafe_allow_html=True)

    scenarios = {
        "Bi quan (S1)": y_forecast[n:] * 0.90,
        "Cơ sở (S2)": y_forecast[n:],
        "Lạc quan (S3)": y_forecast[n:] * 1.08,
    }
    scenario_df = pd.DataFrame({
        "Năm": forecast_years,
        "S1 - Bi quan (Tỷ VND)": np.round(scenarios["Bi quan (S1)"], 1),
        "S2 - Cơ sở (Tỷ VND)": np.round(scenarios["Cơ sở (S2)"], 1),
        "S3 - Lạc quan (Tỷ VND)": np.round(scenarios["Lạc quan (S3)"], 1),
        "S2 GDP (Tỷ USD)": np.round(scenarios["Cơ sở (S2)"] * 1000 / 23807, 1),
    })
    st.dataframe(scenario_df.style.format({
        "S1 - Bi quan (Tỷ VND)": "{:.1f}",
        "S2 - Cơ sở (Tỷ VND)": "{:.1f}",
        "S3 - Lạc quan (Tỷ VND)": "{:.1f}",
        "S2 GDP (Tỷ USD)": "{:.1f}",
    }), width='stretch')

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    fig3, fig4 = st.columns(2)
    with fig3:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=digital, y=gdp_billion_actual,
            mode="markers+text",
            marker=dict(color=digital, colorscale="Viridis", size=12,
                       line=dict(color="white", width=1)),
            text=[str(int(y)) for y in years],
            textposition="top center", textfont={"color": "#E0E0FF", "size": 10},
            hovertemplate="Năm %{text}: KTS=%{x:.1f}%, GDP=%{y:.1f}B<extra></extra>",
        ))
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=380,
            xaxis=dict(gridcolor="#1F2A5A", title="Tỷ lệ kinh tế số (%)"),
            yaxis=dict(gridcolor="#1F2A5A", title="GDP (Tỷ USD)"),
            title=dict(text="GDP vs Kinh tế số", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_sc, width='stretch')

    with fig4:
        lp_vals = lp / lp[0] * 100
        fig_lp = go.Figure()
        fig_lp.add_trace(go.Scatter(
            x=list(years), y=list(lp_vals),
            mode="lines+markers", name="Năng suất (2020=100)",
            line=dict(color=NEON_GREEN, width=2.5),
            marker=dict(size=8),
            hovertemplate="Năm %{x}: %{y:.1f}<extra></extra>",
        ))
        fig_lp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=380,
            xaxis=dict(gridcolor="#1F2A5A", title="Năm"),
            yaxis=dict(gridcolor="#1F2A5A", title="Index (2020=100)"),
            title=dict(text="Năng suất lao động (2020=100)", x=0.5, font={"color": NEON_GREEN, "size": 14}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_lp, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    tfp_avg = np.mean(ga["TFP"]) if len(ga["TFP"]) > 0 else 0
    tfp_last = ga["TFP"][-1] if len(ga["TFP"]) > 0 else 0
    tfp_trend = "tăng" if (len(ga["TFP"]) > 1 and ga["TFP"][-1] > np.mean(ga["TFP"][:-1])) else "giảm"
    tfp_direction = 1 if (len(ga["TFP"]) > 1 and ga["TFP"][-1] > np.mean(ga["TFP"][:-1])) else -1

    contrib_L = np.mean(np.abs(ga["contrib_L"]))
    contrib_K = np.mean(np.abs(ga["contrib_K"]))
    contrib_D = np.mean(np.abs(ga["contrib_D"]))
    top_contrib_name = max([("Lao động", contrib_L), ("Vốn", contrib_K), ("Kinh tế số", contrib_D)], key=lambda x: x[1])[0]
    top_contrib_val = max(contrib_L, contrib_K, contrib_D)

    gdp_2030_vnd = y_forecast[-1]
    gdp_2030_usd = y_forecast[-1] * 1000 / 23807
    s3_gdp = y_forecast[-1] * 1.08
    s3_usd = s3_gdp * 1000 / 23807
    digital_2030 = D_all[-1]
    cagr_forecast = cagr(gdp[-1], y_forecast[-1], n_forecast)

    with st.expander("TFP của Việt Nam đang tăng hay giảm và phản ánh điều gì?"):
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0; line-height:1.8;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        TFP trung bình giai đoạn {int(years[0])}-{int(years[-1])} đạt <b style="color:#00D4FF;">{tfp_avg:.2f}%/năm</b>.
        Năm {int(years[-1])}, TFP đạt <b style="color:#00FF88;">{tfp_last:.2f}%</b>,
        cho thấy năng suất nhân tố tổng hợp có xu hướng {"tích cực" if tfp_direction > 0 else "suy giảm nhẹ"}
        trong giai đoạn gần đây. Mô hình Cobb-Douglas mở rộng với hệ số R² = {r2:.4f} cho thấy
        TFP đóng góp {"đáng kể" if abs(tfp_last) > 0.5 else "ổn định"} vào tăng trưởng GDP.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        TFP tăng phản ánh hiệu quả sản xuất được cải thiện nhờ: (1) Ứng dụng công nghệ số hóa
        vào quy trình sản xuất kinh doanh. (2) Cải tiến quản lý và tổ chức lao động. (3) Chuyển giao
        công nghệ từ FDI. Hệ số γ (kinh tế số) = {gamma:.4f} cho thấy mỗi 1% tăng trưởng kinh tế số
        tương đương với {gamma:.4f}% tăng trưởng GDP, trực tiếp thúc đẩy TFP.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        TFP dương có nghĩa tăng trưởng GDP không chỉ đến từ gia tăng vốn và lao động,
        mà còn từ <b style="color:#FFD700;">chất lượng tăng trưởng</b>. Điều này cho thấy nền kinh tế
        Việt Nam đang dần chuyển từ mô hình tăng trưởng theo chiều rộng sang chiều sâu.
        Hiệu quả sử dụng nguồn lực được cải thiện, đây là dấu hiệu tích cực cho phát triển bền vững.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Đẩy mạnh chuyển đổi số</b> — Kinh tế số là động lực trực tiếp thúc đẩy TFP.
        (2) <b style="color:#00D4FF;">Nâng cao chất lượng FDI</b> — Ưu tiên dự án có chuyển giao công nghệ cao.
        (3) <b style="color:#00D4FF;">Đầu tư giáo dục và đào tạo</b> — Nâng cao kỹ năng lao động để tăng năng suất.
        (4) <b style="color:#00D4FF;">Thúc đẩy đổi mới sáng tạo</b> — Tăng cường R&D trong doanh nghiệp để duy trì đà tăng TFP.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Yếu tố nào đóng góp lớn nhất vào tăng trưởng GDP?"):
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Phân rã tăng trưởng GDP (Growth Accounting) cho thấy: Lao động đóng góp trung bình
        <b style="color:#00FF88;">{contrib_L:.2f}%/năm</b>, Vốn đóng góp <b style="color:#BF00FF;">{contrib_K:.2f}%/năm</b>,
        Kinh tế số đóng góp <b style="color:#FFD700;">{contrib_D:.2f}%/năm</b>.
        <b style="color:#00D4FF;">{top_contrib_name}</b> là yếu tố đóng góp lớn nhất với {top_contrib_val:.2f}%/năm.
        Hệ số α = {alpha:.4f}, β = {beta:.4f}, γ = {gamma:.4f} cho thấy lao động và vốn
        là hai trụ cột chính của tăng trưởng Việt Nam, trong khi kinh tế số đang dần trở thành
        yếu tố ngày càng quan trọng.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Lao động chiếm tỷ trọng lớn nhất vì Việt Nam có lực lượng lao động dồi dào ({df['population_million'].values[-1]:.1f} triệu người),
        nguồn lao động trẻ và chi phí lao động cạnh tranh. Vốn (FDI + đầu tư nội địa) đứng thứ hai
        nhờ dòng vốn FDI mạnh vào Việt Nam trong giai đoạn {int(years[0])}-{int(years[-1])}.
        Kinh tế số có hệ số γ = {gamma:.4f} — thấp hơn lao động nhưng có xu hướng tăng,
        cho thấy chuyển đổi số đang bắt đầu phát huy tác dụng.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Mô hình tăng trưởng của Việt Nam hiện tại vẫn mang tính <b style="color:#FF6B35;">chiều rộng</b> —
        phụ thuộc chủ yếu vào gia tăng yếu tố đầu vào (lao động, vốn). Tổng α + β + γ = {alpha+beta+gamma:.4f},
        gần quy mô không đổi (returns to scale), cho thấy tăng quy mô đầu vào 1% sẽ tăng GDP khoảng
        {(alpha+beta+gamma)*100:.2f}%. Cần chú trọng nâng cao chất lượng tăng trưởng bằng cách đẩy mạnh
        kinh tế số để tăng hệ số γ.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Chiến lược lao động</b> — Nâng cao chất lượng đào tạo, chuyển lao động từ
        nông nghiệp sang công nghiệp và dịch vụ. (2) <b style="color:#00D4FF;">Thu hút vốn chất lượng cao</b> —
        FDI cần đi vào ngành có giá trị gia tăng cao. (3) <b style="color:#00D4FF;">Ưu tiên kinh tế số</b> —
        Với γ = {gamma:.4f}, tăng 1% kinh tế số tương đương tăng {gamma:.4f}% GDP.
        Mỗi 1% tăng trưởng kinh tế số mang lại hiệu quả tăng trưởng cao nhất trên mỗi đơn vị đầu tư.
        (4) <b style="color:#00D4FF;">Cân bằng chiến lược</b> — Duy trì đầu vào lao động và vốn, đồng thời đẩy mạnh kinh tế số.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Mục tiêu kinh tế số và GDP 2030 có khả thi không?"):
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Theo dự báo cơ sở (S2), GDP năm 2030 đạt <b style="color:#00D4FF;">{gdp_2030_vnd:.0f} Tỷ VND</b>
        (<b style="color:#00FF88;">{gdp_2030_usd:.1f} Tỷ USD</b>), CAGR 2025-2030 = {cagr_forecast:.2f}%/năm.
        Kịch bản lạc quan (S3) cho GDP 2030 = {s3_usd:.1f} Tỷ USD.
        Kinh tế số dự báo đạt <b style="color:#FFD700;">{digital_2030:.0f}% GDP</b> năm 2030
        (với tốc độ tăng trưởng ~5%/năm như giai đoạn {int(years[0])}-{int(years[-1])}).
        Mô hình Cobb-Douglas với R² = {r2:.4f} cho thấy dự báo có độ tin cậy cao.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Tốc độ tăng trưởng kinh tế số Việt Nam giai đoạn {int(years[0])}-{int(years[-1])} đạt bình quân
        ~5%/năm. Mô hình dự báo giả định mức tăng 5%/năm cho giai đoạn 2026-2030.
        Với hệ số γ = {gamma:.4f}, mỗi 1% tăng kinh tế số trực tiếp thúc đẩy GDP tăng {gamma:.4f}%.
        GDP tăng trưởng {cagr_forecast:.2f}%/năm trong dự báo phản ánh xu hướng lịch sử
        và kỳ vọng về tiếp tục cải thiện năng suất lao động thông qua số hóa.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nếu đạt được mục tiêu kinh tế số {digital_2030:.0f}% GDP, Việt Nam sẽ thuộc nhóm
        <b style="color:#00FF88;">dẫn đầu ASEAN</b> về mức độ số hóa. GDP {gdp_2030_usd:.1f} Tỷ USD
        đưa thu nhập bình quân đầu người lên khoảng {gdp_2030_usd*1000/103/(df['population_million'].values[-1]):.0f} USD,
        tiến gần hơn tới mục tiêu thoát khỏi mức thu nhập trung bình thấp.
        Tuy nhiên, kịch bản lạc quan {s3_usd:.1f} Tỷ USD đòi hỏi đầu tư mạnh vào chuyển đổi số.
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Đầu tư hạ tầng số</b> — Phủ sóng 5G, trung tâm dữ liệu, nền tảng chính phủ số.
        (2) <b style="color:#00D4FF;">Hỗ trợ doanh nghiệp SME chuyển đổi số</b> — Ngân sách ưu đãi, đào tạo kỹ năng số.
        (3) <b style="color:#00D4FF;">Xây dựng chính phủ số</b> — Dịch vụ công trực tuyến, dữ liệu mở.
        (4) <b style="color:#00D4FF;">Phát triển nguồn nhân lực số</b> — Đào tạo 1 triệu lập trình viên và kỹ sư AI.
        (5) <b style="color:#00D4FF;">Hoàn thiện khung pháp lý</b> — An ninh mạng, bảo vệ dữ liệu, thương mại điện tử.
        </div>
        """, unsafe_allow_html=True)
