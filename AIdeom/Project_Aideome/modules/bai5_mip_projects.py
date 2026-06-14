import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_sectors
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.charts import bar_chart, pie_chart

try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False


def run():
    df = load_sectors()
    if df is None:
        st.warning("Không tìm thấy dữ liệu vietnam_sectors_2024.csv.")
        return

    col_name = "sector_name_en"
    col_gdp = "gdp_share_2024_pct"
    col_growth = "growth_rate_2024_pct"
    col_export = "export_billion_USD"
    col_digital = "digital_index_0_100"
    col_ai = "ai_readiness_0_100"
    col_labor = "labor_million"
    col_spillover = "spillover_coef_0_1"
    col_fdi = "fdi_attraction_billion_USD"
    col_rd = "rd_intensity_pct"
    col_auto = "automation_risk_pct"

    sectors = df[col_name].tolist()
    gdp_shares = df[col_gdp].values.astype(float)
    growth = df[col_growth].values.astype(float)
    exports = df[col_export].values.astype(float)

    
    digital = df[col_digital].values.astype(float)
    ai = df[col_ai].values.astype(float)
    labor = df[col_labor].values.astype(float)
    spillover = df[col_spillover].values.astype(float)
    fdi = df[col_fdi].values.astype(float)
    rd = df[col_rd].values.astype(float)
    auto_risk = df[col_auto].values.astype(float)
    n = len(sectors)

    # Tạo 22 dự án
    projects = []
    pid = 0
    for i, sec in enumerate(sectors):
        for j in range(2):
            np.random.seed(42 + pid)
            benefit = (
                0.25 * gdp_shares[i] +
                0.25 * growth[i] +
                0.15 * exports[i] / 10.0 +
                0.15 * digital[i] / 100.0 +
                0.1 * ai[i] / 100.0 +
                0.1 * spillover[i]
            ) * (0.8 + 0.4 * np.random.rand())
            risk = (
                0.3 * auto_risk[i] / 100.0 +
                0.3 * (1 - ai[i] / 100.0) +
                0.2 * (1 - rd[i]) +
                0.2 * (1 - digital[i] / 100.0)
            ) * (0.7 + 0.6 * np.random.rand())
            cost = (
                0.3 * fdi[i] +
                0.3 * gdp_shares[i] * 5.0 +
                0.2 * labor[i] * 2.0 +
                0.2 * rd[i] * 50.0
            ) * (0.6 + 0.8 * np.random.rand())

            projects.append({
                "id": pid,
                "name": f"DA-{pid+1:02d}: {sec[:30]}",
                "sector": sec,
                "benefit": benefit,
                "risk": risk,
                "cost": cost,
                "net_benefit": benefit - risk,
            })
            pid += 1

    projects_df = pd.DataFrame(projects)
    n_total = len(projects)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình
    st.markdown(section_header("Cấu hình mô hình MIP", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(3)
    max_projects = cfg_cols[0].slider("Số dự án tối đa", 5, 20, 15)
    total_budget = cfg_cols[1].number_input("Tổng ngân sách (Tỷ USD)", value=50.0, min_value=1.0, max_value=500.0, step=1.0)
    min_per_sector = cfg_cols[2].slider("Dự án tối thiểu/ngành", 0, 3, 1)

    show_projects = st.checkbox("Hiển thị danh sách dự án", value=True)

    # ---- 5.4.1 Hàm mục tiêu ----
    st.markdown('<div class="section-header-purple">5.4.1 Hàm mục tiêu</div>', unsafe_allow_html=True)

    render_html(glass_card(f"""
    <div class="body-text" style="line-height:2;">
        <b style="color:#00D4FF;">Mô hình MIP (PuLP):</b><br>
        Tối đa hóa <b>Z = Σ (Lợi nhuận ròngᵢ × xᵢ)</b><br>
        Trong đó xᵢ ∈ {{0, 1}} là biến nhị phân chọn dự án i<br><br>
        <b>Ràng buộc:</b><br>
        • Σ xᵢ ≤ {max_projects} (số dự án tối đa)<br>
        • Σ (Chi phíᵢ × xᵢ) ≤ {total_budget:.1f}B USD (ngân sách)<br>
        • Σ xᵢ (ngành j) ≥ {min_per_sector} với mỗi ngành<br><br>
        <b>Định nghĩa:</b><br>
        Lợi nhuận ròng = Lợi ích - Rủi ro
    </div>
    """))

    if show_projects:
        st.markdown(section_header("Danh sách 22 dự án", "purple"), unsafe_allow_html=True)
        disp = projects_df[["id", "name", "sector", "benefit", "risk", "cost", "net_benefit"]].copy()
        disp.columns = ["ID", "Tên dự án", "Ngành", "Lợi ích", "Rủi ro", "Chi phí (B USD)", "Lợi nhuận ròng"]
        st.dataframe(disp.style.format({
            "Lợi ích": "{:.3f}",
            "Rủi ro": "{:.3f}",
            "Chi phí (B USD)": "{:.2f}",
            "Lợi nhuận ròng": "{:.3f}",
        }), use_container_width=True)

    # ---- 5.4.2 Danh mục dự án ----
    st.markdown(section_header("5.4.2 Danh mục dự án được chọn", "purple"), unsafe_allow_html=True)

    if HAS_PULP:
        model = pulp.LpProblem("Chon_Du_An", pulp.LpMaximize)
        x = {p["id"]: pulp.LpVariable(f"x_{p['id']}", cat="Binary") for _, p in projects_df.iterrows()}
        model += pulp.lpSum([p["net_benefit"] * x[p["id"]] for _, p in projects_df.iterrows()]), "Tong_loi_nhuan"
        model += pulp.lpSum([x[p["id"]] for _, p in projects_df.iterrows()]) <= max_projects, "So_du_an_toi_da"
        model += pulp.lpSum([p["cost"] * x[p["id"]] for _, p in projects_df.iterrows()]) <= total_budget, "Tong_chi_phi"

        for sec in sectors:
            sec_projects = [p["id"] for _, p in projects_df.iterrows() if p["sector"] == sec]
            if len(sec_projects) > 0:
                model += pulp.lpSum([x[pid] for pid in sec_projects]) >= min_per_sector, f"Min_{sec[:10]}"

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[model.status]

        selected = [p["id"] for _, p in projects_df.iterrows() if pulp.value(x[p["id"]]) > 0.5]
        total_ben = sum(projects_df[projects_df["id"] == pid]["benefit"].values[0] for pid in selected)
        total_rsk = sum(projects_df[projects_df["id"] == pid]["risk"].values[0] for pid in selected)
        total_cst = sum(projects_df[projects_df["id"] == pid]["cost"].values[0] for pid in selected)
        total_net = total_ben - total_rsk

        st.success(f"Trạng thái: **{status}** | Chọn: {len(selected)}/{n_total} | Objective: {pulp.value(model.objective):.3f}")

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            render_html(kpi_card(f"{len(selected)}", "Dự án được chọn", f"{len(selected)/n_total*100:.0f}% tổng số", NEON_BLUE))
        with kpi_cols[1]:
            render_html(kpi_card(f"{total_ben:.2f}", "Tổng lợi ích", "Đơn vị hiệu suất", NEON_GREEN))
        with kpi_cols[2]:
            render_html(kpi_card(f"{total_rsk:.2f}", "Tổng rủi ro", "Đơn vị rủi ro", NEON_ORANGE))
        with kpi_cols[3]:
            render_html(kpi_card(f"{total_net:.2f}", "Lợi nhuận ròng", "Hiệu suất chốt", NEON_PURPLE))

        sel_df = projects_df[projects_df["id"].isin(selected)][["id", "name", "sector", "benefit", "risk", "cost", "net_benefit"]].copy()
        sel_df.columns = ["ID", "Tên dự án", "Ngành", "Lợi ích", "Rủi ro", "Chi phí (B USD)", "Lợi nhuận ròng"]
        sel_df = sel_df.sort_values("Lợi nhuận ròng", ascending=False)
        st.dataframe(sel_df.style.format({
            "Lợi ích": "{:.3f}",
            "Rủi ro": "{:.3f}",
            "Chi phí (B USD)": "{:.2f}",
            "Lợi nhuận ròng": "{:.3f}",
        }), use_container_width=True)

        # ---- 5.4.3 Hiệu quả đầu tư ----
        st.markdown(section_header("5.4.3 Hiệu quả đầu tư", "green"), unsafe_allow_html=True)

        eff_df = pd.DataFrame({
            "Dự án": [f"DA-{pid+1:02d}" for pid in selected],
            "Chi phí (B USD)": [projects_df[projects_df["id"]==pid]["cost"].values[0] for pid in selected],
            "Lợi nhuận ròng": [projects_df[projects_df["id"]==pid]["net_benefit"].values[0] for pid in selected],
            "Hiệu suất (Lợi/Chi)": [projects_df[projects_df["id"]==pid]["net_benefit"].values[0] / max(projects_df[projects_df["id"]==pid]["cost"].values[0], 0.01) for pid in selected],
        }).sort_values("Hiệu suất (Lợi/Chi)", ascending=False)
        st.dataframe(eff_df.style.format({
            "Chi phí (B USD)": "{:.2f}",
            "Lợi nhuận ròng": "{:.3f}",
            "Hiệu suất (Lợi/Chi)": "{:.4f}",
        }), width='stretch')

    else:
        st.error("PuLP chưa được cài đặt. Chạy: pip install pulp")

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    import plotly.graph_objects as go

    if HAS_PULP:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[f"DA-{pid+1:02d}" for pid in selected],
                y=[projects_df[projects_df["id"]==pid]["net_benefit"].values[0] for pid in selected],
                marker_color=NEON_BLUE, opacity=0.85,
                hovertemplate="DA-%{x}: Lợi NH= %{y:.3f}<extra></extra>",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=380,
                xaxis=dict(gridcolor="#1F2A5A", tickangle=30, title="Dự án"),
                yaxis=dict(gridcolor="#1F2A5A", title="Lợi nhuận ròng"),
                title=dict(text="Lợi nhuận ròng dự án được chọn", x=0.5, font={"color": NEON_BLUE, "size": 14}),
                hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
            )
            st.plotly_chart(fig_bar, width='stretch')

        # Pie chart in second column
        with chart_cols[1]:
            # breakdown of selected projects by sector
            sel_sectors = sel_df["Ngành"].value_counts() if 'sel_df' in locals() else projects_df[projects_df["id"].isin(selected)]["sector"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=[s[:20] for s in sel_sectors.index],
                values=list(sel_sectors.values),
                marker_colors=[NEON_BLUE, NEON_GREEN, NEON_PURPLE, NEON_ORANGE, "#FFD700", "#FF4B8C", "#00BFFF", "#8B00FF"],
                textinfo="label+percent", textposition="outside",
                hole=0.45,
                hovertemplate="%{label}: %{percent}<extra></extra>",
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=380,
                title=dict(text="Phân bổ dự án theo ngành", x=0.5, font={"color": NEON_PURPLE, "size": 14}),
                legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            )
            st.plotly_chart(fig_pie, width='stretch')

        # Scatter: risk vs net_benefit (below columns)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=list(projects_df["risk"]),
            y=list(projects_df["net_benefit"]),
            mode="markers+text",
            marker=dict(
                color=[NEON_BLUE if pid in selected else "#2A3A5A" for pid in projects_df["id"]],
                size=[14 if pid in selected else 8 for pid in projects_df["id"]],
                symbol=["star" if pid in selected else "circle" for pid in projects_df["id"]],
                line=dict(color="white", width=1),
            ),
            text=[f"DA-{p['id']+1:02d}" for _, p in projects_df.iterrows()],
            textposition="top center", textfont={"color": "#E0E0FF", "size": 9},
            hovertemplate="DA-%{text}: Rủi ro=%{x:.3f}, Lợi=%{y:.3f}<extra></extra>",
        ))
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=380,
            xaxis=dict(gridcolor="#1F2A5A", title="Rủi ro"),
            yaxis=dict(gridcolor="#1F2A5A", title="Lợi nhuận ròng"),
            shapes=[dict(type="line", x0=0, x1=1, y0=0, y1=0, xref="paper", yref="y", line=dict(color="white", width=1, dash="dash"))],
            title=dict(text="Lợi nhuận vs Rủi ro (★ = được chọn)", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_sc, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    if HAS_PULP:
        top_sel = projects_df[projects_df["id"].isin(selected)].nlargest(3, "net_benefit")
        bot_not_sel = projects_df[~projects_df["id"].isin(selected)].nlargest(3, "net_benefit")
        top_names = ", ".join([f"DA-{r['id']+1:02d}" for _, r in top_sel.iterrows()])
        top_sectors = ", ".join([r['sector'][:20] for _, r in top_sel.iterrows()])
        bot_names = ", ".join([f"DA-{r['id']+1:02d}" for _, r in bot_not_sel.iterrows()])
        eff_ratio = total_net / max(total_cst, 0.01)
        avg_selected_cost = total_cst / len(selected)
        sel_per_sector = projects_df[projects_df["id"].isin(selected)]["sector"].value_counts()
        top_sel_sector = sel_per_sector.index[0] if len(sel_per_sector) > 0 else "N/A"

        with st.expander("Những dự án nào được chọn và tiêu chí ra sao?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            MIP chọn <b style="color:#00FF88;">{len(selected)}/{n_total}</b> dự án từ danh mục 22 dự án tiềm năng.
            Top 3 dự án được chọn: <b style="color:#FFD700;">{top_names}</b> — thuộc các ngành
            {top_sectors}. Các dự án này có <b style="color:#00D4FF;">lợi nhuận ròng cao nhất</b>
            (trung bình {top_sel['net_benefit'].mean():.3f}), <b style="color:#00FF88;">chi phí vừa phải</b>
            (trung bình {top_sel['cost'].mean():.2f}B USD), và <b style="color:#BF00FF;">rủi ro thấp</b>.
            Tổng ngân sách sử dụng: <b style="color:#00D4FF;">{total_cst:.1f}B / {total_budget:.1f}B USD</b>.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Các dự án được chọn đạt tiêu chí tối ưu hóa đa yếu tố: (1) <b style="color:#00D4FF;">Lợi ích cao</b> —
            Tổng hợp GDP, tăng trưởng, xuất khẩu, số hóa, AI Readiness. (2) <b style="color:#00FF88;">Rủi ro thấp</b> —
            Có automation risk thấp, AI Readiness ngành cao, và spillover tốt. (3) <b style="color:#BF00FF;">Chi phí hợp lý</b> —
            Nằm trong ngân sách và phân bổ đều giữa các ngành (mỗi ngành tối thiểu {min_per_sector} dự án).
            (4) <b style="color:#FFD700;">Hiệu ứng lan tỏa cao</b> — Dự án tạo lợi ích chuyển tiếp cho các ngành liên quan.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Danh mục {len(selected)} dự án được chọn đảm bảo <b style="color:#00FF88;">tối đa hóa lợi nhuận ròng</b>
            ({total_net:.2f}) trong phạm vi ràng buộc ngân sách và số lượng.
            Mỗi ngành có ít nhất {min_per_sector} dự án đảm bảo <b style="color:#FFD700;">tính bao phủ</b>
            và tránh bỏ rơi ngành nào. Ngành có nhiều dự án nhất: <b style="color:#00D4FF;">{top_sel_sector}</b>,
            cho thấy đây là ngành có nhiều dự án đạt tiêu chí tối ưu nhất.
            Hiệu suất đầu tư (lợi nhuận ròng / chi phí) = <b style="color:#FFD700;">{eff_ratio:.2f}</b>.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Ưu tiên triển khai Top 3</b> — {top_names} có hiệu suất cao nhất,
            nên được triển khai ngay trong giai đoạn đầu. (2) <b style="color:#00D4FF;">Đảm bảo bao phủ ngành</b> —
            Mỗi ngành cần ít nhất 1 dự án để duy trì phát triển đồng đều. (3) <b style="color:#00D4FF;">Giám sát tiến độ</b> —
            Theo dõi KPI của từng dự án hàng quý, loại bỏ dự án không đạt mục tiêu.
            (4) <b style="color:#00D4FF;">Dự phòng ngân sách</b> — Giữ lại {total_budget - total_cst:.1f}B USD
            để mở rộng danh mục nếu cơ hội tốt xuất hiện.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Tại sao một số dự án bị loại và điều đó có hợp lý?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            <b style="color:#FF6B35;">{n_total - len(selected)}/{n_total}</b> dự án bị loại.
            Top 3 dự án gần được chọn nhưng bị loại: <b style="color:#FFD700;">{bot_names}</b>.
            Các dự án này có lợi nhuận ròng dương nhưng không đủ cao để vượt qua
            các dự án được chọn trong phạm vi ngân sách và giới hạn số lượng.
            Một số bị loại vì <b style="color:#FF6B35;">rủi ro cao</b>, một số vì <b style="color:#FF6B35;">chi phí lớn</b>,
            và một số vì <b style="color:#FF6B35;">ngành đã đạt quota</b> (đủ dự án được chọn).
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Dự án bị loại chủ yếu vì: (1) <b style="color:#FF6B35;">Rủi ro automation cao</b> —
            ngành có tỷ lệ tự động hóa cao, lao động dễ bị thay thế. (2) <b style="color:#FF6B35;">AI Readiness thấp</b> —
            ngành chưa sẵn sàng ứng dụng công nghệ mới. (3) <b style="color:#FF6B35;">Chi phí vượt ngưỡng hiệu quả</b> —
            dự án đòi hỏi đầu tư lớn nhưng lợi ích không tương xứng. (4) <b style="color:#FF6B35;">Ngành đã đạt quota</b> —
            mỗi ngành chỉ có thể có tối đa {max_projects} dự án được chọn.
            Việc loại bỏ các dự án rủi ro cao là hợp lý vì bảo toàn ngân sách.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Việc MIP loại bỏ dự án rủi ro cao là chiến lược <b style="color:#FFD700;">quản trị rủi ro</b>
            hợp lý. Tuy nhiên, một số dự án bị loại có thể có giá trị chiến lược dài hạn
            (ví dụ: dự án tại vùng khó khăn, dự án có tính lan tỏa xã hội) mà mô hình
            định lượng chưa phản ánh đầy đủ. Cần xem xét lại các dự án bị loại
            dưới góc độ <b style="color:#00FF88;">an sinh xã hội</b> và <b style="color:#00FF88;">phát triển bền vững</b>.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Xem xét dự án gần được chọn</b> — {bot_names} gần đạt ngưỡng,
            cần cân nhắc nếu có thêm ngân sách. (2) <b style="color:#00D4FF;">Danh sách dự phòng</b> —
            Lưu danh sách dự án bị loại để triển khai khi có cơ hội.
            (3) <b style="color:#00D4FF;">Bổ sung tiêu chí phi kinh tế</b> — Thêm ràng buộc về tác động
            môi trường, xã hội vào mô hình MIP. (4) <b style="color:#00D4FF;">Chính sách hỗ trợ dự án bị loại</b> —
            Cung cấp hỗ trợ khác (ODA, vốn mềm) cho dự án có giá trị xã hội cao.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Tiêu chí lựa chọn dự án có khả thi trong thực tế không?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Tiêu chí lựa chọn của MIP kết hợp <b style="color:#00FF88;">5 yếu tố định lượng</b>:
            GDP, tăng trưởng, xuất khẩu, số hóa, AI Readiness — phản ánh tốt ưu tiên
            chính sách phát triển kinh tế số. Tổng lợi ích = <b style="color:#00D4FF;">{total_ben:.2f}</b>,
            tổng rủi ro = <b style="color:#FF6B35;">{total_rsk:.2f}</b>,
            lợi nhuận ròng = <b style="color:#FFD700;">{total_net:.2f}</b>.
            Ràng buộc đảm bảo mỗi ngành có ít nhất {min_per_sector} dự án, phản ánh
            yêu cầu phát triển đồng đều giữa các ngành.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Tiêu chí hiện tại có ưu điểm: (1) <b style="color:#00FF88;">Toàn diện</b> — đánh giá đa chiều
            từ quy mô (GDP), hiệu quả (số hóa), đến năng lực (AI). (2) <b style="color:#00FF88;">Cân bằng</b> —
            ràng buộc đảm bảo mỗi ngành được đại diện. (3) <b style="color:#00FF88;">Quản trị rủi ro</b> —
            tính đến rủi ro automation và AI Readiness. Hạn chế: chưa tính đầy đủ
            <b style="color:#FF6B35;">tác động môi trường</b>, <b style="color:#FF6B35;">việc làm địa phương</b>,
            và <b style="color:#FF6B35;">tính khả thi triển khai thực tế</b>.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            MIP cung cấp căn cứ <b style="color:#FFD700;">khoa học và khách quan</b> cho việc lựa chọn dự án,
            giảm thiểu quyết định chủ quan. Danh mục {len(selected)} dự án được chọn đảm bảo
            hiệu suất tối ưu trên tổng ngân sách {total_budget:.1f}B USD.
            Hiệu suất trung bình mỗi dự án: <b style="color:#00D4FF;">{total_net/len(selected):.3f}</b>
            đơn vị lợi nhuận ròng. Tuy nhiên, cần kết hợp với đánh giá chuyên gia
            và khảo sát thực địa trước khi quyết định đầu tư.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Dùng MIP làm căn cứ ban đầu</b> — MIP cung cấp danh sách dự án
            ưu tiên, không phải quyết định cuối cùng. (2) <b style="color:#00D4FF;">Bổ sung đánh giá chuyên gia</b> —
            Hội đồng thẩm định xem xét khả năng triển khai thực tế. (3) <b style="color:#00D4FF;">Cập nhật dữ liệu định kỳ</b> —
            Đánh giá lại hiệu suất dự án và cập nhật trọng số tiêu chí hàng năm.
            (4) <b style="color:#00D4FF;">Mở rộng mô hình</b> — Bổ sung tiêu chí môi trường (E),
            xã hội (S), và quản trị (G) để phù hợp với xu hướng ESG.
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander("Mô hình MIP chưa sẵn sàng"):
            st.markdown("""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            PuLP chưa được cài đặt. Vui lòng chạy lệnh: <code>pip install pulp</code>
            để kích hoạt mô hình MIP lựa chọn dự án tối ưu.
            Sau khi cài đặt, hệ thống sẽ tự động chạy thuật toán tối ưu hóa
            để chọn danh mục dự án có lợi nhuận ròng cao nhất trong phạm vi ngân sách.
            </div>
            """, unsafe_allow_html=True)
