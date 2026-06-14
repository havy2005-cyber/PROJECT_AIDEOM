import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_sectors
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.charts import bar_chart, pie_chart

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Tối ưu phân bổ ngân sách chuyển đổi số cho 10 ngành kinh tế Việt Nam.",
        "Tối đa hóa tác động đầu tư dựa trên GDP, xuất khẩu, chỉ số số hóa và AI Readiness.",
        "So sánh phân bổ tối ưu với phân bổ đều để đánh giá hiệu quả.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)

    df = load_sectors()
    if df is None:
        st.warning("Không tìm thấy dữ liệu vietnam_sectors_2024.csv.")
        return

    data_desc = [
        "10 ngành kinh tế: Nông nghiệp, Chế biến, Xây dựng, Khai khoáng, Bán lẻ, Tài chính, Logistics, CNTT, Giáo dục, Y tế.",
        "Nhu cầu đầu tư: Proxy qua FDI thu hút, quy mô GDP ngành.",
        "Hiệu quả đầu tư: GDP, xuất khẩu, chỉ số số hóa, AI Readiness.",
    ]
    for d in data_desc:
        st.markdown(f"- {d}")

    col_name = "sector_name_en"
    col_gdp = "gdp_share_2024_pct"
    col_export = "export_billion_USD"
    col_digital = "digital_index_0_100"
    col_ai = "ai_readiness_0_100"
    col_spillover = "spillover_coef_0_1"
    col_fdi = "fdi_attraction_billion_USD"

    sectors = df[col_name].tolist()
    gdp_shares = df[col_gdp].values.astype(float)
    exports = df[col_export].values.astype(float)
    digital_idx = df[col_digital].values.astype(float)
    ai_idx = df[col_ai].values.astype(float)
    spillover = df[col_spillover].values.astype(float)
    fdi = df[col_fdi].values.astype(float)
    n_sectors = len(sectors)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình
    st.markdown(section_header("Cấu hình mô hình", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(3)
    total_budget = cfg_cols[0].number_input("Tổng ngân sách (Tỷ USD)", value=10.0, min_value=1.0, max_value=100.0, step=0.5)
    min_pct = cfg_cols[1].slider("Tỷ lệ tối thiểu/ngành (%)", 1, 20, 5) / 100.0
    max_pct = cfg_cols[2].slider("Tỷ lệ tối đa/ngành (%)", 10, 50, 25) / 100.0

    use_weighted = st.checkbox("Sử dụng trọng số (GDP + Xuất khẩu + Số hóa + AI)", value=True)
    use_spillover = st.checkbox("Tính hệ số lan tỏa (spillover)", value=True)

    if use_weighted:
        w_gdp, w_exp, w_dig, w_ai = 0.3, 0.25, 0.25, 0.2
        if HAS_PULP:
            impact_scores = (
                w_gdp * (gdp_shares / gdp_shares.max()) +
                w_exp * (exports / (exports.max() + 1e-6)) +
                w_dig * (digital_idx / 100.0) +
                w_ai * (ai_idx / 100.0)
            )
            if use_spillover:
                impact_scores = impact_scores * (1 + spillover)
            impact_scores = impact_scores / impact_scores.sum()
        else:
            impact_scores = np.ones(n_sectors) / n_sectors
    else:
        impact_scores = np.ones(n_sectors) / n_sectors

    # ---- 2.4.1 Hàm mục tiêu ----
    st.markdown('<div class="section-header-purple">2.4.1 Hàm mục tiêu</div>', unsafe_allow_html=True)

    render_html(glass_card(f"""
    <div class="body-text" style="line-height:2;">
        <b style="color:#00D4FF;">Mô hình LP (PuLP):</b><br>
        Tối đa hóa <b>Z = Σ wᵢ × xᵢ</b><br>
        Trong đó xᵢ = tỷ lệ ngân sách ngành i<br><br>
        <b>Ràng buộc:</b><br>
        • Σ xᵢ = 1 (tổng ngân sách = 100%)<br>
        • {min_pct*100:.0f}% ≤ xᵢ ≤ {max_pct*100:.0f}% với mọi i<br><br>
        <b>Trọng số tác động (wᵢ):</b><br>
        wᵢ = {w_gdp:.0%} × (GDPᵢ/GDP_max) + {w_exp:.0%} × (XKᵢ/XK_max) + {w_dig:.0%} × (SHᵢ/100) + {w_ai:.0%} × (AIᵢ/100)
        {f'<br>+ Hiệu ứng lan tỏa spillover × {spillover.mean():.2f}' if use_spillover else ''}
    </div>
    """))

    # ---- 2.4.2 Phân bổ tối ưu ----
    st.markdown(section_header("2.4.2 Phân bổ tối ưu", "purple"), unsafe_allow_html=True)

    if HAS_PULP:
        model = pulp.LpProblem("Phan_bo_Ngan_sach_So", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", lowBound=min_pct, upBound=max_pct) for i in range(n_sectors)}
        model += pulp.lpSum([impact_scores[i] * x[i] for i in range(n_sectors)]), "Hieu_qua_toi_da"
        model += pulp.lpSum([x[i] for i in range(n_sectors)]) == 1.0, "Tong_ngan_sach"
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[model.status]
        allocations = np.array([pulp.value(x[i]) for i in range(n_sectors)]) * 100
        budget_values = allocations / 100 * total_budget

        st.success(f"Trạng thái tối ưu: **{status}** | Objective = {pulp.value(model.objective):.4f}")

        top_sector = sectors[np.argmax(allocations)]
        bot_sector = sectors[np.argmin(allocations)]

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            render_html(kpi_card(f"{np.sum(allocations):.1f}%", "Tổng phân bổ", "Tổng % được phân bổ", NEON_BLUE))
        with kpi_cols[1]:
            render_html(kpi_card(f"{np.max(allocations):.1f}%", "Ngành nhiều nhất", top_sector[:20], NEON_GREEN))
        with kpi_cols[2]:
            render_html(kpi_card(f"{np.min(allocations):.1f}%", "Ngành ít nhất", bot_sector[:20], NEON_ORANGE))
        with kpi_cols[3]:
            render_html(kpi_card(f"{total_budget:.1f}B", "Ngân sách", "Tổng ngân sách", NEON_PURPLE))

        # Bảng phân bổ
        alloc_df = pd.DataFrame({
            "Ngành": sectors,
            "Trọng số tác động": np.round(impact_scores * 100, 2),
            "Phân bổ (%)": np.round(allocations, 2),
            "Ngân sách (Tỷ USD)": np.round(budget_values, 2),
            "Xuất khẩu (B USD)": exports,
            "AI Readiness": ai_idx,
        }).sort_values("Phân bổ (%)", ascending=False)
        st.dataframe(alloc_df.style.format({
            "Trọng số tác động": "{:.2f}",
            "Phân bổ (%)": "{:.2f}",
            "Ngân sách (Tỷ USD)": "{:.2f}",
            "Xuất khẩu (B USD)": "{:.1f}",
            "AI Readiness": "{:.0f}",
        }), width='stretch')

        # ---- 2.4.3 So sánh trước/sau ----
        st.markdown(section_header("2.4.3 So sánh trước và sau tối ưu", "green"), unsafe_allow_html=True)

        equal_alloc = np.ones(n_sectors) / n_sectors * 100
        comp_df = pd.DataFrame({
            "Ngành": sectors,
            "Phân bổ đều (%)": np.round(equal_alloc, 1),
            "Phân bổ tối ưu (%)": np.round(allocations, 1),
            "Chênh lệch (%)": np.round(allocations - equal_alloc, 1),
        })
        st.dataframe(comp_df.style.format({
            "Phân bổ đều (%)": "{:.1f}",
            "Phân bổ tối ưu (%)": "{:.1f}",
            "Chênh lệch (%)": "{:+.1f}",
        }), width='stretch')

    else:
        st.error("PuLP chưa được cài đặt. Chạy: pip install pulp")

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    if HAS_PULP:
        chart_cols = st.columns(2)
        with chart_cols[0]:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[s[:20] for s in sectors],
                y=list(allocations) if HAS_PULP else list(np.ones(n_sectors) / n_sectors * 100),
                marker_color=NEON_PURPLE, opacity=0.85,
                hovertemplate="Ngành: %{x}<br>Phân bổ: %{y:.2f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=400,
                xaxis=dict(gridcolor="#1F2A5A", tickangle=25, title="Ngành"),
                yaxis=dict(gridcolor="#1F2A5A", title="Phân bổ (%)"),
                title=dict(text="Phân bổ ngân sách tối ưu (%)", x=0.5, font={"color": NEON_PURPLE, "size": 14}),
                hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
            )
            st.plotly_chart(fig_bar, width='stretch')
        with chart_cols[1]:
            fig_pie = go.Figure(data=[go.Pie(
                labels=sectors,
                values=list(budget_values),
                marker_colors=[NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, "#FFD700", "#FF4B8C", "#00BFFF", "#8B00FF", "#FF00AA", "#00FFAA"],
                textinfo="label+percent", textposition="outside",
                hole=0.45,
                hovertemplate="%{label}: %{percent}<extra></extra>",
            )])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E0E0FF"}, height=400,
                title=dict(text="Ngân sách theo ngành (Tỷ USD)", x=0.5, font={"color": NEON_PURPLE, "size": 14}),
                legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            )
            st.plotly_chart(fig_pie, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    if HAS_PULP:
        top_idx = np.argmax(allocations)
        bot_idx = np.argmin(allocations)
        top_sector = sectors[top_idx]
        bot_sector = sectors[bot_idx]
        top_alloc = allocations[top_idx]
        bot_alloc = allocations[bot_idx]
        top_budget = budget_values[top_idx]
        bot_budget = budget_values[bot_idx]
        eff_gain = np.sum(np.abs(allocations - equal_alloc)) / 2
        top_digital = digital_idx[top_idx]
        top_ai = ai_idx[top_idx]
        top_export = exports[top_idx]
        bot_digital = digital_idx[bot_idx]
        bot_ai = ai_idx[bot_idx]

        with st.expander("Ngành nào nhận vốn lớn nhất và tại sao?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Mô hình LP tối ưu phân bổ <b style="color:#00D4FF;">{total_budget:.1f} Tỷ USD</b>
            cho 10 ngành kinh tế. <b style="color:#00FF88;">{top_sector}</b>
            dẫn đầu với tỷ lệ <b style="color:#FFD700;">{top_alloc:.2f}%</b> ngân sách
            (tương đương <b style="color:#00D4FF;">{top_budget:.2f} Tỷ USD</b>).
            Ngành có điểm số tác động cao nhất nhờ kết hợp tối ưu giữa GDP, xuất khẩu,
            số hóa và AI Readiness.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            {top_sector} nhận vốn lớn nhất vì có lợi thế tổng hợp: (1) <b style="color:#00D4FF;">Xuất khẩu {top_export:.1f}B USD</b>
            — tạo nguồn ngoại tệ và thu hút FDI. (2) <b style="color:#00FF88;">Chỉ số số hóa {top_digital:.0f}/100</b>
            — đảm bảo hiệu quả sử dụng vốn số. (3) <b style="color:#BF00FF;">AI Readiness {top_ai:.0f}/100</b>
            — sẵn sàng ứng dụng công nghệ. (4) GDP share lớn — là đầu tàu kinh tế.
            Trọng số (30% GDP + 25% XK + 25% Số hóa + 20% AI) ưu tiên ngành có
            sự kết hợp tối ưu giữa quy mô và năng lực công nghệ.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Việc phân bổ {top_budget:.2f} Tỷ USD cho {top_sector} có ý nghĩa chiến lược:
            ngành này đóng vai trò <b style="color:#FFD700;">đầu tàu kinh tế</b>, tạo hiệu ứng lan tỏa
            (spillover) sang các ngành khác thông qua chuỗi cung ứng và dịch vụ hỗ trợ.
            Đầu tư vào ngành có AI Readiness cao ({top_ai:.0f}/100) đảm bảo vốn được sử dụng
            hiệu quả, thúc đẩy tăng năng suất lao động và nâng cao năng lực cạnh tranh quốc tế.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Ưu tiên ngân sách cho ngành dẫn đầu</b> — Tập trung nguồn lực vào ngành có
            hiệu quả lan tỏa cao và AI Readiness tốt. (2) <b style="color:#00D4FF;">Thúc đẩy chuyển đổi số ngành</b> —
            Nâng cao chỉ số số hóa cho ngành có tiềm năng nhưng chưa được đầu tư đúng mức.
            (3) <b style="color:#00D4FF;">Kết nối chuỗi giá trị</b> — Khuyến khích ngành dẫn đầu hỗ trợ ngành
            thượng nguồn và hạ nguồn. (4) <b style="color:#00D4FF;">Xây dựng quỹ hỗ trợ SME</b> —
            Giúp doanh nghiệp nhỏ trong ngành tiếp cận vốn và công nghệ số.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Phân bổ tối ưu hiệu quả hơn phân bổ đều bao nhiêu?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Phân bổ tối ưu LP mang lại mức hiệu quả tác động cao hơn đáng kể so với phân bổ đều.
            Chênh lệch tổng cộng giữa hai phương pháp đạt <b style="color:#FFD700;">{eff_gain:.1f}%</b>
            điểm hiệu suất. Ngành dẫn đầu ({top_sector}) được phân bổ cao hơn mức đều
            {top_alloc - equal_alloc[top_idx]:.1f}%, trong khi ngành yếu nhất ({bot_sector})
            được giảm {equal_alloc[bot_idx] - bot_alloc:.1f}% so với mức đều.
            Khoảng cách GDP giữa ngành dẫn đầu và ngành cuối trong phân bổ tối ưu
            phản ánh chiến lược phân bổ theo năng lực cạnh tranh.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Phân bổ đều (mỗi ngành 10%) bỏ qua sự khác biệt về hiệu quả đầu tư giữa các ngành.
            LP tối ưu hóa hàm mục tiêu dựa trên trọng số tác động — ưu tiên ngành có:
            GDP lớn (đầu tàu), xuất khẩu cao (tạo ngoại tệ), AI Readiness cao (dùng vốn hiệu quả),
            và số hóa tốt (chi phí chuyển đổi thấp). Hiệu quả lan tỏa (spillover) được tính thêm
            {spillover.mean()*100:.1f}% nếu bật, khuyến khích đầu tư vào ngành có tác động liên ngành.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Với tổng ngân sách {total_budget:.1f} Tỷ USD, phân bổ tối ưu đảm bảo mỗi đồng đầu tư
            mang lại <b style="color:#00FF88;">tác động GDP cao nhất</b>. Ngành có spillover cao nhân vốn hiệu quả,
            tạo ra lợi ích vượt trội so với đầu tư đều. Mô hình cũng đảm bảo mỗi ngành được
            tối thiểu {min_pct*100:.0f}% và tối đa {max_pct*100:.0f}% ngân sách — tránh cực đoan
            và đảm bảo tính khả thi chính trị của phân bổ.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Áp dụng phân bổ có trọng số</b> — Bộ KH&ĐT nên dùng mô hình LP
            để xây dựng kế hoạch phân bổ ngân sách hàng năm. (2) <b style="color:#00D4FF;">Thiết lập ngưỡng tối thiểu</b> —
            Đảm bảo mỗi ngành strategic được tối thiểu 5% ngân sách. (3) <b style="color:#00D4FF;">Cơ chế đánh giá hiệu quả</b> —
            Đo lường KPI thực tế sau đầu tư để điều chỉnh phân bổ năm sau.
            (4) <b style="color:#00D4FF;">Ưu tiên ngành có spillover cao</b> — Vì lợi ích lan tỏa tạo hiệu ứng
            nhân vốn cho toàn nền kinh tế.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Ngành nào cần ưu tiên nâng cấp và đầu tư thế nào?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            <b style="color:#FF6B35;">{bot_sector}</b> nhận tỷ lệ ngân sách thấp nhất ({bot_alloc:.2f}%)
            do có chỉ số số hóa ({bot_digital:.0f}/100) và AI Readiness ({bot_ai:.0f}/100) thấp.
            Ngành này có tiềm năng phát triển nhưng chưa sẵn sàng hấp thụ vốn hiệu quả.
            Cần chiến lược nâng cấp kép: vừa cải thiện hạ tầng số, vừa đào tạo nhân lực.
            Tuy nhiên, nếu {bot_sector} là ngành thiết yếu (nông nghiệp, y tế, giáo dục),
            vẫn cần đảm bảo đầu tư tối thiểu để duy trì an sinh xã hội.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            {bot_sector} bị giảm phân bổ vì: (1) <b style="color:#FF6B35;">Chỉ số số hóa thấp ({bot_digital:.0f}/100)</b>
            — đầu tư vào ngành chưa số hóa sẽ kém hiệu quả. (2) <b style="color:#FF6B35;">AI Readiness thấp ({bot_ai:.0f}/100)</b>
            — nhân lực chưa đủ kỹ năng để vận hành công nghệ mới. (3) <b style="color:#FF6B35;">Xuất khẩu thấp</b> —
            không tạo nguồn ngoại tệ để bù đắp chi phí đầu tư. (4) <b style="color:#FF6B35;">Hiệu ứng lan tỏa yếu</b> —
            đầu tư vào ngành này không tạo nhiều lợi ích chuyển tiếp sang ngành khác.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Việc giảm phân bổ cho ngành yếu không đồng nghĩa với bỏ rơi. Nếu đây là ngành nông nghiệp
            hoặc dịch vụ thiết yếu, cần có <b style="color:#FFD700;">chính sách bảo hộ đặc biệt</b> để đảm bảo
            an ninh lương thực và an sinh xã hội. Chiến lược đúng đắn là: đầu tư ngân sách tập trung
            vào ngành có AI Readiness cao để tối đa hóa hiệu quả, đồng thời dành ngân sách phát triển
            (ODA, vốn mền) cho ngành yếu nâng cấp hạ tầng.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Đầu tư hạ tầng số hóa</b> — Nâng cấp chỉ số số hóa từ {bot_digital:.0f}
            lên ít nhất 50/100 để tăng hiệu quả đầu tư. (2) <b style="color:#00D4FF;">Chương trình đào tạo kỹ năng số</b> —
            Nâng cao AI Readiness từ {bot_ai:.0f} lên 40+ cho ngành {bot_sector}. (3) <b style="color:#00D4FF;">Cơ chế hỗ trợ đặc thù</b> —
            Chính sách thuế, tín dụng ưu đãi cho doanh nghiệp trong ngành chuyển đổi số.
            (4) <b style="color:#00D4FF;">Giám sát và điều chỉnh</b> — Theo dõi KPI ngành hàng quý,
            tái cân bằng phân bổ nếu ngành đạt ngưỡng số hóa tối thiểu.
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander("Mô hình LP chưa sẵn sàng"):
            st.markdown("""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            PuLP chưa được cài đặt. Vui lòng chạy lệnh: <code>pip install pulp</code>
            để kích hoạt mô hình tối ưu phân bổ ngân sách số.
            Sau khi cài đặt, hệ thống sẽ tự động hiển thị phân bổ tối ưu cho 10 ngành kinh tế
            dựa trên GDP, xuất khẩu, số hóa và AI Readiness.
            </div>
            """, unsafe_allow_html=True)
