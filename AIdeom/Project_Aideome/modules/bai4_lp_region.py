import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_regions
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.helpers import vietnamese_region_name

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Tối ưu phân bổ nguồn lực cho 6 vùng kinh tế Việt Nam.",
        "Cân bằng giữa GRDP, dân số, FDI, số hóa và AI Readiness.",
        "Giảm thiểu chênh lệch phát triển giữa các vùng.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)

    df_r = load_regions()
    if df_r is None:
        st.warning("Không tìm thấy dữ liệu vietnam_regions_2024.csv.")
        return

    data_desc = [
        "6 vùng kinh tế: Đông Bắc, Đồng Bằng Sông Hồng, Bắc Trung Bộ, Tây Nguyên, Đông Nam Bộ, Đồng Bằng Sông Cửu Long.",
        "GRDP (Tỷ VND): Tổng sản phẩm vùng.",
        "Dân số (Triệu người): Quy mô lao động.",
        "FDI (Tỷ USD): Vốn đầu tư nước ngoài đăng ký.",
        "Chỉ số số hóa & AI Readiness (0-100).",
    ]
    for d in data_desc:
        st.markdown(f"- {d}")

    col_rname = "region_name_en"
    col_grdp = "grdp_trillion_VND"
    col_pop = "population_million"
    col_fdi = "fdi_registered_billion_USD"
    col_digital = "digital_index_0_100"
    col_ai = "ai_readiness_0_100"

    regions = df_r[col_rname].tolist()
    regions_vn = [vietnamese_region_name(r) for r in regions]
    grdp = df_r[col_grdp].values.astype(float)
    pop = df_r[col_pop].values.astype(float)
    fdi = df_r[col_fdi].values.astype(float)
    digital = df_r[col_digital].values.astype(float)
    ai = df_r[col_ai].values.astype(float)
    n_regions = len(regions)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình
    st.markdown(section_header("Cấu hình phân bổ nguồn lực", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(3)
    total_resource = cfg_cols[0].number_input("Tổng nguồn lực (Tỷ VND)", value=500.0, min_value=10.0, max_value=5000.0, step=10.0)
    min_share = cfg_cols[1].slider("Tỷ lệ tối thiểu/vùng (%)", 5, 30, 10) / 100.0
    max_share = cfg_cols[2].slider("Tỷ lệ tối đa/vùng (%)", 20, 60, 40) / 100.0

    use_impact = st.checkbox("Sử dụng chỉ số tác động (FDI + Số hóa + AI)", value=True)

    impact = (
        0.3 * (grdp / grdp.max()) +
        0.3 * (fdi / (fdi.max() + 0.1)) +
        0.2 * digital / 100.0 +
        0.2 * ai / 100.0
    )
    if use_impact:
        impact_norm = impact / impact.sum()
    else:
        impact_norm = np.ones(n_regions) / n_regions

    # ---- 4.4.1 Mô hình LP ----
    st.markdown('<div class="section-header-purple">4.4.1 Mô hình LP</div>', unsafe_allow_html=True)

    render_html(glass_card(f"""
    <div class="body-text" style="line-height:2;">
        <b style="color:#00D4FF;">Mô hình PuLP:</b><br>
        Tối đa hóa <b>Z = Σ wᵢ × xᵢ</b><br>
        Trong đó xᵢ = tỷ lệ nguồn lực vùng i<br><br>
        <b>Ràng buộc:</b><br>
        • Σ xᵢ = 1 (tổng nguồn lực = 100%)<br>
        • {min_share*100:.0f}% ≤ xᵢ ≤ {max_share*100:.0f}% với mọi i<br><br>
        <b>Trọng số tác động:</b><br>
        wᵢ = 30% × GRDP + 30% × FDI + 20% × Số hóa + 20% × AI
    </div>
    """))

    # ---- 4.4.2 Phân bổ nguồn lực ----
    st.markdown(section_header("4.4.2 Phân bổ nguồn lực tối ưu", "purple"), unsafe_allow_html=True)

    if HAS_PULP:
        model = pulp.LpProblem("Phan_bo_Nguon_luc_Vung", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", lowBound=min_share, upBound=max_share) for i in range(n_regions)}
        model += pulp.lpSum([impact_norm[i] * x[i] for i in range(n_regions)]), "Hieu_qua_toi_da"
        model += pulp.lpSum([x[i] for i in range(n_regions)]) == 1.0, "Tong_nguon_luc"
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[model.status]
        allocations = np.array([pulp.value(x[i]) for i in range(n_regions)])
        resource_values = allocations * total_resource

        st.success(f"Trạng thái: **{status}** | Objective: {pulp.value(model.objective):.4f}")

        kpi_cols = st.columns(3)
        top_idx = np.argmax(allocations)
        bot_idx = np.argmin(allocations)
        gini_alloc = 1 - np.sum(allocations) ** 2 / np.sum(allocations ** 2) / n_regions

        with kpi_cols[0]:
            render_html(kpi_card(f"{np.max(allocations)*100:.1f}%", "Vùng nhiều nhất", regions_vn[top_idx][:25], NEON_BLUE))
        with kpi_cols[1]:
            render_html(kpi_card(f"{np.min(allocations)*100:.1f}%", "Vùng ít nhất", regions_vn[bot_idx][:25], NEON_ORANGE))
        with kpi_cols[2]:
            render_html(kpi_card(f"{gini_alloc:.3f}", "Hệ số Gini", "Phân bổ nguồn lực", NEON_PURPLE))

        alloc_df = pd.DataFrame({
            "Vùng": regions_vn,
            "GRDP (Tỷ VND)": grdp,
            "Dân số (Triệu)": pop,
            "FDI (B USD)": fdi,
            "Số hóa": digital,
            "AI Readiness": ai,
            "Phân bổ (%)": np.round(allocations * 100, 2),
            "Nguồn lực (Tỷ VND)": np.round(resource_values, 1),
        }).sort_values("Phân bổ (%)", ascending=False)
        st.dataframe(alloc_df.style.format({
            "GRDP (Tỷ VND)": "{:.1f}",
            "Dân số (Triệu)": "{:.2f}",
            "FDI (B USD)": "{:.1f}",
            "Số hóa": "{:.0f}",
            "AI Readiness": "{:.0f}",
            "Phân bổ (%)": "{:.2f}",
            "Nguồn lực (Tỷ VND)": "{:.1f}",
        }), width='stretch')

        # ---- 4.4.3 Kết quả tối ưu ----
        st.markdown(section_header("4.4.3 Kết quả tối ưu", "green"), unsafe_allow_html=True)

        opt_df = pd.DataFrame({
            "Vùng": regions_vn,
            "Phân bổ đều (Tỷ VND)": np.round(np.ones(n_regions) / n_regions * total_resource, 1),
            "Phân bổ tối ưu (Tỷ VND)": np.round(resource_values, 1),
            "Chênh lệch (Tỷ VND)": np.round(resource_values - np.ones(n_regions) / n_regions * total_resource, 1),
        })
        st.dataframe(opt_df.style.format({
            "Phân bổ đều (Tỷ VND)": "{:.1f}",
            "Phân bổ tối ưu (Tỷ VND)": "{:.1f}",
            "Chênh lệch (Tỷ VND)": "{:+.1f}",
        }), width='stretch')

    else:
        st.error("PuLP chưa được cài đặt. Chạy: pip install pulp")

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    import plotly.graph_objects as go

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=regions_vn,
            y=list(allocations * 100) if HAS_PULP else list(np.ones(n_regions) / n_regions * 100),
            marker_color=NEON_BLUE, opacity=0.85,
            hovertemplate="Vùng: %{x}<br>Phân bổ: %{y:.2f}%<extra></extra>",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=400,
            xaxis=dict(gridcolor="#1F2A5A", tickangle=25, title="Vùng"),
            yaxis=dict(gridcolor="#1F2A5A", title="Phân bổ (%)"),
            title=dict(text="Tỷ lệ phân bổ nguồn lực tối ưu (%)", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_bar, width='stretch')

    with chart_cols[1]:
        fig_pie = go.Figure(data=[go.Pie(
            labels=regions_vn,
            values=list(resource_values) if HAS_PULP else list(np.ones(n_regions) / n_regions * total_resource),
            marker_colors=[NEON_BLUE, NEON_GREEN, NEON_PURPLE, NEON_ORANGE, "#FFD700", "#FF4B8C"],
            textinfo="label+percent", textposition="outside",
            hole=0.45,
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )])
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=400,
            title=dict(text="Nguồn lực phân bổ theo vùng (Tỷ VND)", x=0.5, font={"color": NEON_PURPLE, "size": 14}),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        )
        st.plotly_chart(fig_pie, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    if HAS_PULP:
        sorted_regions = sorted(zip(grdp, regions_vn, ai, digital, fdi, allocations * 100, resource_values), reverse=True)
        top_r = sorted_regions[0]
        bot_r = sorted_regions[-1]
        gini_alloc = 1 - np.sum(allocations) ** 2 / np.sum(allocations ** 2) / n_regions
        equal_alloc_vals = np.ones(n_regions) / n_regions * total_resource
        dev_gap = top_r[0] / bot_r[0] if bot_r[0] > 0 else float('inf')

        with st.expander("Vùng nào được ưu tiên đầu tư nguồn lực nhất?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Mô hình LP phân bổ <b style="color:#00D4FF;">{total_resource:.0f} Tỷ VND</b> cho 6 vùng kinh tế.
            <b style="color:#00FF88;">{top_r[1]}</b> được ưu tiên hàng đầu với GRDP = <b style="color:#FFD700;">{top_r[0]:.0f} Tỷ VND</b>,
            chiếm <b style="color:#00FF88;">{top_r[5]:.1f}%</b> nguồn lực (tương đương
            <b style="color:#00D4FF;">{top_r[6]:.0f} Tỷ VND</b>).
            Vùng này có AI Readiness = {top_r[2]:.0f}/100, chỉ số số hóa = {top_r[3]:.0f}/100,
            và FDI = {top_r[4]:.1f}B USD — cao nhất cả nước.
            Hệ số Gini của phân bổ = {gini_alloc:.3f}, cho thấy mức độ phân hóa vừa phải.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            {top_r[1]} được ưu tiên vì: (1) <b style="color:#00D4FF;">GRDP cao nhất ({top_r[0]:.0f}T VND)</b> —
            là động lực tăng trưởng chính của cả nước. (2) <b style="color:#00FF88;">FDI lớn ({top_r[4]:.1f}B USD)</b> —
            thu hút vốn nước ngoài mạnh, tạo hiệu ứng lan tỏa. (3) <b style="color:#BF00FF;">Số hóa tốt ({top_r[3]:.0f}/100)</b> —
            hạ tầng số hóa tốt đảm bảo hiệu quả sử dụng vốn. (4) <b style="color:#FFD700;">AI Readiness cao ({top_r[2]:.0f}/100)</b> —
            sẵn sàng ứng dụng công nghệ tiên tiến. Trọng số tác động ưu tiên vùng có
            GRDP lớn + FDI cao + số hóa và AI tốt.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Ưu tiên {top_r[1]} có ý nghĩa <b style="color:#00FF88;">hiệu ứng đầu tàu</b>:
            vùng này tạo ra GRDP lớn nhất, thu hút FDI nhiều nhất, và có AI Readiness cao nhất.
            Đầu tư vào vùng dẫn đầu sẽ tạo ra <b style="color:#FFD700;">hiệu ứng nhân vốn</b> cho
            toàn bộ nền kinh tế thông qua chuỗi cung ứng, logistics, và dịch vụ hỗ trợ.
            Tuy nhiên, cần đảm bảo ngưỡng tối thiểu cho các vùng khó khăn để tránh
            phân hóa quá mức (hệ số Gini = {gini_alloc:.3f} kiểm soát được mức bất bình đẳng).
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Chiến lược đầu tàu</b> — Tập trung đầu tư vào {top_r[1]}
            để tối đa hóa GDP. (2) <b style="color:#00D4FF;">Kết nối liên vùng</b> — Xây dựng
            hạ tầng giao thông, logistics kết nối {top_r[1]} với các vùng xung quanh.
            (3) <b style="color:#00D4FF;">Lan tỏa công nghệ</b> — Khuyến khích doanh nghiệp FDI trong
            vùng đầu tàu đầu tư vào chuỗi cung ứng tại vùng khác. (4) <b style="color:#00D4FF;">Chính sách đặc thù vùng</b> —
            Áp dụng cơ chế ưu đãi thuế, đất đai cho {top_r[1]} để duy trì lợi thế cạnh tranh.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Phân bổ tối ưu có giảm được chênh lệch phát triển vùng?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Phân bổ tối ưu có xu hướng <b style="color:#FFD700;">giảm bất bình đẳng</b> so với
            phân bổ đều. Khoảng cách phát triển giữa vùng dẫn đầu và vùng yếu nhất
            là <b style="color:#FF6B35;">{dev_gap:.1f}x</b> (GRDP). Mô hình LP đặt ngưỡng tối thiểu
            {min_share*100:.0f}% cho mỗi vùng, đảm bảo không vùng nào bị bỏ rơi hoàn toàn.
            Hệ số Gini = {gini_alloc:.3f} cho thấy mức phân hóa ở mức chấp nhận được.
            Vùng yếu nhất ({bot_r[1]}) nhận được {bot_r[5]:.1f}% nguồn lực
            dù có GRDP chỉ bằng {100/dev_gap:.1f}% so với vùng dẫn đầu.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            LP tối ưu không giảm khoảng cách GRDP một cách trực tiếp, mà phân bổ theo
            <b style="color:#00D4FF;">tác động của đầu tư</b>: vùng có GRDP cao + FDI lớn + AI tốt
            → hiệu quả sử dụng vốn cao → nhận nhiều hơn. Ràng buộc tối thiểu
            {min_share*100:.0f}% giới hạn mức phân hóa. Tuy nhiên, nếu mục tiêu là
            giảm khoảng cách phát triển, cần bổ sung ràng buộc cân bằng (balancing constraint)
            hoặc sử dụng mô hình đa mục tiêu như NSGA-II (Bài 7) để tối ưu đồng thời
            GDP và bình đẳng.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            Phân bổ tối ưu hiện tại phản ánh nguyên tắc <b style="color:#FFD700;">ưu tiên hiệu quả</b>
            (efficiency) hơn bình đẳng (equity). Trong ngắn hạn, điều này hợp lý vì
            tập trung nguồn lực vào vùng có sẵn nền tảng sẽ tạo ra tăng trưởng nhanh nhất.
            Tuy nhiên, trong dài hạn, bất bình đẳng quá lớn sẽ gây bất ổn xã hội
            và cản trở tăng trưởng chung. Cần kết hợp với các chính sách bù đắp
            (ODA nội vùng, đầu tư công) cho vùng khó khăn.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Kết hợp mô hình LP và NSGA-II</b> — Dùng LP để phân bổ
            nguồn lực hiệu quả, NSGA-II để cân bằng GDP và bình đẳng. (2) <b style="color:#00D4FF;">Quỹ cân bằng vùng</b> —
            Trích {min_share*100:.0f}% ngân sách để đầu tư hạ tầng cơ bản cho vùng yếu ({bot_r[1]}).
            (3) <b style="color:#00D4FF;">Chương trình hỗ trợ đặc thù</b> — Đầu tư công vào giao thông,
            điện, internet cho vùng chưa phát triển. (4) <b style="color:#00D4FF;">Giám sát hệ số Gini vùng</b> —
            Đặt ngưỡng tối đa cho phân hóa, can thiệp nếu Gini vượt 0.4.
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Vùng nào cần chính sách đặc thù và cần gì nhất?"):
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            <b style="color:#FF6B35;">{bot_r[1]}</b> là vùng cần chính sách đặc thù nhất với
            GRDP chỉ bằng <b style="color:#FFD700;">{100/dev_gap:.1f}%</b> so với vùng dẫn đầu.
            Vùng này có AI Readiness = {bot_r[2]:.0f}/100 (thấp),
            chỉ số số hóa = {bot_r[3]:.0f}/100 (thấp), và FDI = {bot_r[4]:.1f}B USD (ít nhất).
            Tổng nguồn lực nhận được: <b style="color:#00D4FF;">{bot_r[6]:.0f} Tỷ VND</b>
            ({bot_r[5]:.1f}% phân bổ). Các vùng cần ưu tiên thứ hai là
            {sorted_regions[-2][1]} và {sorted_regions[-3][1]}.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            {bot_r[1]} yếu vì: (1) <b style="color:#FF6B35;">Hạ tầng số hóa yếu ({bot_r[3]:.0f}/100)</b> —
            thiếu hạ tầng internet, trung tâm dữ liệu. (2) <b style="color:#FF6B35;">AI Readiness thấp ({bot_r[2]:.0f}/100)</b> —
            nhân lực thiếu kỹ năng số. (3) <b style="color:#FF6B35;">FDI ít ({bot_r[4]:.1f}B USD)</b> —
            khó thu hút vốn nước ngoài do hạ tầng và vị trí. (4) <b style="color:#FF6B35;">GRDP thấp</b> —
            tạo vòng xoáy nghèo: ít thu hút được đầu tư, ít đầu tư dẫn đến tăng trưởng thấp.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            {bot_r[1]} đại diện cho thách thức <b style="color:#FF6B35;">bẫy thu nhập trung bình</b>
            ở cấp vùng: nếu không được đầu tư đủ, vùng sẽ mãi ở mức thấp.
            Tuy nhiên, đầu tư không trọng điểm sẽ phí tổn mà không hiệu quả.
            Chiến lược đúng: đầu tư có trọng tâm vào <b style="color:#FFD700;">hạ tầng số hóa cơ bản</b>
            để vùng có thể tham gia vào nền kinh tế số, kết nối với các vùng dẫn đầu.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
            (1) <b style="color:#00D4FF;">Ưu tiên hạ tầng số cơ bản</b> — Phủ sóng internet,
            xây dựng trung tâm hạ tầng số tại {bot_r[1]}. (2) <b style="color:#00D4FF;">Chính sách ưu đãi đặc biệt</b> —
            Thuế, đất đai, thủ tục hành chính ưu đãi để thu hút FDI vào vùng.
            (3) <b style="color:#00D4FF;">Đào tạo nhân lực số</b> — Chương trình đào tạo kỹ năng số
            cho lao động địa phương. (4) <b style="color:#00D4FF;">Kết nối vùng</b> — Đầu tư giao thông
            kết nối {bot_r[1]} với {top_r[1]} để tạo dòng chảy hàng hóa, dịch vụ và lao động.
            </div>
            """, unsafe_allow_html=True)
    else:
        with st.expander("Mô hình LP chưa sẵn sàng"):
            st.markdown("""
            <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
            PuLP chưa được cài đặt. Vui lòng chạy lệnh: <code>pip install pulp</code>
            để kích hoạt mô hình tối ưu phân bổ nguồn lực vùng.
            Sau khi cài đặt, hệ thống sẽ tự động phân tích và hiển thị
            phân bổ nguồn lực tối ưu cho 6 vùng kinh tế Việt Nam.
            </div>
            """, unsafe_allow_html=True)
