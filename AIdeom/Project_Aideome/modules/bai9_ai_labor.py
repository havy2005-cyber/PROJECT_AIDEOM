import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_sectors, load_macro
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box

np.random.seed(42)


def estimate_ai_impact(df_s, df_m):
    col_name = "sector_name_en"
    col_auto = "automation_risk_pct"
    col_labor = "labor_million"
    col_gdp = "gdp_share_2024_pct"
    col_ai = "ai_readiness_0_100"
    col_digital = "digital_index_0_100"

    results = []
    for _, row in df_s.iterrows():
        auto_risk = row[col_auto]
        labor = row[col_labor]
        gdp_share = row[col_gdp]
        ai_readiness = row[col_ai]
        digital_idx = row[col_digital]

        displaced = labor * (auto_risk / 100.0) * 0.7
        new_jobs = labor * ((ai_readiness / 100.0) * 0.3 + (digital_idx / 100.0) * 0.2 + (gdp_share / 30.0) * 0.1)
        net_change = new_jobs - displaced
        reskill_needed = displaced * 0.6
        upskill_needed = labor * 0.3 * (ai_readiness / 100.0)

        results.append({
            "sector": row[col_name],
            "labor": labor,
            "auto_risk": auto_risk,
            "ai_readiness": ai_readiness,
            "displaced": displaced,
            "new_jobs": new_jobs,
            "net_change": net_change,
            "reskill": reskill_needed,
            "upskill": upskill_needed,
        })

    return pd.DataFrame(results)


def run():

    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Đánh giá tác động của AI đến thị trường lao động 10 ngành kinh tế.",
        "Xác định lao động bị thay thế và lao động mới được tạo ra.",
        "Đề xuất chính sách đào tạo và chuyển đổi nghề nghiệp.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)
    st.markdown("- Lao động (Triệu): Quy mô lao động theo ngành")
    st.markdown("- Automation Risk (%): Tỷ lệ rủi ro tự động hóa của ngành")
    st.markdown("- AI Readiness (0-100): Mức độ sẵn sàng ứng dụng AI")
    st.markdown("- Chỉ số số hóa (0-100): Mức độ số hóa ngành")

    df_s = load_sectors()
    df_m = load_macro()
    if df_s is None or df_m is None:
        st.warning("Không tìm thấy dữ liệu cần thiết.")
        return

    # Cấu hình
    st.markdown(section_header("Cấu hình mô hình", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(3)
    ai_adoption_rate = cfg_cols[0].slider("Tỷ lệ chấp nhận AI (%)", 10, 80, 40)
    automation_multiplier = cfg_cols[1].slider("Hệ số tự động hóa", 0.5, 1.5, 1.0)
    new_job_multiplier = cfg_cols[2].slider("Hệ số việc làm mới", 0.5, 2.0, 1.0)

    impact_df = estimate_ai_impact(df_s, df_m)
    impact_df["displaced"] *= automation_multiplier
    impact_df["new_jobs"] *= new_job_multiplier
    impact_df["net_change"] = impact_df["new_jobs"] - impact_df["displaced"]

    total_labor = impact_df["labor"].sum()
    total_displaced = impact_df["displaced"].sum()
    total_new = impact_df["new_jobs"].sum()
    total_net = impact_df["net_change"].sum()

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # ---- 9.4.1 Lao động bị ảnh hưởng ----
    st.markdown('<div class="section-header-purple">9.4.1 Lao động bị ảnh hưởng</div>', unsafe_allow_html=True)

    disp1_df = impact_df[["sector", "labor", "auto_risk", "ai_readiness", "displaced"]].copy()
    disp1_df.columns = ["Ngành", "LĐ hiện tại (M)", "Rủi ro TD (%)", "AI Readiness", "Bị thay thế (M)"]
    disp1_df = disp1_df.sort_values("Bị thay thế (M)", ascending=False)
    st.dataframe(disp1_df.style.format({
        "LĐ hiện tại (M)": "{:.2f}",
        "Rủi ro TD (%)": "{:.1f}",
        "AI Readiness": "{:.0f}",
        "Bị thay thế (M)": "{:.3f}",
    }), width='stretch')

    # KPIs
    kpi_cols = st.columns(5)
    pct_disp = total_displaced / total_labor * 100
    pct_new = total_new / total_labor * 100
    pct_net = total_net / total_labor * 100
    avg_ai = (impact_df["ai_readiness"] * impact_df["labor"]).sum() / total_labor

    with kpi_cols[0]:
        render_html(kpi_card(f"{total_labor:.1f}M", "Tổng lao động", "10 ngành", NEON_BLUE))
    with kpi_cols[1]:
        render_html(kpi_card(f"{total_displaced:.2f}M", "Bị thay thế", f"{pct_disp:.1f}% tổng LĐ", NEON_ORANGE))
    with kpi_cols[2]:
        render_html(kpi_card(f"{total_new:.2f}M", "Việc làm mới", f"{pct_new:.1f}% tổng LĐ", NEON_GREEN))
    with kpi_cols[3]:
        render_html(kpi_card(f"{total_net:+.2f}M", "Thay đổi ròng", f"{pct_net:+.1f}%", NEON_PURPLE if total_net >= 0 else NEON_ORANGE))
    with kpi_cols[4]:
        render_html(kpi_card(f"{avg_ai:.0f}", "AI Readiness TB", "Bình quân", "#FFD700"))

    # ---- 9.4.2 Lao động mới ----
    st.markdown(section_header("9.4.2 Lao động mới được tạo ra", "purple"), unsafe_allow_html=True)

    disp2_df = impact_df[["sector", "new_jobs", "ai_readiness", "net_change"]].copy()
    disp2_df.columns = ["Ngành", "Việc làm mới (M)", "AI Readiness", "Thay đổi ròng (M)"]
    disp2_df = disp2_df.sort_values("Việc làm mới (M)", ascending=False)
    st.dataframe(disp2_df.style.format({
        "Việc làm mới (M)": "{:.3f}",
        "AI Readiness": "{:.0f}",
        "Thay đổi ròng (M)": "{:+.3f}",
    }), width='stretch')

    # ---- 9.4.3 Chuyển dịch việc làm ----
    st.markdown(section_header("9.4.3 Chuyển dịch việc làm", "green"), unsafe_allow_html=True)

    reskill_df = impact_df[["sector", "displaced", "reskill", "upskill"]].copy()
    reskill_df.columns = ["Ngành", "LĐ bị thay thế (M)", "Cần đào tạo lại (M)", "Cần nâng cao kỹ năng (M)"]
    reskill_df = reskill_df.sort_values("LĐ bị thay thế (M)", ascending=False)
    st.dataframe(reskill_df.style.format({
        "LĐ bị thay thế (M)": "{:.3f}",
        "Cần đào tạo lại (M)": "{:.3f}",
        "Cần nâng cao kỹ năng (M)": "{:.3f}",
    }), use_container_width=True)

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    import plotly.graph_objects as go

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig1 = go.Figure()
        sorted_disp = impact_df.sort_values("displaced", ascending=True)
        fig1.add_trace(go.Bar(
            y=[s[:20] for s in sorted_disp["sector"]],
            x=list(sorted_disp["displaced"]),
            orientation="h", name="Bị thay thế",
            marker_color=NEON_ORANGE, opacity=0.8,
            hovertemplate="%{y}: %{x:.3f}M bị thay thế<extra></extra>",
        ))
        fig1.add_trace(go.Bar(
            y=[s[:20] for s in sorted_disp["sector"]],
            x=list(sorted_disp["new_jobs"]),
            orientation="h", name="Việc làm mới",
            marker_color=NEON_GREEN, opacity=0.8,
            hovertemplate="%{y}: %{x:.3f}M việc mới<extra></extra>",
        ))
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=420,
            xaxis=dict(gridcolor="#1F2A5A", title="Triệu người"),
            yaxis=dict(gridcolor="#1F2A5A", title=""),
            barmode="group",
            title=dict(text="Bị thay thế vs Việc làm mới theo ngành", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_cols[1]:
        fig2 = go.Figure()
        sorted_net = impact_df.sort_values("net_change", ascending=False)
        colors = [NEON_GREEN if v >= 0 else NEON_ORANGE for v in sorted_net["net_change"]]
        fig2.add_trace(go.Bar(
            x=[s[:20] for s in sorted_net["sector"]],
            y=list(sorted_net["net_change"]),
            marker_color=colors, opacity=0.85,
            hovertemplate="%{x}: %{y:+.3f}M<extra></extra>",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=420,
            xaxis=dict(gridcolor="#1F2A5A", tickangle=30, title=""),
            yaxis=dict(gridcolor="#1F2A5A", title="Thay đổi ròng (M người)"),
            shapes=[dict(type="line", x0=-0.5, x1=len(impact_df)-0.5, y0=0, y1=0,
                         xref="paper", yref="y", line=dict(color="white", width=1, dash="dash"))],
            title=dict(text="Thay đổi ròng việc làm theo ngành", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    disp_sorted = impact_df.sort_values("displaced", ascending=False)
    top_disp = disp_sorted.iloc[0]
    top_disp_name = top_disp["sector"]
    top_disp_val = top_disp["displaced"]
    total_reskill = impact_df["reskill"].sum()
    total_upskill = impact_df["upskill"].sum()
    net_positive = total_net > 0
    eff_ratio = total_new / max(total_displaced, 0.01)
    avg_auto_risk = (impact_df["auto_risk"] * impact_df["labor"]).sum() / total_labor
    avg_ai_readiness = (impact_df["ai_readiness"] * impact_df["labor"]).sum() / total_labor

    with st.expander("AI tác động như thế nào đến thị trường lao động 10 ngành?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Với tỷ lệ chấp nhận AI = <b style="color:#00D4FF;">{ai_adoption_rate}%</b>,
        mô hình ước tính: <b style="color:#FF6B35;">{total_displaced:.2f} triệu lao động</b>
        bị thay thế, nhưng đồng thời <b style="color:#00FF88;">{total_new:.2f} triệu việc làm mới</b>
        được tạo ra. Thay đổi ròng: <b style="color:#FFD700;">{total_net:+.2f} triệu</b>
        ({pct_net:+.1f}% tổng lao động {total_labor:.1f}M). Hiệu suất chuyển đổi:
        <b style="color:#FFD700;">{eff_ratio:.2f}x</b> — mỗi lao động bị thay thế tạo ra {eff_ratio:.2f}
        việc làm mới. Ngành chịu tác động mạnh nhất: <b style="color:#FF6B35;">{top_disp_name}</b>
        với {top_disp_val:.3f}M lao động bị thay thế.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Tác động kép đến từ hai cơ chế đối lập: (1) <b style="color:#FF6B35;">Thay thế (Displacement)</b> —
        AI và tự động hóa thay thế các công việc lặp lại, dễ tự động hóa.
        Rủi ro tự động hóa trung bình = <b style="color:#FF6B35;">{avg_auto_risk:.1f}%</b>,
        cao nhất ở các ngành sản xuất và chế biến. (2) <b style="color:#00FF88;">Tạo mới (Creation)</b> —
        AI tạo ra các ngành nghề mới: phân tích dữ liệu, vận hành robot, quản lý hệ thống số.
        AI Readiness trung bình = <b style="color:#00FF88;">{avg_ai_readiness:.0f}/100</b>,
        cho thấy nền kinh tế có năng lực nhất định để tạo việc làm mới.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Kết quả <b style="color:#00FF88;">{("tích cực" if net_positive else "cần lưu ý")}</b>:
        {"thị trường lao động có thể hấp thụ" if net_positive else "vẫn có thể tạo thêm việc làm"} số lao động bị thay thế
        nhờ các ngành mới nổi. Tuy nhiên, vấn đề cốt lõi là <b style="color:#FFD700;">bất đồng bộ kỹ năng</b>:
        lao động bị thay thế trong ngành cũ chưa chắc đã có kỹ năng cho việc làm mới.
        Hiệu suất chuyển đổi {eff_ratio:.2f}x cho thấy về trung bình mỗi lao động bị thay thế
        có thể tìm được {eff_ratio:.2f} việc làm mới, nhưng phân bổ theo ngành và khu vực
        có thể rất không đồng đều.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Chương trình đào tạo lại quy mô lớn</b> — Hỗ trợ {total_reskill:.2f}M
        lao động bị thay thế chuyển sang ngành mới. (2) <b style="color:#00D4FF;">Nâng cao kỹ năng số</b> —
        Đào tạo {total_upskill:.2f}M lao động nâng cao kỹ năng AI và số hóa.
        (3) <b style="color:#00D4FF;">Ưu tiên ngành tạo việc làm mới</b> — CNTT, Tài chính, Logistics
        cần được đầu tư để tạo thêm việc làm. (4) <b style="color:#00D4FF;">An sinh xã hội</b> —
        Chính sách bảo hiểm thất nghiệp và hỗ trợ thu nhập cho lao động chuyển đổi.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Ngành nào bị ảnh hưởng mạnh nhất và cần ưu tiên hỗ trợ?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        <b style="color:#FF6B35;">{top_disp_name}</b> chịu tác động mạnh nhất với
        <b style="color:#FFD700;">{top_disp_val:.3f} triệu lao động</b> bị thay thế
        (rủi ro tự động hóa = {top_disp["auto_risk"]:.1f}%, AI Readiness = {top_disp["ai_readiness"]:.0f}/100).
        Đây là ngành có rủi ro tự động hóa cao nhưng AI Readiness thấp —
        nghĩa là dễ bị thay thế bởi AI nhưng lại khó tạo việc làm mới.
        Các ngành như chế biến, sản xuất, logistics truyền thống nằm trong nhóm
        rủi ro cao nhất. Ngược lại, ngành có AI Readiness cao nhất
        (CNTT, Tài chính) có ít rủi ro bị thay thế và nhiều cơ hội việc làm mới.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {top_disp_name} bị ảnh hưởng mạnh vì: (1) <b style="color:#FF6B35;">Rủi ro tự động hóa cao</b> —
        nhiều công đoạn lặp lại, dễ thay thế bằng máy móc và AI. (2) <b style="color:#FF6B35;">AI Readiness thấp</b> —
        {top_disp["ai_readiness"]:.0f}/100 cho thấy ngành chưa sẵn sàng ứng dụng AI để tạo việc làm mới.
        (3) <b style="color:#FF6B35;">Quy mô lao động lớn</b> — {top_disp["labor"]:.2f}M lao động,
        nên dù tỷ lệ bị thay thế vừa phải, con số tuyệt đối vẫn rất lớn. (4) <b style="color:#FF6B35;">Kỹ năng chuyên môn hẹp</b> —
        lao động ngành này có kỹ năng khó chuyển đổi sang ngành khác.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {top_disp_name} là ngành có <b style="color:#FF6B35;">rủi ro thất nghiệp cơ cấu</b> cao nhất.
        Số lao động bị thay thế {top_disp_val:.3f}M cần được chuyển đổi sang ngành khác,
        nếu không sẽ tạo ra áp lực thất nghiệp và bất bình đẳng xã hội.
        Đồng thời, đây cũng là ngành có <b style="color:#FFD700;">tiềm năng tăng năng suất lớn</b>
        nếu áp dụng AI đúng cách: tự động hóa các công đoạn lặp lại, để lao động tập trung
        vào công việc sáng tạo và giá trị cao hơn. Vấn đề là cân bằng giữa
        tăng năng suất và bảo vệ việc làm.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Chương trình chuyển đổi nghề nghiệp cho {top_disp_name}</b> —
        Đào tạo kỹ năng số, kỹ năng mềm cho {top_disp_val:.3f}M lao động bị thay thế.
        (2) <b style="color:#00D4FF;">Nâng cao AI Readiness ngành</b> — Từ {top_disp["ai_readiness"]:.0f} lên 50+
        bằng cách đầu tư vào đào tạo, trang thiết bị số, và quản lý công nghệ.
        (3) <b style="color:#00D4FF;">Hỗ trợ doanh nghiệp tự động hóa có trách nhiệm</b> —
        Khuyến khích doanh nghiệp giữ chân và chuyển đổi lao động thay vì sa thải.
        (4) <b style="color:#00D4FF;">Xây dựng mạng lưới an sinh</b> — Bảo hiểm thất nghiệp,
        cho vay ưu đãi tái đào tạo cho lao động ngành {top_disp_name}.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Chính sách đào tạo và chuyển đổi nghề nghiệp nên triển khai thế nào?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Tổng nhu cầu đào tạo: <b style="color:#FFD700;">{total_reskill:.2f} triệu lao động</b>
        cần đào tạo lại hoàn toàn và <b style="color:#00D4FF;">{total_upskill:.2f} triệu lao động</b>
        cần nâng cao kỹ năng. Với tỷ lệ chấp nhận AI = {ai_adoption_rate}% và hệ số
        tự động hóa = {automation_multiplier:.1f}x, con số này phản ánh kịch bản
        {ai_adoption_rate}% AI được triển khai trong các doanh nghiệp.
        Ưu tiên đào tạo nên tập trung vào: (1) <b style="color:#00FF88;">Ngành có nhiều lao động bị thay thế</b>.
        (2) <b style="color:#00FF88;">Ngành có nhiều việc làm mới cần tuyển dụng</b> (CNTT, Tài chính, Logistics).
        (3) <b style="color:#00FF88;">Kỹ năng số cơ bản cho toàn bộ lực lượng lao động</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nhu cầu đào tạo lớn đến từ: (1) <b style="color:#FF6B35;">Bất đồng bộ kỹ năng cấu trúc</b> —
        Kỹ năng của lao động bị thay thế (thao tác thủ công, giám sát máy) không phù hợp
        với kỹ năng cần cho việc làm mới (lập trình, phân tích dữ liệu, quản lý hệ thống số).
        (2) <b style="color:#FF6B35;">Tốc độ thay đổi công nghệ nhanh</b> — AI phát triển rất nhanh,
        đòi hỏi lao động phải liên tục cập nhật kỹ năng. (3) <b style="color:#FF6B35;">Chênh lệch thế hệ</b> —
        Lao động lớn tuổi khó thích ứng hơn lao động trẻ. (4) <b style="color:#FF6B35;">Chi phí đào tạo</b> —
        {total_reskill:.2f}M người cần đào tạo lại đòi hỏi nguồn lực rất lớn từ ngân sách nhà nước và doanh nghiệp.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nếu không đào tạo kịp thời, {total_displaced:.2f}M lao động bị thay thế sẽ
        <b style="color:#FF6B35;">tăng tỷ lệ thất nghiệp cơ cấu</b>, gây bất ổn xã hội.
        Ngược lại, đầu tư vào đào tạo sẽ tạo ra <b style="color:#00FF88;">vòng xoắn tích cực</b>:
        lao động có kỹ năng số → năng suất tăng → thu nhập tăng → tiêu dùng tăng →
        tăng trưởng kinh tế. Chi phí đào tạo {total_reskill:.2f}M + {total_upskill:.2f}M lao động
        là <b style="color:#FFD700;">đầu tư mang tính chiến lược</b>, hoàn toàn có thể thu hồi
        qua tăng năng suất và giảm chi phí an sinh xã hội.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Quỹ đào tạo quốc gia</b> — Trích 1% GDP cho chương trình
        đào tạo lại và nâng cao kỹ năng, ưu tiên ngành có nhiều lao động bị thay thế.
        (2) <b style="color:#00D4FF;">Hợp tác doanh nghiệp - trường đào tạo</b> — Chương trình
        đào tạo gắn với nhu cầu doanh nghiệp, đảm bảo việc làm sau đào tạo.
        (3) <b style="color:#00D4FF;">Học bổng kỹ năng số</b> — Miễn phí đào tạo cho lao động
        chuyển đổi ngành, đặc biệt ưu tiên chuyển sang CNTT, Tài chính số, Logistics thông minh.
        (4) <b style="color:#00D4FF;">Đào tạo liên tục (lifelong learning)</b> — Xây dựng nền tảng
        học trực tuyến miễn phí cho toàn bộ lực lượng lao động.
        (5) <b style="color:#00D4FF;">Hỗ trợ thu nhập trong thời gian đào tạo</b> —
        Trợ cấp cho lao động mất việc trong thời gian chuyển đổi nghề.
        </div>
        """, unsafe_allow_html=True)
