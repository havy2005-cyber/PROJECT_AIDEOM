import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_regions
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.helpers import vietnamese_region_name, hex_to_rgba


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Xếp hạng năng lực phát triển số của 6 vùng kinh tế Việt Nam.",
        "Đánh giá đa tiêu chí: số hóa, AI Readiness, Internet, FDI, R&D.",
        "Xác định vùng dẫn đầu và vùng cần ưu tiên đầu tư.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)

    df = load_regions()
    if df is None:
        st.warning("Không tìm thấy dữ liệu vietnam_regions_2024.csv.")
        return

    data_desc = [
        "Chỉ số số hóa (0-100): Mức độ số hóa kinh tế vùng.",
        "AI Readiness (0-100): Sự sẵn sàng ứng dụng AI.",
        "Internet (%): Tỷ lệ sử dụng internet.",
        "FDI (Tỷ USD): Vốn FDI đăng ký.",
        "R&D (%): Cường độ nghiên cứu phát triển.",
    ]
    for d in data_desc:
        st.markdown(f"- {d}")

    col_rname = "region_name_en"
    col_grdp = "grdp_trillion_VND"
    col_grdp_pc = "grdp_per_capita_million_VND"
    col_fdi = "fdi_registered_billion_USD"
    col_export = "exports_billion_USD"
    col_digital = "digital_index_0_100"
    col_ai = "ai_readiness_0_100"
    col_growth = "grdp_growth_pct"
    col_rd = "rd_intensity_pct"
    col_internet = "internet_penetration_pct"
    col_labor = "trained_labor_pct"

    regions = df[col_rname].tolist()
    regions_vn = [vietnamese_region_name(r) for r in regions]
    grdp = df[col_grdp].values.astype(float)
    grdp_pc = df[col_grdp_pc].values.astype(float)
    fdi = df[col_fdi].values.astype(float)
    digital = df[col_digital].values.astype(float)
    ai = df[col_ai].values.astype(float)
    rd = df[col_rd].values.astype(float)
    internet = df[col_internet].values.astype(float)
    n = len(regions)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình
    st.markdown(section_header("Cấu hình tiêu chí TOPSIS", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(4)
    w_digital = cfg_cols[0].slider("Trọng số: Số hóa", 0.0, 1.0, 0.25, 0.05)
    w_ai = cfg_cols[1].slider("Trọng số: AI Readiness", 0.0, 1.0, 0.25, 0.05)
    w_internet = cfg_cols[2].slider("Trọng số: Internet", 0.0, 1.0, 0.20, 0.05)
    w_fdi = cfg_cols[3].slider("Trọng số: FDI", 0.0, 1.0, 0.20, 0.05)
    w_rd = st.slider("Trọng số: R&D", 0.0, 1.0, 0.10, 0.05)

    total_w = w_digital + w_ai + w_internet + w_fdi + w_rd
    if abs(total_w) < 1e-9:
        total_w = 1.0
    w_digital /= total_w
    w_ai /= total_w
    w_internet /= total_w
    w_fdi /= total_w
    w_rd /= total_w

    criteria_cols = [col_digital, col_ai, col_internet, col_fdi, col_rd]
    weights = np.array([w_digital, w_ai, w_internet, w_fdi, w_rd])

    mat = df[criteria_cols].values.astype(float)
    norm_mat = mat / np.sqrt((mat ** 2).sum(axis=0))
    weighted = norm_mat * weights
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    dist_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    closeness = dist_neg / (dist_pos + dist_neg)

    df_result = df[[col_rname, col_grdp, col_fdi, col_digital, col_ai, col_growth, col_rd, col_internet, col_grdp_pc]].copy()
    df_result["TOPSIS_Score"] = closeness
    df_result["Rank"] = df_result["TOPSIS_Score"].rank(ascending=False).astype(int)
    df_result["Ten_VN"] = regions_vn
    df_result = df_result.sort_values("Rank")

    # ---- 6.4.1 Chuẩn hóa ----
    st.markdown('<div class="section-header-purple">6.4.1 Chuẩn hóa dữ liệu</div>', unsafe_allow_html=True)

    norm_display = pd.DataFrame({
        "Vùng": regions_vn,
        "Số hóa chuẩn": np.round(mat[:, 0] / np.sqrt((mat[:, 0]**2).sum()), 4),
        "AI chuẩn": np.round(mat[:, 1] / np.sqrt((mat[:, 1]**2).sum()), 4),
        "Internet chuẩn": np.round(mat[:, 2] / np.sqrt((mat[:, 2]**2).sum()), 4),
        "FDI chuẩn": np.round(mat[:, 3] / np.sqrt((mat[:, 3]**2).sum()), 4),
        "R&D chuẩn": np.round(mat[:, 4] / np.sqrt((mat[:, 4]**2).sum()), 4),
    })
    st.dataframe(norm_display.style.format({
        "Số hóa chuẩn": "{:.4f}",
        "AI chuẩn": "{:.4f}",
        "Internet chuẩn": "{:.4f}",
        "FDI chuẩn": "{:.4f}",
        "R&D chuẩn": "{:.4f}",
    }), width='stretch')

    render_html(glass_card(f"""
    <div class="body-text" style="line-height:1.8;">
        <b style="color:#00D4FF;">Trọng số đã chuẩn hóa:</b><br>
        Số hóa={w_digital:.0%}, AI={w_ai:.0%}, Internet={w_internet:.0%}, FDI={w_fdi:.0%}, R&D={w_rd:.0%}
    </div>
    """))

    # KPIs
    kpi_cols = st.columns(6)
    rank_colors = [NEON_BLUE, NEON_GREEN, NEON_PURPLE, NEON_ORANGE, "#FFD700", "#FF4B8C"]
    for rank_idx, (_, row) in enumerate(df_result.iterrows()):
        with kpi_cols[rank_idx]:
            render_html(kpi_card(
                f"#{int(row['Rank'])}", row["Ten_VN"][:18], f"{row['TOPSIS_Score']:.4f}", rank_colors[rank_idx]
            ))

    # ---- 6.4.2 Khoảng cách TOPSIS ----
    st.markdown(section_header("6.4.2 Khoảng cách TOPSIS", "purple"), unsafe_allow_html=True)

    dist_df = pd.DataFrame({
        "Vùng": df_result["Ten_VN"],
        "Khoảng cách dương (d⁺)": np.round(dist_pos[df_result.index], 4),
        "Khoảng cách âm (d⁻)": np.round(dist_neg[df_result.index], 4),
        "Điểm TOPSIS Cᵢ": np.round(closeness[df_result.index], 4),
    })
    st.dataframe(dist_df.style.format({
        "Khoảng cách dương (d⁺)": "{:.4f}",
        "Khoảng cách âm (d⁻)": "{:.4f}",
        "Điểm TOPSIS Cᵢ": "{:.4f}",
    }), width='stretch')

    # ---- 6.4.3 Xếp hạng vùng ----
    st.markdown(section_header("6.4.3 Xếp hạng vùng", "green"), unsafe_allow_html=True)

    disp_df = df_result[["Rank", "Ten_VN", col_grdp, col_fdi, col_digital, col_ai, col_rd, "TOPSIS_Score"]].copy()
    disp_df.columns = ["Hạng", "Vùng (VI)", "GRDP (Tỷ VND)", "FDI (B USD)", "Số hóa", "AI", "R&D (%)", "Điểm TOPSIS"]
    st.dataframe(disp_df.style.format({
        "GRDP (Tỷ VND)": "{:.1f}",
        "FDI (B USD)": "{:.1f}",
        "Số hóa": "{:.0f}",
        "AI": "{:.0f}",
        "R&D (%)": "{:.2f}",
        "Điểm TOPSIS": "{:.4f}",
    }), width='stretch')

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    import plotly.graph_objects as go

    sorted_df = df_result.sort_values("TOPSIS_Score", ascending=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=[v[:22] for v in sorted_df["Ten_VN"]],
            x=list(sorted_df["TOPSIS_Score"]),
            orientation="h",
            marker_color=rank_colors,
            opacity=0.85,
            hovertemplate="Vùng: %{y}<br>Điểm: %{x:.4f}<extra></extra>",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=400,
            xaxis=dict(gridcolor="#1F2A5A", title="Điểm TOPSIS"),
            yaxis=dict(gridcolor="#1F2A5A", title=""),
            title=dict(text="Điểm TOPSIS theo vùng", x=0.5, font={"color": NEON_BLUE, "size": 14}),
            hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
        )
        st.plotly_chart(fig_bar, width='stretch')

    with chart_cols[1]:
        fig_pie = go.Figure(data=[go.Pie(
            labels=sorted_df["Ten_VN"],
            values=list(sorted_df["TOPSIS_Score"]),
            marker_colors=rank_colors,
            textinfo="label+percent", textposition="outside",
            hole=0.45,
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )])
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E0E0FF"}, height=400,
            title=dict(text="Tỷ lệ điểm TOPSIS theo vùng", x=0.5, font={"color": NEON_PURPLE, "size": 14}),
            legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        )
        st.plotly_chart(fig_pie, width='stretch')

    # Radar Top 3
    fig_radar = go.Figure()
    top3 = df_result.head(3)
    radar_labels = ["Số hóa", "AI", "Internet", "FDI", "R&D"]
    for i, (_, row) in enumerate(top3.iterrows()):
        vals = [
            row[col_digital] / 100,
            row[col_ai] / 100,
            row[col_internet] / 100,
            row[col_fdi] / 25,
            row[col_rd] / 1.0,
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=list(vals) + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor=hex_to_rgba(rank_colors[i], 0.15),
            line=dict(color=rank_colors[i], width=2),
            name=f"#{int(row['Rank'])} {row['Ten_VN'][:18]}",
        ))
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=450,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="#1F2A5A", rotation=90),
            radialaxis=dict(gridcolor="#1F2A5A", range=[0, 1]),
        ),
        title=dict(text="Radar Top 3 vùng dẫn đầu", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
    )
    st.plotly_chart(fig_radar, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    top1 = df_result.iloc[0]
    top2 = df_result.iloc[1]
    top3 = df_result.iloc[2]
    bot1 = df_result.iloc[-1]
    avg_topsis = df_result["TOPSIS_Score"].mean()
    score_gap = top1["TOPSIS_Score"] - bot1["TOPSIS_Score"]
    digital_gap = top1[col_digital] - bot1[col_digital]
    ai_gap = top1[col_ai] - bot1[col_ai]

    with st.expander("Vùng nào dẫn đầu về năng lực số hóa và AI?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        <b style="color:#00FF88;">{top1['Ten_VN']}</b> xếp hạng <b style="color:#FFD700;">#1</b>
        với điểm TOPSIS = <b style="color:#00D4FF;">{top1['TOPSIS_Score']:.4f}</b>,
        cao hơn mức trung bình {avg_topsis:.4f} tới <b style="color:#FFD700;">{((top1['TOPSIS_Score']/avg_topsis)-1)*100:.1f}%</b>.
        Ba vùng dẫn đầu: <b style="color:#FFD700;">{top1['Ten_VN']}</b> (điểm {top1['TOPSIS_Score']:.4f}),
        <b style="color:#FFD700;">{top2['Ten_VN']}</b> ({top2['TOPSIS_Score']:.4f}),
        và <b style="color:#FFD700;">{top3['Ten_VN']}</b> ({top3['TOPSIS_Score']:.4f}).
        Điểm TOPSIS của Top 3 cao hơn đáng kể so với các vùng còn lại,
        cho thấy sự tập trung rõ rệt về năng lực số hóa.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {top1['Ten_VN']} dẫn đầu nhờ: (1) <b style="color:#00D4FF;">Số hóa cao nhất ({top1[col_digital]:.0f}/100)</b> —
        hạ tầng số, thương mại điện tử, chính phủ số phát triển mạnh. (2) <b style="color:#00FF88;">AI Readiness tốt ({top1[col_ai]:.0f}/100)</b> —
        nhân lực sẵn sàng ứng dụng AI trong sản xuất và dịch vụ. (3) <b style="color:#BF00FF;">Internet phổ biến ({top1[col_internet]:.0f}%)</b> —
        tỷ lệ người dùng internet cao tạo nền tảng cho kinh tế số. (4) <b style="color:#FFD700;">FDI mạnh ({top1[col_fdi]:.1f}B USD)</b> —
        vốn nước ngoài mang theo công nghệ và thực hành quản lý số. (5) <b style="color:#FF6B35;">R&D ({top1[col_rd]:.2f}%)</b> —
        đầu tư vào nghiên cứu và phát triển công nghệ.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {top1['Ten_VN']} với điểm TOPSIS {top1['TOPSIS_Score']:.4f} là <b style="color:#00FF88;">động lực số</b>
        của cả nước. Vùng này thu hút FDI công nghệ cao, tạo ra việc làm chất lượng,
        và có khả năng <b style="color:#FFD700;">lan tỏa công nghệ</b> sang các vùng lân cận.
        Trong bối cảnh chuyển đổi số quốc gia, {top1['Ten_VN']} đóng vai trò
        <b style="color:#00D4FF;">đầu tàu số</b> — nơi thử nghiệm và chứng minh hiệu quả
        của các giải pháp số hóa trước khi nhân rộng ra cả nước.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Đầu tư trọng điểm vào {top1['Ten_VN']}</b> —
        Biến vùng dẫn đầu thành trung tâm AI và số hóa của quốc gia. (2) <b style="color:#00D4FF;">Chính sách thu hút FDI công nghệ</b> —
        Ưu tiên dự án có chuyển giao công nghệ, R&D tại {top1['Ten_VN']}. (3) <b style="color:#00D4FF;">Xây dựng hệ sinh thái số</b> —
        Thu hút startup công nghệ, trung tâm R&D, và đại học nghiên cứu AI.
        (4) <b style="color:#00D4FF;">Lan tỏa sang vùng lân cận</b> — Kết nối {top1['Ten_VN']}
        với {top2['Ten_VN']} và {top3['Ten_VN']} để tạo cụm công nghệ liên vùng.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Khoảng cách số giữa các vùng phản ánh vấn đề gì?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Khoảng cách số giữa vùng dẫn đầu và vùng yếu nhất rất lớn:
        Số hóa chênh <b style="color:#FF6B35;">{digital_gap:.0f} điểm</b> ({top1[col_digital]:.0f} vs {bot1[col_digital]:.0f}),
        AI Readiness chênh <b style="color:#FF6B35;">{ai_gap:.0f} điểm</b> ({top1[col_ai]:.0f} vs {bot1[col_ai]:.0f}).
        Điểm TOPSIS chênh lệch: <b style="color:#FFD700;">{score_gap:.4f}</b>
        (từ {bot1['TOPSIS_Score']:.4f} đến {top1['TOPSIS_Score']:.4f}).
        <b style="color:#FF6B35;">{bot1['Ten_VN']}</b> xếp hạng #{int(bot1['Rank'])} —
        khoảng cách số cản trở phát triển kinh tế của vùng này.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Khoảng cách số xuất phát từ nhiều nguyên nhân cấu trúc: (1) <b style="color:#FF6B35;">Phân bổ dân cư không đều</b> —
        Dân cư tập trung ở vùng kinh tế trọng điểm, vùng khó khăn có dân số thưa.
        (2) <b style="color:#FF6B35;">Hạ tầng internet yếu</b> — Vùng khó khăn thiếu hạ tầng kỹ thuật
        để triển khai số hóa. (3) <b style="color:#FF6B35;">Thu nhập thấp</b> — Người dân vùng khó khăn
        chưa có điều kiện tiếp cận thiết bị số và internet. (4) <b style="color:#FF6B35;">Nhân lực khan hiếm</b> —
        Vùng khó tuyển dụng và giữ chân nhân lực công nghệ. (5) <b style="color:#FF6B35;">Hiệu ứng cửa ngõ</b> —
        Nguồn lực (vốn, nhân lực, công nghệ) có xu hướng tập trung ở vùng đã phát triển.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Khoảng cách số tạo ra <b style="color:#FF6B35;">bất bình đẳng kinh tế</b> kéo dài:
        vùng số hóa tốt thu hút đầu tư, tạo việc làm tốt, thu hút nhân lực —
        vùng khó khăn ngày càng tụt hậu. Nếu không can thiệp, khoảng cách này
        sẽ ngày càng lớn (hiệu ứng Matthew). Đặc biệt trong kỷ nguyên AI,
        vùng có AI Readiness thấp sẽ <b style="color:#FF6B35;">bị tụt hậu vĩnh viễn</b>
        trong năng lực cạnh tranh.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Đầu tư hạ tầng số ưu tiên</b> — Phủ sóng internet băng rộng,
        trung tâm dữ liệu, hạ tầng số hóa tại {bot1['Ten_VN']} và các vùng khó khăn.
        (2) <b style="color:#00D4FF;">Chương trình kỹ năng số miễn phí</b> — Đào tạo kỹ năng số
        cơ bản và nâng cao cho lao động vùng khó. (3) <b style="color:#00D4FF;">Chính sách thu hút nhân lực</b> —
        Hỗ trợ nhà ở, lương, điều kiện làm việc để thu hút nhân lực công nghệ
        đến vùng khó. (4) <b style="color:#00D4FF;">Mô hình chia sẻ hạ tầng</b> — Cho phép vùng
        khó tiếp cận hạ tầng số của vùng lân cận thông qua kết nối.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Chiến lược nào để cải thiện năng lực số cho vùng yếu?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        <b style="color:#FF6B35;">{bot1['Ten_VN']}</b> với điểm TOPSIS = {bot1['TOPSIS_Score']:.4f}
        cần chiến lược cải thiện toàn diện. Phân tích 5 tiêu chí TOPSIS cho thấy
        vùng này yếu trên hầu hết các tiêu chí: Số hóa = {bot1[col_digital]:.0f}/100,
        AI = {bot1[col_ai]:.0f}/100, Internet = {bot1[col_internet]:.0f}%,
        FDI = {bot1[col_fdi]:.1f}B USD, R&D = {bot1[col_rd]:.2f}%.
        Không thể cải thiện tất cả cùng lúc — cần ưu tiên có trọng tâm theo lộ trình.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {bot1['Ten_VN']} yếu toàn diện vì các yếu tố bổ trợ lẫn nhau tạo thành
        <b style="color:#FF6B35;">vòng xoáy tiêu cực</b>: (1) Hạ tầng kém → khó thu hút FDI → ít việc làm công nghệ.
        (2) Ít việc làm công nghệ → nhân lực di cư → thiếu nguồn nhân lực số.
        (3) Thiếu nhân lực số → doanh nghiệp không đầu tư → hạ tầng không phát triển.
        Để phá vỡ vòng xoáy này, cần một <b style="color:#FFD700;">can thiệp mạnh từ bên ngoài</b>
        (chính phủ, ODA, FDI chiến lược).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nếu cải thiện được năng lực số của {bot1['Ten_VN']}, cả nước sẽ hưởng lợi:
        (1) <b style="color:#00FF88;">Mở rộng thị trường số</b> — Thêm hàng triệu người tiếp cận
        thương mại điện tử và dịch vụ số. (2) <b style="color:#00FF88;">Giảm bất bình đẳng</b> —
        Thu hẹp khoảng cách GDP per capita giữa các vùng. (3) <b style="color:#00FF88;">Tăng năng suất lao động</b> —
        Lao động được đào tạo kỹ năng số sẽ năng suất cao hơn. (4) <b style="color:#00FF88;">Phân tán rủi ro</b> —
        Nền kinh tế không phụ thuộc quá nhiều vào 1-2 vùng dẫn đầu.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Giai đoạn 1 (2026-2028): Hạ tầng</b> — Đầu tư internet băng rộng,
        trung tâm hành chính số, cơ sở dữ liệu vùng. (2) <b style="color:#00D4FF;">Giai đoạn 2 (2028-2030): Nhân lực</b> —
        Chương trình đào tạo kỹ năng số, thu hút nhân lực công nghệ về.
        (3) <b style="color:#00D4FF;">Giai đoạn 3 (2030+): Kinh tế</b> — Thu hút FDI công nghệ,
        phát triển nền tảng số địa phương, kết nối với chuỗi cung ứng số của {top1['Ten_VN']}.
        (4) <b style="color:#00D4FF;">Cơ chế đặc thù</b> — Ưu đãi thuế, đất đai, hành chính
        để thu hút đầu tư số vào {bot1['Ten_VN']}.
        </div>
        """, unsafe_allow_html=True)
