import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_sectors
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.charts import bar_chart, pie_chart
from utils.helpers import vietnamese_sector_name, hex_to_rgba


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Xếp hạng ưu tiên phát triển 10 ngành kinh tế Việt Nam dựa trên nhiều tiêu chí.",
        "Sử dụng phương pháp TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).",
        "Cung cấp căn cứ khoa học cho việc phân bổ nguồn lực và chính sách phát triển ngành.",
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
        "GDP Share (%): Tỷ trọng GDP của từng ngành",
        "Xuất khẩu (B USD): Giá trị xuất khẩu của ngành",
        "Chỉ số số hóa (0-100): Mức độ số hóa của ngành",
        "AI Readiness (0-100): Sự sẵn sàng ứng dụng AI của ngành",
        "Tăng trưởng 2024 (%): Tốc độ tăng trưởng ngành năm 2024",
    ]
    for d in data_desc:
        st.markdown(f"- {d}")

    col_name = "sector_name_en"
    col_gdp = "gdp_share_2024_pct"
    col_export = "export_billion_USD"
    col_digital = "digital_index_0_100"
    col_ai = "ai_readiness_0_100"
    col_growth = "growth_rate_2024_pct"

    sectors = df[col_name].tolist()
    n = len(sectors)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # Cấu hình
    st.markdown(section_header("Cấu hình tiêu chí TOPSIS", "blue"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    w_gdp = c1.slider("Trọng số: GDP", 0.0, 1.0, 0.30, 0.05)
    w_exp = c2.slider("Trọng số: Xuất khẩu", 0.0, 1.0, 0.25, 0.05)
    w_dig = c3.slider("Trọng số: Số hóa", 0.0, 1.0, 0.20, 0.05)
    w_ai = st.slider("Trọng số: AI Readiness", 0.0, 1.0, 0.15, 0.05)
    w_gro = st.slider("Trọng số: Tăng trưởng", 0.0, 1.0, 0.10, 0.05)

    total_w = w_gdp + w_exp + w_dig + w_ai + w_gro
    if abs(total_w) < 1e-9:
        w_gdp, w_exp, w_dig, w_ai, w_gro = 0.3, 0.25, 0.2, 0.15, 0.1
        total_w = 1.0
    w_gdp /= total_w
    w_exp /= total_w
    w_dig /= total_w
    w_ai /= total_w
    w_gro /= total_w

    weights = np.array([w_gdp, w_exp, w_dig, w_ai, w_gro])
    criteria_cols = [col_gdp, col_export, col_digital, col_ai, col_growth]
    benefit_criteria = criteria_cols

    # TOPSIS calculation
    mat = df[criteria_cols].values.astype(float)
    norm_mat = mat / np.sqrt((mat ** 2).sum(axis=0))
    weighted = norm_mat * weights
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    dist_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    closeness = dist_neg / (dist_pos + dist_neg)

    df_result = df[[col_name, col_gdp, col_export, col_digital, col_ai, col_growth]].copy()
    df_result["TOPSIS_Score"] = closeness
    df_result["Rank"] = df_result["TOPSIS_Score"].rank(ascending=False).astype(int)
    df_result = df_result.sort_values("Rank")
    df_result["Ten_VN"] = df_result[col_name].apply(vietnamese_sector_name)

    # ---- 3.4.1 Chuẩn hóa dữ liệu ----
    st.markdown('<div class="section-header-purple">3.4.1 Chuẩn hóa dữ liệu</div>', unsafe_allow_html=True)

    norm_display = pd.DataFrame({
        "Ngành": sectors,
        "GDP chuẩn": np.round(mat[:, 0] / np.sqrt((mat[:, 0]**2).sum()), 4),
        "XK chuẩn": np.round(mat[:, 1] / np.sqrt((mat[:, 1]**2).sum()), 4),
        "Số hóa chuẩn": np.round(mat[:, 2] / np.sqrt((mat[:, 2]**2).sum()), 4),
        "AI chuẩn": np.round(mat[:, 3] / np.sqrt((mat[:, 3]**2).sum()), 4),
        "Tăng trưởng chuẩn": np.round(mat[:, 4] / np.sqrt((mat[:, 4]**2).sum()), 4),
    })
    st.dataframe(norm_display.style.format({
        "GDP chuẩn": "{:.4f}",
        "XK chuẩn": "{:.4f}",
        "Số hóa chuẩn": "{:.4f}",
        "AI chuẩn": "{:.4f}",
        "Tăng trưởng chuẩn": "{:.4f}",
    }), width='stretch')

    render_html(glass_card(f"""
    <div class="body-text" style="line-height:1.8;">
        <b style="color:#00D4FF;">Công thức chuẩn hóa Vector:</b> rᵢⱼ = xᵢⱼ / √(Σxᵢⱼ²)<br>
        <b style="color:#00D4FF;">Trọng số đã chuẩn hóa:</b> vᵢⱼ = wⱼ × rᵢⱼ<br>
        <b style="color:#00D4FF;">Trọng số sử dụng:</b>
        GDP={w_gdp:.0%}, Xuất khẩu={w_exp:.0%}, Số hóa={w_dig:.0%}, AI={w_ai:.0%}, Tăng trưởng={w_gro:.0%}
    </div>
    """))

    # KPIs
    kpi_cols = st.columns(4)
    for idx, (_, row) in enumerate(df_result.head(3).iterrows()):
        colors = [NEON_BLUE, NEON_GREEN, NEON_PURPLE]
        with kpi_cols[idx]:
            render_html(kpi_card(f"#{int(row['Rank'])}", row["Ten_VN"][:20], f"TOPSIS: {row['TOPSIS_Score']:.4f}", colors[idx]))
    with kpi_cols[3]:
        render_html(kpi_card(f"{df_result['TOPSIS_Score'].mean():.4f}", "Điểm TB", "TOPSIS trung bình", NEON_ORANGE))

    # ---- 3.4.2 Tính điểm ưu tiên ----
    st.markdown(section_header("3.4.2 Tính điểm ưu tiên", "purple"), unsafe_allow_html=True)

    dist_df = pd.DataFrame({
        "Ngành": df_result["Ten_VN"],
        "Khoảng cách dương (d⁺)": np.round(dist_pos[df_result.index], 4),
        "Khoảng cách âm (d⁻)": np.round(dist_neg[df_result.index], 4),
        "Điểm Cᵢ = d⁻/(d⁺+d⁻)": np.round(closeness[df_result.index], 4),
    })
    st.dataframe(dist_df.style.format({
        "Khoảng cách dương (d⁺)": "{:.4f}",
        "Khoảng cách âm (d⁻)": "{:.4f}",
        "Điểm Cᵢ = d⁻/(d⁺+d⁻)": "{:.4f}",
    }), use_container_width=True)

    # ---- 3.4.3 Xếp hạng ngành ----
    st.markdown(section_header("3.4.3 Xếp hạng ngành", "green"), unsafe_allow_html=True)

    display_df = df_result[["Rank", "Ten_VN", col_gdp, col_export, col_digital, col_ai, col_growth, "TOPSIS_Score"]].copy()
    display_df.columns = ["Hạng", "Ngành (VI)", "GDP (%)", "Xuất khẩu (B USD)", "Số hóa", "AI", "Tăng trưởng (%)", "Điểm TOPSIS"]
    st.dataframe(display_df.style.format({
        "GDP (%)": "{:.2f}",
        "Xuất khẩu (B USD)": "{:.1f}",
        "Số hóa": "{:.0f}",
        "AI": "{:.0f}",
        "Tăng trưởng (%)": "{:.2f}",
        "Điểm TOPSIS": "{:.4f}",
    }), width='stretch')

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    import plotly.graph_objects as go
    import plotly.express as px

    sorted_df = df_result.sort_values("TOPSIS_Score", ascending=True)

    fig_bar = go.Figure()
    colors_bar = [NEON_GREEN if i < 3 else NEON_BLUE if i < 7 else NEON_ORANGE for i in range(n)]
    fig_bar.add_trace(go.Bar(
        y=[v[:22] for v in sorted_df["Ten_VN"]],
        x=list(sorted_df["TOPSIS_Score"]),
        orientation="h",
        marker_color=colors_bar,
        opacity=0.85,
        hovertemplate="Ngành: %{y}<br>Điểm: %{x:.4f}<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=450,
        xaxis=dict(gridcolor="#1F2A5A", title="Điểm TOPSIS"),
        yaxis=dict(gridcolor="#1F2A5A", title=""),
        title=dict(text="Xếp hạng ngành theo TOPSIS", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_bar, width='stretch')

    # Radar Top 3
    fig_radar = go.Figure()
    top3 = df_result.head(3)
    radar_labels = ["GDP", "Xuất khẩu", "Số hóa", "AI", "Tăng trưởng"]
    radar_colors = [NEON_BLUE, NEON_GREEN, NEON_PURPLE]
    for i, (_, row) in enumerate(top3.iterrows()):
        vals = [
            row[col_gdp] / 25,
            row[col_export] / 300,
            row[col_digital] / 100,
            row[col_ai] / 100,
            row[col_growth] / 12,
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=list(vals) + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor=hex_to_rgba(radar_colors[i], 0.15),
            line=dict(color=radar_colors[i], width=2),
            name=f"#{int(row['Rank'])} {row['Ten_VN'][:18]}",
            hovertemplate=f"#{int(row['Rank'])} %{{theta}}: %{{r:.2f}}<extra></extra>",
        ))
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=450,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="#1F2A5A", rotation=90),
            radialaxis=dict(gridcolor="#1F2A5A", range=[0, 1]),
        ),
        title=dict(text="Radar Top 3 ngành ưu tiên", x=0.5, font={"color": NEON_BLUE, "size": 14}),
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
    avg_score = df_result["TOPSIS_Score"].mean()
    score_gap = top1["TOPSIS_Score"] - bot1["TOPSIS_Score"]
    score_range = df_result["TOPSIS_Score"].max() - df_result["TOPSIS_Score"].min()

    with st.expander("Ngành nào được xếp hạng ưu tiên số 1 và đánh giá ra sao?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        <b style="color:#00FF88;">{top1['Ten_VN']}</b> xếp hạng <b style="color:#FFD700;">#1</b>
        với điểm TOPSIS = <b style="color:#00D4FF;">{top1['TOPSIS_Score']:.4f}</b>,
        cao hơn điểm trung bình của toàn bộ ngành ({avg_score:.4f}).
        Điểm này phản ánh sự kết hợp tối ưu giữa: GDP = {top1[col_gdp]:.2f}%,
        Xuất khẩu = {top1[col_export]:.1f}B USD, Số hóa = {top1[col_digital]:.0f}/100,
        AI Readiness = {top1[col_ai]:.0f}/100, Tăng trưởng = {top1[col_growth]:.2f}%.
        Ba ngành dẫn đầu lần lượt là <b style="color:#FFD700;">{top1['Ten_VN']}</b>,
        <b style="color:#FFD700;">{top2['Ten_VN']}</b>, và <b style="color:#FFD700;">{top3['Ten_VN']}</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {top1['Ten_VN']} dẫn đầu nhờ: (1) <b style="color:#00D4FF;">Quy mô GDP lớn</b> — chiếm {top1[col_gdp]:.1f}% GDP,
        đóng vai trò đầu tàu kinh tế. (2) <b style="color:#00FF88;">Xuất khẩu cao ({top1[col_export]:.1f}B USD)</b> —
        tạo nguồn ngoại tệ ổn định. (3) <b style="color:#BF00FF;">Số hóa tốt ({top1[col_digital]:.0f}/100)</b> —
        quy trình sản xuất kinh doanh đã ứng dụng công nghệ số. (4) <b style="color:#FFD700;">AI Readiness cao ({top1[col_ai]:.0f}/100)</b> —
        sẵn sàng ứng dụng trí tuệ nhân tạo. (5) <b style="color:#FF6B35;">Tăng trưởng {top1[col_growth]:.2f}%</b> —
        động lực phát triển mạnh. Phương pháp TOPSIS với trọng số GDP={w_gdp:.0%}, XK={w_exp:.0%},
        Số hóa={w_dig:.0%}, AI={w_ai:.0%}, Tăng trưởng={w_gro:.0%} cho thấy đây là ngành
        có năng lực cạnh tranh toàn diện nhất.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Ngành dẫn đầu TOPSIS là <b style="color:#00FF88;">động lực tăng trưởng chính</b> của nền kinh tế.
        Với điểm {top1['TOPSIS_Score']:.4f}, ngành này đã vượt trội so với mức trung bình {avg_score:.4f}
        tới <b style="color:#FFD700;">{((top1['TOPSIS_Score']/avg_score)-1)*100:.1f}%</b>.
        Đầu tư ưu tiên vào ngành dẫn đầu sẽ tạo hiệu ứng nhân vốn cao nhất
        cho toàn bộ chuỗi kinh tế. Tuy nhiên, khoảng cách điểm TOPSIS giữa
        ngành #1 và #10 là {score_gap:.4f}, cho thấy sự phân hóa rõ rệt giữa các ngành.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Tập trung nguồn lực vào Top 3</b> — {top1['Ten_VN']}, {top2['Ten_VN']},
        {top3['Ten_VN']} cần được ưu tiên đầu tư và hỗ trợ chính sách. (2) <b style="color:#00D4FF;">Xây dựng chuỗi giá trị</b> —
        Kết nối ngành dẫn đầu với các ngành hỗ trợ để tối đa hóa hiệu ứng lan tỏa.
        (3) <b style="color:#00D4FF;">Chính sách khuyến khích số hóa</b> — Hỗ trợ doanh nghiệp Top 3
        nâng cao AI Readiness lên trên 60/100. (4) <b style="color:#00D4FF;">Giám sát định kỳ</b> —
        Đánh giá lại xếp hạng TOPSIS hàng năm để điều chỉnh ưu tiên kịp thời.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Khoảng cách giữa các ngành phản ánh vấn đề gì?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Khoảng cách điểm TOPSIS giữa ngành đứng đầu và cuối bảng là
        <b style="color:#FF6B35;">{score_gap:.4f}</b> (từ {bot1['TOPSIS_Score']:.4f} đến {top1['TOPSIS_Score']:.4f}).
        Điều này cho thấy sự <b style="color:#FFD700;">phân hóa mạnh</b> giữa các ngành kinh tế.
        Nhóm dẫn đầu (Top 3) có điểm TOPSIS trên {df_result.iloc[2]['TOPSIS_Score']:.4f},
        trong khi nhóm cuối (Bottom 3) dưới {df_result.iloc[-3]['TOPSIS_Score']:.4f}.
        Sự chênh lệch này phản ánh bất bình đẳng trong năng lực cạnh tranh giữa các ngành.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Phân hóa ngành xuất phát từ: (1) <b style="color:#FF6B35;">Chênh lệch số hóa lớn</b> —
        Ngành dẫn đầu có chỉ số số hóa cao hơn đáng kể so với ngành cuối.
        (2) <b style="color:#FF6B35;">AI Readiness không đồng đều</b> — Một số ngành đã ứng dụng AI
        trong khi ngành khác gần như chưa có nền tảng. (3) <b style="color:#FF6B35;">Xuất khẩu chênh lệch</b> —
        Ngành dẫn đầu có kim ngạch xuất khẩu gấp nhiều lần ngành cuối.
        (4) <b style="color:#FF6B35;">Tăng trưởng không đồng đều</b> — Ngành công nghệ và dịch vụ tài chính
        tăng trưởng nhanh hơn nhiều so với ngành nông nghiệp hoặc khai khoáng.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Phân hóa ngành có cả mặt tích cực và tiêu cực. Mặt tích cực: nền kinh tế có
        <b style="color:#00FF88;">đầu tàu rõ ràng</b> để tập trung phát triển. Mặt tiêu cực:
        ngành cuối bảng ({bot1['Ten_VN']}) có nguy cơ bị bỏ lại phía sau,
        tạo ra <b style="color:#FF6B35;">bất bình đẳng thu nhập</b> giữa lao động các ngành.
        Nếu xu hướng này tiếp diễn, khoảng cách sẽ ngày càng lớn hơn (hiệu ứng thượng tầng).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Chính sách thu hẹp khoảng cách</b> — Đầu tư có trọng tâm vào ngành
        có tiềm năng nhưng chưa được khai thác. (2) <b style="color:#00D4FF;">Chương trình nâng cấp ngành</b> —
        Hỗ trợ {bot1['Ten_VN']} cải thiện chỉ số số hóa và AI Readiness.
        (3) <b style="color:#00D4FF;">Đào tạo chuyển đổi nghề</b> — Di chuyển lao động từ ngành
        suy giảm sang ngành tăng trưởng. (4) <b style="color:#00D4FF;">Cơ chế chia sẻ lợi ích</b> —
        Khuyến khích ngành dẫn đầu chia sẻ công nghệ và đào tạo với ngành yếu hơn.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Nên ưu tiên sản xuất hay ứng dụng AI cho ngành cuối bảng?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        <b style="color:#FF6B35;">{bot1['Ten_VN']}</b> xếp hạng cuối với điểm TOPSIS = {bot1['TOPSIS_Score']:.4f}.
        Ngành này đối mặt với lựa chọn chiến lược: (1) Ưu tiên <b style="color:#00FF88;">ứng dụng AI</b>
        để nâng cao năng suất — phù hợp nếu có nền tảng công nghệ sẵn sàng.
        (2) Ưu tiên <b style="color:#00D4FF;">phát triển sản xuất</b> trước khi số hóa —
        phù hợp nếu ngành cần tăng quy mô trước khi hiện đại hóa.
        TOPSIS cho thấy ngành có AI Readiness và Số hóa thấp nhất đều xếp hạng yếu,
        bất kể quy mô GDP. Do đó, cả hai yếu tố đều cần được phát triển song song.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {bot1['Ten_VN']} yếu trên cả 5 tiêu chí TOPSIS: GDP share thấp, xuất khẩu hạn chế,
        số hóa ({bot1[col_digital]:.0f}/100) và AI ({bot1[col_ai]:.0f}/100) đều ở mức thấp,
        tăng trưởng chậm ({bot1[col_growth]:.2f}%). Việc chỉ tập trung một tiêu chí
        (ví dụ chỉ sản xuất hoặc chỉ AI) sẽ không cải thiện đáng kể xếp hạng tổng thể.
        Chiến lược tốt nhất là: tăng quy mô sản xuất đồng thời với nâng cao số hóa
        và AI Readiness theo lộ trình phù hợp với đặc thù ngành.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Nếu chỉ ưu tiên sản xuất mà bỏ qua số hóa, ngành sẽ mất năng lực cạnh tranh
        trong dài hạn khi các ngành khác đều số hóa. Ngược lại, nếu chỉ ưu tiên AI
        mà chưa có nền tảng sản xuất, công nghệ sẽ không phát huy tác dụng.
        Cần <b style="color:#FFD700;">chiến lược lai ghép</b>: phát triển sản xuất tạo doanh thu,
        đồng thời đầu tư có chọn lọc vào số hóa và AI ở những khâu có hiệu quả cao nhất.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Lộ trình 2 giai đoạn</b> — Giai đoạn 1: Nâng cấp số hóa cơ bản
        (từ {bot1[col_digital]:.0f} lên 40+). Giai đoạn 2: Ứng dụng AI ở các khâu
        có tiềm năng cao. (2) <b style="color:#00D4FF;">Hỗ trợ tài chính có điều kiện</b> —
        Cho vay ưu đãi gắn với KPI số hóa cụ thể. (3) <b style="color:#00D4FF;">Kết nối với ngành dẫn đầu</b> —
        Để {bot1['Ten_VN']} tham gia chuỗi cung ứng của {top1['Ten_VN']} hoặc {top2['Ten_VN']}.
        (4) <b style="color:#00D4FF;">Chính sách nhân lực</b> — Đào tạo kỹ năng số cho lao động
        ngành {bot1['Ten_VN']} để chuẩn bị cho chuyển đổi.
        </div>
        """, unsafe_allow_html=True)
