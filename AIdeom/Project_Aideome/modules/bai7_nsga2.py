import streamlit as st
import pandas as pd
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_regions
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box

np.random.seed(42)


def nsga2(population_size, n_generations, df_r, n_obj=3):
    col_grdp = "grdp_trillion_VND"
    col_digital = "digital_index_0_100"
    col_ai = "ai_readiness_0_100"
    col_labor = "trained_labor_pct"
    col_pop = "population_million"

    grdp = df_r[col_grdp].values.astype(float)
    digital = df_r[col_digital].values.astype(float)
    ai = df_r[col_ai].values.astype(float)
    trained = df_r[col_labor].values.astype(float)
    pop = df_r[col_pop].values.astype(float)
    n_regions = len(grdp)

    # Normalize each objective metric to [0, 1] so they contribute equally
    grdp_norm = grdp / grdp.max()
    digital_norm = digital / 100.0
    ai_norm = ai / 100.0
    labor_norm = (trained * pop / 100.0) / (trained * pop / 100.0).max()

    max_budget = 50.0
    min_alloc = 0.5

    def create_individual():
        indiv = np.random.rand(n_regions)
        indiv = indiv / indiv.sum() * max_budget
        return indiv

    def evaluate(indiv):
        # Return positive objective values (maximize)
        alloc = indiv
        weight = alloc / alloc.sum()
        g_gdp = np.dot(weight, grdp_norm)
        g_ai = np.dot(weight, (ai_norm + digital_norm) / 2)
        g_labor = np.dot(weight, labor_norm)
        return g_gdp, g_ai, g_labor

    def dominates(p, q):
        return all(pi >= qi for pi, qi in zip(p, q)) and any(pi > qi for pi, qi in zip(p, q))

    def sbx_crossover(p1, p2, eta=15):
        child1 = np.empty_like(p1)
        child2 = np.empty_like(p2)
        for i in range(len(p1)):
            if random.random() < 0.9:
                u = random.random()
                if u <= 0.5:
                    q = (2 * u) ** (1 / (eta + 1))
                else:
                    q = (1 / (2 - 2 * u)) ** (1 / (eta + 1))
                child1[i] = 0.5 * ((1 + q) * p1[i] + (1 - q) * p2[i])
                child2[i] = 0.5 * ((1 - q) * p1[i] + (1 + q) * p2[i])
            else:
                child1[i], child2[i] = p1[i], p2[i]
        child1 = np.clip(child1, 0, max_budget)
        child2 = np.clip(child2, 0, max_budget)
        return child1, child2

    def poly_mutate(indiv, eta=20):
        for i in range(len(indiv)):
            if random.random() < 0.1:
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 - 2 * u) ** (1 / (eta + 1))
                indiv[i] += delta * max_budget * 0.2
                indiv[i] = np.clip(indiv[i], 0, max_budget)

    def normalize_to_budget(indiv):
        indiv = np.clip(indiv, 0, max_budget)
        if indiv.sum() > 0:
            indiv = indiv / indiv.sum() * max_budget
        indiv = np.clip(indiv, min_alloc, max_budget)
        if indiv.sum() > 0:
            indiv = indiv / indiv.sum() * max_budget
        return indiv

    random.seed(42)
    np.random.seed(42)

    population = [create_individual() for _ in range(population_size)]
    population = [normalize_to_budget(ind) for ind in population]
    objectives = [evaluate(ind) for ind in population]

    for gen in range(n_generations):
        offspring = []
        for _ in range(population_size // 2):
            p1_idx, p2_idx = random.sample(range(population_size), 2)
            p1, p2 = population[p1_idx], population[p2_idx]
            child1, child2 = sbx_crossover(p1, p2)
            poly_mutate(child1)
            poly_mutate(child2)
            child1 = normalize_to_budget(child1)
            child2 = normalize_to_budget(child2)
            offspring.extend([child1, child2])

        all_inds = population + offspring
        all_objs = [evaluate(ind) for ind in all_inds]
        n_all = len(all_inds)

        # --- Fast non-dominated sorting ---
        n_dominated = [0] * n_all
        dominated_by = [[] for _ in range(n_all)]
        fronts = [[]]

        for i in range(n_all):
            for j in range(n_all):
                if i == j:
                    continue
                if dominates(all_objs[i], all_objs[j]):
                    dominated_by[i].append(j)
                elif dominates(all_objs[j], all_objs[i]):
                    n_dominated[i] += 1
            if n_dominated[i] == 0:
                fronts[0].append(i)

        k = 0
        while k < len(fronts) and fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        fronts = [f for f in fronts if f]

        # --- Crowding distance ---
        crowding = [0.0] * n_all
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    crowding[i] = float('inf')
                continue
            for obj_i in range(n_obj):
                front_sorted = sorted(front, key=lambda i: all_objs[i][obj_i])
                crowding[front_sorted[0]] = float('inf')
                crowding[front_sorted[-1]] = float('inf')
                obj_range = all_objs[front_sorted[-1]][obj_i] - all_objs[front_sorted[0]][obj_i]
                if abs(obj_range) < 1e-12:
                    continue
                for j in range(1, len(front_sorted) - 1):
                    crowding[front_sorted[j]] += (
                        (all_objs[front_sorted[j + 1]][obj_i] - all_objs[front_sorted[j - 1]][obj_i]) / obj_range
                    )

        # --- Environmental selection ---
        selected = []
        front_idx = 0
        while len(selected) + len(fronts[front_idx]) <= population_size:
            selected.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break
        if len(selected) < population_size:
            remaining = [i for i in fronts[front_idx] if i not in selected]
            remaining.sort(key=lambda i: -crowding[i])
            selected.extend(remaining[:population_size - len(selected)])

        population = [all_inds[i] for i in selected]
        objectives = [all_objs[i] for i in selected]

    # Final Pareto front from last population
    pareto_mask = [True] * len(population)
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j and dominates(objectives[j], objectives[i]):
                pareto_mask[i] = False
                break

    return population, objectives, pareto_mask


def run():
    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Tối ưu hóa đồng thời 3 mục tiêu: GDP, Việc làm và AI Readiness.",
        "Tìm tập Pareto — các phương án không bị thống trị.",
        "Xác định điểm cân bằng giữa các mục tiêu xung đột.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)
    st.markdown("- GDP: GRDP của 6 vùng kinh tế")
    st.markdown("- Labor: Tỷ lệ lao động được đào tạo × Dân số")
    st.markdown("- AI Index: AI Readiness × Chỉ số số hóa")

    df_r = load_regions()
    if df_r is None:
        st.warning("Không tìm thấy dữ liệu vietnam_regions_2024.csv.")
        return

    # Cấu hình
    st.markdown(section_header("Cấu hình NSGA-II", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(4)
    pop_size = cfg_cols[0].slider("Quần thể (Population)", 50, 200, 100)
    n_gen = cfg_cols[1].slider("Số thế hệ (Generation)", 10, 100, 30)
    obj1_weight = cfg_cols[2].slider("Trọng số GDP", 0.0, 1.0, 0.40, 0.05)
    obj2_weight = cfg_cols[3].slider("Trọng số Lao động", 0.0, 1.0, 0.35, 0.05)
    obj3_weight = st.slider("Trọng số AI", 0.0, 1.0, 0.25, 0.05)

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # ---- 7.4.1 Pareto Front ----
    st.markdown('<div class="section-header-purple">7.4.1 Pareto Front</div>', unsafe_allow_html=True)

    import plotly.graph_objects as go
    population, objectives, pareto_mask = nsga2(pop_size, n_gen, df_r, n_obj=3)

    pareto_obj1 = [objectives[i][0] for i, m in enumerate(pareto_mask) if m]
    pareto_obj2 = [objectives[i][1] for i, m in enumerate(pareto_mask) if m]
    all_obj1 = [o[0] for o in objectives]
    all_obj2 = [o[1] for o in objectives]
    all_obj3 = [o[2] for o in objectives]

    combined_score = [obj1_weight * o[0] + obj2_weight * o[1] + obj3_weight * o[2] for o in objectives]
    best_idx = np.argmax(combined_score)
    best_ind = population[best_idx]
    best_obj = objectives[best_idx]

    n_pareto = sum(pareto_mask)

    fig_pf = go.Figure()
    fig_pf.add_trace(go.Scatter(
        x=all_obj1, y=all_obj2,
        mode="markers", marker=dict(color="#2A3A5A", size=7),
        name="Quần thể", hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>"
    ))
    fig_pf.add_trace(go.Scatter(
        x=pareto_obj1, y=pareto_obj2,
        mode="lines+markers",
        marker=dict(color=NEON_BLUE, size=10, symbol="star"),
        line=dict(color=NEON_BLUE, width=2),
        name="Pareto Front", hovertemplate="Pareto: (%{x:.3f}, %{y:.3f})<extra></extra>"
    ))
    fig_pf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=400,
        xaxis=dict(gridcolor="#1F2A5A", title="GDP Index (cao hơn = tốt hơn)"),
        yaxis=dict(gridcolor="#1F2A5A", title="Lao động Index (cao hơn = tốt hơn)"),
        title=dict(text="Pareto Front: GDP vs Lao động", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_pf, width='stretch')

    # 3D Pareto
    pareto_3d = [(objectives[i][0], objectives[i][1], objectives[i][2]) for i, m in enumerate(pareto_mask) if m]
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=all_obj1, y=all_obj2, z=all_obj3,
        mode="markers", marker=dict(color="#2A3A5A", size=5, opacity=0.6),
        name="Quần thể"
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[p[0] for p in pareto_3d],
        y=[p[1] for p in pareto_3d],
        z=[p[2] for p in pareto_3d],
        mode="lines+markers",
        marker=dict(color=NEON_BLUE, size=8, symbol="diamond"),
        line=dict(color=NEON_BLUE, width=2),
        name="Pareto Front"
    ))
    fig3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=450,
        scene=dict(
            xaxis=dict(gridcolor="#1F2A5A", title="GDP"),
            yaxis=dict(gridcolor="#1F2A5A", title="Lao động"),
            zaxis=dict(gridcolor="#1F2A5A", title="AI"),
        ),
        title=dict(text="Pareto Front 3D: GDP vs Lao động vs AI", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
    )
    st.plotly_chart(fig3d, width='stretch')

    # ---- 7.4.2 Nghiệm tối ưu ----
    st.markdown(section_header("7.4.2 Nghiệm tối ưu theo trọng số", "purple"), unsafe_allow_html=True)

    max_budget = 50.0
    best_alloc = best_ind
    regions = df_r["region_name_en"].tolist()
    regions_vn_map = {
        "Northern Midlands and Mountains": "Trung du miền núi",
        "Red River Delta": "Đồng Bằng Sông Hồng",
        "North Central and South Central Coast": "Bắc Trung Bộ",
        "Central Highlands": "Tây Nguyên",
        "Southeast": "Đông Nam Bộ",
        "Mekong Delta": "Đồng Bằng Sông Cửu Long",
    }
    regions_vn = [regions_vn_map.get(r, r) for r in regions]

    alloc_df = pd.DataFrame({
        "Vùng": regions_vn,
        "Phân bổ (Tỷ USD)": np.round(best_alloc, 3),
        "Tỷ lệ (%)": np.round(best_alloc / best_alloc.sum() * 100, 2),
    }).sort_values("Phân bổ (Tỷ USD)", ascending=False)
    st.dataframe(alloc_df.style.format({
        "Phân bổ (Tỷ USD)": "{:.3f}",
        "Tỷ lệ (%)": "{:.2f}",
    }), width='stretch')

    # KPIs
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_html(kpi_card(f"{n_pareto}", "Điểm Pareto", "Số điểm tối ưu", NEON_BLUE))
    with kpi_cols[1]:
        render_html(kpi_card(f"{best_obj[0]:.2f}", "GDP Index", "Đối tượng 1", NEON_GREEN))
    with kpi_cols[2]:
        render_html(kpi_card(f"{best_obj[1]:.2f}", "Lao động Index", "Đối tượng 2", NEON_PURPLE))
    with kpi_cols[3]:
        render_html(kpi_card(f"{best_obj[2]:.2f}", "AI Index", "Đối tượng 3", NEON_ORANGE))

    # ---- 7.4.3 So sánh mục tiêu ----
    st.markdown(section_header("7.4.3 So sánh mục tiêu", "green"), unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        "Mục tiêu": ["GDP", "Lao động", "AI"],
        "Trọng số": [f"{obj1_weight:.0%}", f"{obj2_weight:.0%}", f"{obj3_weight:.0%}"],
        "Giá trị nghiệm tối ưu": [f"{best_obj[0]:.3f}", f"{best_obj[1]:.3f}", f"{best_obj[2]:.3f}"],
        "Mô tả": [
            "GRDP × hệ số phân bổ",
            "Lao động đào tạo × phân bổ",
            "AI × Số hóa × phân bổ",
        ]
    })
    st.dataframe(comp_df, width='stretch')

    # ===== BIỂU ĐỒ =====
    st.markdown(section_header("BIỂU ĐỒ", "blue"), unsafe_allow_html=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=regions_vn, y=list(best_alloc),
        marker_color=NEON_BLUE, opacity=0.85,
        hovertemplate="Vùng: %{x}<br>Phân bổ: %{y:.2f}B<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=400,
        xaxis=dict(gridcolor="#1F2A5A", tickangle=25, title=""),
        yaxis=dict(gridcolor="#1F2A5A", title="Phân bổ (Tỷ USD)"),
        title=dict(text="Phân bổ nguồn lực tối ưu theo vùng", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_bar, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    regions = df_r["region_name_en"].tolist()
    regions_vn_map = {
        "Northern Midlands and Mountains": "Trung du miền núi",
        "Red River Delta": "Đồng Bằng Sông Hồng",
        "North Central and South Central Coast": "Bắc Trung Bộ",
        "Central Highlands": "Tây Nguyên",
        "Southeast": "Đông Nam Bộ",
        "Mekong Delta": "Đồng Bằng Sông Cửu Long",
    }
    regions_vn = [regions_vn_map.get(r, r) for r in regions]
    grdp_vals = df_r["grdp_trillion_VND"].values
    alloc_sorted = sorted(zip(best_ind, regions_vn, grdp_vals), reverse=True)
    top_alloc_r = alloc_sorted[0]
    bot_alloc_r = alloc_sorted[-1]
    max_alloc = best_ind.max()
    min_alloc_r = best_ind.min()
    gdp_weight = obj1_weight
    labor_weight = obj2_weight
    ai_weight = obj3_weight

    with st.expander("Pareto Front cho thấy điều gì về sự đánh đổi giữa các mục tiêu?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        NSGA-II tìm được <b style="color:#FFD700;">{n_pareto} điểm Pareto</b> — mỗi điểm là một
        phương án phân bổ nguồn lực không bị thống trị bởi phương án nào khác.
        Pareto Front phẳng/cong phản ánh mức độ <b style="color:#FF6B35;">đánh đổi (trade-off)</b>
        giữa 3 mục tiêu: GDP, Lao động, và AI. Nếu đường Pareto dốc,
        có nghĩa tăng một mục tiêu phải hy sinh nhiều mục tiêu khác.
        Điểm tối ưu theo trọng số (GDP={gdp_weight:.0%}, Lao động={labor_weight:.0%}, AI={ai_weight:.0%})
        nằm trên Pareto Front với giá trị: GDP Index = <b style="color:#00D4FF;">{best_obj[0]:.3f}</b>,
        Lao động = <b style="color:#00FF88;">{best_obj[1]:.3f}</b>, AI = <b style="color:#BF00FF;">{best_obj[2]:.3f}</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Sự đánh đổi xuất phát từ <b style="color:#FF6B35;">phân bổ nguồn lực cố định</b>:
        nguồn lực đầu tư vào vùng GDP cao (Đông Nam Bộ) sẽ ít hơn cho vùng lao động đông
        (Đồng Bằng Sông Cửu Long). Tương tự, đầu tư vào vùng có AI Readiness cao
        (Đồng Bằng Sông Hồng) sẽ ít hơn cho vùng có tiềm năng lao động lớn nhưng
        AI thấp. Không có phương án nào vừa tối đa GDP, vừa tối đa Lao động,
        vừa tối đa AI — đây là bản chất của bài toán đa mục tiêu.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Pareto Front cung cấp <b style="color:#FFD700;">bản đồ quyết định</b> cho nhà hoạch định chính sách:
        mỗi điểm trên đường Pareto đại diện cho một sự đánh đổi hợp lý giữa các mục tiêu.
        Ví dụ: điểm ưu tiên GDP cao sẽ tập trung nguồn lực vào Đông Nam Bộ;
        điểm ưu tiên Lao động sẽ phân bổ nhiều hơn cho vùng có dân số đông;
        điểm ưu tiên AI sẽ tập trung vào vùng có AI Readiness tốt. Mỗi phương án
        đều tối ưu theo góc nhìn riêng, không phương án nào hoàn toàn "sai".
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Công khai Pareto Front</b> — Trình bày đường Pareto cho
        lãnh đạo để lựa chọn mức đánh đổi phù hợp với ưu tiên chính trị. (2) <b style="color:#00D4FF;">Điều chỉnh trọng số linh hoạt</b> —
        Thay đổi trọng số theo giai đoạn: ưu tiên GDP trước, sau đó chuyển sang lao động và AI.
        (3) <b style="color:#00D4FF;">Kết hợp nhiều điểm Pareto</b> — Sử dụng kết hợp 2-3 phương án
        Pareto để đa dạng hóa phân bổ. (4) <b style="color:#00D4FF;">Ràng buộc bổ sung</b> —
        Thêm ràng buộc đảm bảo mỗi vùng nhận tối thiểu 1B USD để tránh bỏ rơi vùng khó khăn.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Điểm cân bằng tối ưu theo trọng số nằm ở vùng nào?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Với trọng số GDP={gdp_weight:.0%}, Lao động={labor_weight:.0%}, AI={ai_weight:.0%},
        nghiệm tối ưu phân bổ nhiều nhất cho: <b style="color:#00FF88;">{top_alloc_r[1]}</b>
        với <b style="color:#FFD700;">{top_alloc_r[0]:.3f}B USD</b>
        ({top_alloc_r[0]/max_alloc*100:.1f}% so với vùng nhận nhiều nhất),
        GRDP = {top_alloc_r[2]:.0f}T VND. Vùng nhận ít nhất: <b style="color:#FF6B35;">{bot_alloc_r[1]}</b>
        với {bot_alloc_r[0]:.3f}B USD ({bot_alloc_r[0]/max_alloc*100:.1f}%).
        Tất cả 6 vùng đều nhận được phân bổ >0, đảm bảo không vùng nào bị bỏ rơi.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        {top_alloc_r[1]} được ưu tiên vì: (1) <b style="color:#00D4FF;">GRDP cao ({top_alloc_r[2]:.0f}T VND)</b> —
        đóng góp lớn vào GDP mục tiêu. (2) <b style="color:#00FF88;">Vùng có điểm tổng hợp cao
        trên cả 3 mục tiêu</b> — vừa có GRDP lớn, vừa có dân số đáng kể, vừa có AI Readiness tốt.
        NSGA-II đã chọn phương án phân bổ tối đa hóa tổng hàm mục tiêu có trọng số.
        Điểm khác biệt với LP (Bài 4) là NSGA-II tìm <b style="color:#FFD700;">nhiều phương án</b>
        thay vì một nghiệm duy nhất.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Phân bổ theo trọng số hiện tại phản ánh chiến lược <b style="color:#FFD700;">tập trung nguồn lực</b>:
        vùng dẫn đầu nhận nhiều hơn để tối đa hóa tổng GDP. Tuy nhiên, điều này có thể
        làm trầm trọng thêm bất bình đẳng vùng nếu kéo dài. Mặt khác, {bot_alloc_r[1]}
        dù nhận ít nhất nhưng vẫn có phân bổ dương, cho thấy NSGA-II đã cân bằng
        giữa <b style="color:#00FF88;">hiệu quả</b> và <b style="color:#00FF88;">công bằng</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Tập trung đầu tư vào {top_alloc_r[1]}</b> —
        Biến vùng dẫn đầu thành động lực tăng trưởng chính. (2) <b style="color:#00D4FF;">Đảm bảo ngưỡng tối thiểu</b> —
        Mỗi vùng nhận tối thiểu 1B USD/năm để duy trì phát triển cơ bản.
        (3) <b style="color:#00D4FF;">Chiến lược lai</b> — Kết hợp phân bổ NSGA-II với đầu tư công
        cho vùng khó khăn. (4) <b style="color:#00D4FF;">Giám sát chỉ số đa mục tiêu</b> —
        Theo dõi đồng thời GDP, việc làm và AI của từng vùng hàng năm.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Chính phủ nên áp dụng chiến lược phân bổ nào cho giai đoạn tới?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        NSGA-II cung cấp <b style="color:#FFD700;">{n_pareto} phương án Pareto</b>, mỗi phương án
        đại diện cho một cách đánh đổi khác nhau giữa GDP, Lao động và AI.
        Chiến lược tốt nhất phụ thuộc vào ưu tiên chính sách cụ thể: (1) Nếu ưu tiên
        <b style="color:#00D4FF;">tăng trưởng</b> → chọn điểm Pareto nghiêng về GDP.
        (2) Nếu ưu tiên <b style="color:#00FF88;">việc làm</b> → chọn điểm Pareto nghiêng về Lao động.
        (3) Nếu ưu tiên <b style="color:#BF00FF;">chuyển đổi số</b> → chọn điểm Pareto nghiêng về AI.
        Phương án hiện tại (trọng số {gdp_weight:.0%}/{labor_weight:.0%}/{ai_weight:.0%})
        là một sự cân bằng hợp lý giữa cả 3 mục tiêu.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Việt Nam đang ở giai đoạn phát triển cần cân bằng giữa <b style="color:#00D4FF;">tăng trưởng</b>
        (để thoát bẫy thu nhập trung bình) và <b style="color:#00FF88;">bền vững</b>
        (để tránh bất bình đẳng quá mức). Trọng số {gdp_weight:.0%}/{labor_weight:.0%}/{ai_weight:.0%}
        phản ánh ưu tiên: tăng trưởng GDP là quan trọng nhất ({gdp_weight:.0%}),
        lao động đứng thứ hai ({labor_weight:.0%}), AI đứng thứ ba ({ai_weight:.0%}).
        Đây là ưu tiên hợp lý cho giai đoạn 2026-2030 khi Việt Nam cần duy trì
        đà tăng trưởng mạnh để đạt mục tiêu kinh tế số.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Chiến lược lai (hybrid strategy) là tối ưu: <b style="color:#00D4FF;">{top_alloc_r[1]}</b>
        tiếp tục nhận đầu tư mạnh để duy trì đà tăng trưởng; đồng thời đảm bảo
        nguồn lực tối thiểu cho <b style="color:#FF6B35;">{bot_alloc_r[1]}</b> và các vùng khác
        để tránh phân hóa quá mức. Cần kết hợp phân bổ NSGA-II với
        <b style="color:#FFD700;">chính sách bổ trợ</b>: đầu tư công, ODA, và hỗ trợ kỹ thuật
        cho vùng khó khăn nhằm đảm bảo phát triển toàn diện.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Áp dụng phương án Pareto hiện tại</b> — Dùng kết quả NSGA-II
        với trọng số {gdp_weight:.0%}/{labor_weight:.0%}/{ai_weight:.0%} làm căn cứ phân bổ ngân sách
        vùng hàng năm. (2) <b style="color:#00D4FF;">Điều chỉnh trọng số theo giai đoạn</b> —
        Giai đoạn 2026-2028: ưu tiên GDP nhiều hơn. Giai đoạn 2029-2030: tăng trọng số
        Lao động và AI. (3) <b style="color:#00D4FF;">Kết hợp NSGA-II với LP</b> — Dùng Pareto Front
        để xác định khoảng phân bổ hợp lý, sau đó dùng LP (Bài 4) để tìm phân bổ tối ưu
        cụ thể. (4) <b style="color:#00D4FF;">Chính sách bù đắp</b> — Đầu tư công bổ sung
        cho vùng nhận ít từ NSGA-II ({bot_alloc_r[1]} và các vùng khó khăn).
        </div>
        """, unsafe_allow_html=True)
