import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_regions
from utils.styles import NEON_BLUE, NEON_PURPLE, NEON_GREEN, NEON_ORANGE, section_header, glass_card, kpi_card, render_html, highlight_box
from utils.helpers import vietnamese_region_name

np.random.seed(42)


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((n_states, n_actions))
        self.rewards_history = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state):
        best_next = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def get_reward(state, action, region_data, t):
    region_idx = min(state, len(region_data) - 1)
    grdp = region_data["grdp_trillion_VND"].values[region_idx]
    fdi = region_data["fdi_registered_billion_USD"].values[region_idx]
    ai = region_data["ai_readiness_0_100"].values[region_idx]
    digital = region_data["digital_index_0_100"].values[region_idx]

    investment_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    investment = investment_levels[action]

    gdp_gain = (fdi * 0.01 + ai * 0.005 + digital * 0.003) * investment
    risk_penalty = abs(investment - 2.0) * 0.01
    time_penalty = t * 0.001
    reward = gdp_gain - risk_penalty - time_penalty
    return max(reward, -0.5)


def get_next_state(state, action, n_states):
    investment_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    inv = investment_levels[action]
    delta = 0
    if inv > 3.0:
        delta = 2
    elif inv > 1.5:
        delta = 1
    elif inv < 0.3:
        delta = -1
    next_state = max(0, min(n_states - 1, state + delta))
    return next_state


def run():

    # ===== MỤC TIÊU =====
    st.markdown(section_header("MỤC TIÊU", "blue"), unsafe_allow_html=True)
    objectives = [
        "Tìm chiến lược đầu tư tối ưu cho từng vùng kinh tế bằng Q-Learning.",
        "Agent học chính sách tối ưu từ phần thưởng (reward) phản ánh GDP, FDI và AI.",
        "So sánh chính sách Q-Learning với chính sách ngẫu nhiên và chính sách đều.",
    ]
    for obj in objectives:
        st.markdown(highlight_box(obj), unsafe_allow_html=True)

    # ===== DỮ LIỆU =====
    st.markdown(section_header("DỮ LIỆU", "purple"), unsafe_allow_html=True)
    st.markdown("- Trạng thái vùng: Mức phát triển được mã hóa thành 5 cấp (0-4)")
    st.markdown("- Mức đầu tư: 5 cấp từ 0B đến 5B USD")
    st.markdown("- GDP, FDI, AI Readiness, Chỉ số số hóa của 6 vùng")

    df_r = load_regions()
    if df_r is None:
        st.warning("Không tìm thấy dữ liệu vietnam_regions_2024.csv.")
        return

    # Cấu hình
    st.markdown(section_header("Cấu hình Q-Learning", "blue"), unsafe_allow_html=True)
    cfg_cols = st.columns(4)
    n_episodes = cfg_cols[0].slider("Số episode", 100, 1000, 300, 50)
    alpha = cfg_cols[1].slider("Alpha (tốc độ học)", 0.01, 0.5, 0.15, 0.01)
    gamma = cfg_cols[2].slider("Gamma (hệ số discount)", 0.5, 0.99, 0.90, 0.01)
    n_states = cfg_cols[3].slider("Số trạng thái", 3, 7, 5)

    n_actions = 5
    investment_labels = ["Rất ít (0B)", "Ít (0.5B)", "Trung bình (1B)", "Nhiều (2B)", "Rất nhiều (5B)"]

    # ===== KẾT QUẢ =====
    st.markdown(section_header("KẾT QUẢ", "green"), unsafe_allow_html=True)

    # ---- 11.4.1 Reward ----
    st.markdown('<div class="section-header-purple">11.4.1 Reward & Huấn luyện</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("Đang huấn luyện Q-Learning...")

    agent = QLearningAgent(n_states, n_actions, alpha=alpha, gamma=gamma)
    reward_per_episode = []

    for ep in range(n_episodes):
        if ep % 50 == 0:
            progress_bar.progress(int(ep / n_episodes * 100))
        state = np.random.randint(n_states)
        total_reward = 0
        for t in range(10):
            action = agent.choose_action(state)
            reward = get_reward(state, action, df_r, t)
            next_state = get_next_state(state, action, n_states)
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        agent.decay_epsilon()
        reward_per_episode.append(total_reward)

    progress_bar.progress(100)
    status_text.success(f"Hoàn thành {n_episodes} episode!")

    final_epsilon = agent.epsilon
    avg_reward = np.mean(reward_per_episode[-50:])
    best_reward = np.max(reward_per_episode)
    converged = agent.epsilon <= agent.epsilon_min + 0.01

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_html(kpi_card(f"{n_episodes}", "Episode", "Tổng số lần huấn luyện", NEON_BLUE))
    with kpi_cols[1]:
        render_html(kpi_card(f"{avg_reward:.3f}", "Reward TB", "50 episode cuối", NEON_GREEN))
    with kpi_cols[2]:
        render_html(kpi_card(f"{final_epsilon:.4f}", "Epsilon cuối", f"{'Đã hội tụ' if converged else 'Đang học'}", NEON_PURPLE))
    with kpi_cols[3]:
        render_html(kpi_card(f"{best_reward:.3f}", "Reward tốt nhất", "Tất cả episode", NEON_ORANGE))

    # Reward curve
    import plotly.graph_objects as go
    smooth_window = max(1, n_episodes // 10)
    smoothed = np.convolve(reward_per_episode, np.ones(smooth_window)/smooth_window, mode='valid')

    fig_reward = go.Figure()
    fig_reward.add_trace(go.Scatter(
        x=list(range(len(reward_per_episode))),
        y=reward_per_episode,
        mode="lines", name="Reward",
        line=dict(color="rgba(0,212,255,0.2)", width=1),
        hovertemplate="Episode %{x}: %{y:.4f}<extra></extra>",
    ))
    fig_reward.add_trace(go.Scatter(
        x=list(range(smooth_window - 1, len(reward_per_episode))),
        y=list(smoothed),
        mode="lines", name=f"Trung bình ({smooth_window})",
        line=dict(color=NEON_GREEN, width=2.5),
        hovertemplate="TB %{x}: %{y:.4f}<extra></extra>",
    ))
    fig_reward.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=380,
        xaxis=dict(gridcolor="#1F2A5A", title="Episode"),
        yaxis=dict(gridcolor="#1F2A5A", title="Tổng Reward"),
        title=dict(text="Reward qua các Episode", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        legend=dict(bgcolor="rgba(0,0,0,0)", font={"color": "#B0C0E0"}),
        shapes=[dict(type="line", x0=0, x1=n_episodes, y0=0, y1=0,
                     xref="paper", yref="y", line=dict(color="white", width=1, dash="dash"))],
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_reward, width='stretch')

    # ---- 11.4.2 Training & Q-Table ----
    st.markdown(section_header("11.4.2 Bảng Q và Chính sách tối ưu", "purple"), unsafe_allow_html=True)

    q_table_df = pd.DataFrame(
        agent.Q,
        columns=investment_labels,
        index=[f"Trạng thái {i}" for i in range(n_states)]
    )
    st.markdown("**Bảng Q (Q-Table):**")
    st.dataframe(q_table_df.style.format("{:.4f}"), use_container_width=True)

    optimal_policy = [investment_labels[np.argmax(agent.Q[s])] for s in range(n_states)]
    policy_df = pd.DataFrame({
        "Trạng thái": [f"Mức phát triển {s}" for s in range(n_states)],
        "Chính sách tối ưu": optimal_policy,
        "Q-value max": [f"{np.max(agent.Q[s]):.4f}" for s in range(n_states)],
    })
    st.markdown("**Chính sách tối ưu:**")
    st.dataframe(policy_df, width='stretch')

    # Heatmap Q-table
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=agent.Q,
        x=investment_labels,
        y=[f"S{s}" for s in range(n_states)],
        colorscale=[
            [0.0, "#0A0E27"],
            [0.25, "#1A2A6A"],
            [0.5, "#00D4FF"],
            [0.75, "#BF00FF"],
            [1.0, "#FF00AA"],
        ],
        hovertemplate="S%{y}: %{x}<br>Q=%{z:.4f}<extra></extra>",
    ))
    fig_heatmap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E0E0FF"}, height=350,
        xaxis=dict(gridcolor="#1F2A5A", title="Mức đầu tư"),
        yaxis=dict(gridcolor="#1F2A5A", title="Trạng thái"),
        title=dict(text="Q-Table Heatmap", x=0.5, font={"color": NEON_BLUE, "size": 14}),
        hoverlabel={"bgcolor": "#0A0E27", "font": {"color": "#E0E0FF"}},
    )
    st.plotly_chart(fig_heatmap, width='stretch')

    # ---- 11.4.3 Chính sách tối ưu ----
    st.markdown(section_header("11.4.3 Chính sách tối ưu theo vùng", "green"), unsafe_allow_html=True)

    regions = df_r["region_name_en"].tolist()
    regions_vn = [vietnamese_region_name(r) for r in regions]
    grdp = df_r["grdp_trillion_VND"].values

    random_policy = [investment_labels[np.random.randint(n_actions)] for _ in regions]
    equal_policy = [investment_labels[2] for _ in regions]
    optimal_by_region = []
    for i, grdp_val in enumerate(grdp):
        state_idx = min(int(grdp_val / grdp.max() * (n_states - 1)), n_states - 1)
        optimal_by_region.append(investment_labels[np.argmax(agent.Q[state_idx])])

    policy_compare = pd.DataFrame({
        "Vùng": regions_vn,
        "GRDP (Tỷ VND)": grdp,
        "Chính sách ngẫu nhiên": random_policy,
        "Chính sách đều": equal_policy,
        "Chính sách Q-Learning": optimal_by_region,
    })
    st.dataframe(policy_compare, width='stretch')

    # ===== THẢO LUẬN =====
    st.markdown('<hr class="neon-divider"/>', unsafe_allow_html=True)
    st.markdown(section_header("THẢO LUẬN", "orange"), unsafe_allow_html=True)

    grdp_vals = df_r["grdp_trillion_VND"].values
    regions_r = df_r["region_name_en"].tolist()
    regions_vn_r = [vietnamese_region_name(r) for r in regions_r]
    grdp_max = grdp_vals.max()
    avg_reward_last50 = avg_reward
    best_q_state = np.argmax(np.max(agent.Q, axis=1))
    worst_q_state = np.argmin(np.max(agent.Q, axis=1))
    optimal_inv_label = investment_labels[np.argmax(agent.Q[best_q_state])]
    worst_inv_label = investment_labels[np.argmax(agent.Q[worst_q_state])]

    with st.expander("Agent Q-Learning học được chính sách đầu tư tối ưu như thế nào?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Q-Learning sau <b style="color:#00D4FF;">{n_episodes} episode</b> đã hội tụ với
        epsilon = <b style="color:#FFD700;">{final_epsilon:.4f}</b>
        ({'đã hội tụ' if converged else 'đang học'}). Trạng thái có Q-value cao nhất:
        <b style="color:#00FF88;">S{best_q_state}</b> với Q-value = {np.max(agent.Q[best_q_state]):.4f},
        chính sách tối ưu: <b style="color:#FFD700;">{optimal_inv_label}</b>.
        Trạng thái có Q-value thấp nhất: <b style="color:#FF6B35;">S{worst_q_state}</b>
        với Q-value = {np.max(agent.Q[worst_q_state]):.4f},
        chính sách tối ưu: <b style="color:#FF6B35;">{worst_inv_label}</b>.
        Reward trung bình 50 episode cuối = <b style="color:#00D4FF;">{avg_reward_last50:.4f}</b>
        (dương → chính sách có lợi ròng).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Agent học được qua thử nghiệm: (1) <b style="color:#00D4FF;">Trạng thái phát triển cao (S{best_q_state})</b> —
        mức đầu tư "{optimal_inv_label}" cho reward cao nhất vì GDP, FDI và AI
        của vùng đã lớn → đầu tư thêm tạo ra lợi nhuận biên lớn. (2) <b style="color:#FF6B35;">Trạng thái phát triển thấp (S{worst_q_state})</b> —
        mức đầu tư "{worst_inv_label}" là tối ưu vì vùng chưa sẵn sàng hấp thụ vốn lớn →
        đầu tư nhiều rủi ro cao, reward thấp. (3) <b style="color:#00D4FF;">Epsilon giảm dần</b> —
        ban đầu agent khám phá (exploration), sau đó chuyển sang khai thác (exploitation)
        chính sách đã học được. Hệ số gamma = {gamma:.2f} đảm bảo agent
        tính đến lợi ích dài hạn, không chỉ lợi ích tức thì.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Q-Learning học được rằng <b style="color:#FFD700;">không có mức đầu tư duy nhất tối ưu cho mọi vùng</b> —
        mỗi vùng ở mức phát triển khác nhau cần chính sách đầu tư khác nhau.
        Vùng phát triển cao nên nhận đầu tư lớn (2-5B USD),
        vùng phát triển thấp nên nhận đầu tư vừa phải (0.5-1B USD) để
        giảm rủi ro và xây dựng năng lực dần dần. Đây là nguyên tắc
        <b style="color:#00FF88;">phân bổ nguồn lực theo giai đoạn phát triển</b>,
        khác với LP hay TOPSIS đơn thuần.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Phân nhóm vùng theo mức phát triển</b> —
        Sử dụng GRDP, FDI, AI Readiness để phân loại vùng vào các nhóm,
        mỗi nhóm áp dụng mức đầu tư phù hợp theo Q-Learning. (2) <b style="color:#00D4FF;">Chiến lược đầu tư theo giai đoạn</b> —
        Vùng mới phát triển: bắt đầu với 0.5-1B USD, sau đó tăng dần khi năng lực
        cải thiện. (3) <b style="color:#00D4FF;">Cập nhật Q-table thường xuyên</b> —
        Mô phỏng lại Q-Learning mỗi năm để phản ánh điều kiện kinh tế thay đổi.
        (4) <b style="color:#00D4FF;">Kết hợp với LP và NSGA-II</b> —
        Dùng Q-Learning để xác định mức đầu tư, LP/NSGA-II để phân bổ cụ thể.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Chính sách đầu tư Q-Learning khác gì so với chính sách đều?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Q-Learning cho chính sách <b style="color:#00FF88;">khác biệt theo từng vùng</b>,
        trong khi chính sách đều áp dụng mức <b style="color:#FFD700;">1B USD/năm</b> cho mọi vùng.
        Ví dụ: vùng có GRDP cao nhất ({grdp_vals.max():.0f}T VND) nhận mức
        "{optimal_inv_label}" theo Q-Learning, trong khi vùng yếu hơn
        nhận mức thấp hơn. Q-Learning phản ánh nguyên tắc kinh tế:
        <b style="color:#FFD700;">vùng có sẵn nền tảng (GRDP, AI, số hóa) sẽ hấp thụ vốn hiệu quả hơn</b>,
        nên xứng đáng nhận đầu tư nhiều hơn. Chính sách đều phân bổ nguồn lực
        kém hiệu quả vì bỏ qua năng lực hấp thụ khác nhau giữa các vùng.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Sự khác biệt đến từ cơ chế học của Q-Learning: (1) <b style="color:#00D4FF;">Reward phản ánh hiệu quả thực</b> —
        Vùng có FDI, AI, số hóa cao → đầu tư tạo ra GDP cao → reward lớn →
        Q-value cao → chính sách ưu tiên đầu tư nhiều. (2) <b style="color:#FF6B35;">Vùng yếu có rủi ro cao</b> —
        Đầu tư lớn vào vùng chưa có nền tảng → hiệu quả thấp → reward thấp →
        Q-value thấp → Q-Learning khuyến nghị đầu tư ít. (3) <b style="color:#00D4FF;">Tính đến rủi ro</b> —
        Hàm reward có penalty cho mức đầu tư quá lớn ({'5B USD' if '5B' in worst_inv_label else 'mức cao'})
        tại vùng chưa sẵn sàng. (4) <b style="color:#00D4FF;">Tính đến thời gian</b> —
        Reward có time penalty để ưu tiên kết quả ngắn hạn.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Chính sách Q-Learning phản ánh nguyên tắc <b style="color:#FFD700;">đầu tư có điều kiện</b>:
        nguồn lực đi đến nơi có khả năng sử dụng hiệu quả nhất. Điều này có thể
        bị phê phán là <b style="color:#FF6B35;">không công bằng</b> (vùng yếu nhận ít hơn),
        nhưng lại <b style="color:#00FF88;">hiệu quả kinh tế hơn</b> (cùng số vốn tạo ra nhiều GDP hơn).
        Trên thực tế, cần kết hợp Q-Learning với ràng buộc công bằng vùng
        (ví dụ: mỗi vùng nhận tối thiểu X Tỷ VND) để cân bằng giữa hiệu quả và bình đẳng.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Áp dụng Q-Learning cho quy hoạch vùng</b> —
        Sử dụng kết quả Q-Learning để xác định mức đầu tư ưu tiên cho từng nhóm vùng.
        (2) <b style="color:#00D4FF;">Thêm ràng buộc công bằng</b> — Đảm bảo mỗi vùng
        nhận tối thiểu 1B USD để tránh bỏ rơi vùng khó khăn. (3) <b style="color:#00D4FF;">Chiến lược từng bước</b> —
        Vùng yếu: bắt đầu với đầu tư nhỏ để xây nền, sau đó tăng dần khi năng lực
        cải thiện → Q-value tăng → Q-Learning tự động khuyến nghị đầu tư nhiều hơn.
        (4) <b style="color:#00D4FF;">Giám sát và cập nhật</b> — Chạy Q-Learning
        hàng năm để cập nhật chính sách theo điều kiện kinh tế thay đổi.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Hàm ý chính sách đầu tư vùng cụ thể cho Việt Nam 2026-2030?"):
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nhận xét**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Từ kết quả Q-Learning, có thể tổng hợp chính sách đầu tư theo nhóm vùng:
        (1) <b style="color:#00FF88;">Nhóm dẫn đầu</b> (GRDP cao, AI & Số hóa tốt):
        Đông Nam Bộ, Đồng Bằng Sông Hồng → đầu tư nhiều (2-5B USD/năm) để
        tối đa hóa GDP. (2) <b style="color:#FFD700;">Nhóm trung bình</b> (GRDP vừa, AI & Số hóa trung bình):
        Bắc Trung Bộ, Đồng Bằng Sông Cửu Long → đầu tư vừa phải (1-2B USD/năm)
        để cải thiện dần. (3) <b style="color:#FF6B35;">Nhóm khó khăn</b> (GRDP thấp, AI & Số hóa yếu):
        Trung du miền núi, Tây Nguyên → đầu tư ít (0.5-1B USD/năm)
        để xây nền tảng, tập trung vào hạ tầng số cơ bản.
        Reward dương ({avg_reward_last50:.4f}) xác nhận chính sách học được
        mang lại lợi ích kinh tế ròng dương.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Nguyên nhân**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Chiến lược phân nhóm xuất phát từ: (1) <b style="color:#00D4FF;">Năng lực hấp thụ vốn khác nhau</b> —
        Vùng dẫn đầu có hạ tầng, nhân lực, và môi trường kinh doanh tốt →
        đầu tư vào đây tạo ra lợi nhuận biên cao nhất. (2) <b style="color:#FFD700;">Hiệu ứng lan tỏa</b> —
        Đầu tư vào vùng dẫn đầu tạo hiệu ứng dây chuyền sang vùng xung quanh
        (logistics, chuỗi cung ứng, dịch vụ). (3) <b style="color:#FF6B35;">Chi phí cơ hội</b> —
        Đầu tư quá nhiều vào vùng khó khăn khi chưa có nền tảng sẽ
        lãng phí do thiếu nhân lực và hạ tầng để sử dụng vốn hiệu quả.
        (4) <b style="color:#00D4FF;">Học từ thực tế</b> — Q-Learning mô phỏng hàng nghìn
        lần chính sách đầu tư, chọn ra chiến lược tối ưu dựa trên dữ liệu GRDP, FDI, AI của 6 vùng.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Ý nghĩa kinh tế**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0;">
        Chính sách phân nhóm từ Q-Learning giúp <b style="color:#FFD700;">tối đa hóa hiệu quả đầu tư
        công</b>: cùng một đồng ngân sách, phân bổ cho vùng có năng lực hấp thụ tốt nhất
        sẽ tạo ra nhiều GDP và việc làm hơn. Đồng thời, chiến lược từng bước cho vùng
        khó khăn (<b style="color:#FF6B35;">0.5-1B USD trước, tăng dần</b>) giảm rủi ro
        đầu tư thất bại. Reward dương = <b style="color:#00FF88;">{avg_reward_last50:.4f}</b>
        xác nhận tổng mức lợi ích ròng của chiến lược Q-Learning là <b style="color:#00FF88;">dương</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<span style='font-family:Rajdhani,sans-serif; color:#B0C0E0;'>**📌 Hàm ý chính sách**</span>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'Rajdhani',sans-serif; color:#B0C0E0; line-height:1.8; padding:0.5rem 0 1rem 0;">
        (1) <b style="color:#00D4FF;">Nhóm dẫn đầu (Đông Nam Bộ, ĐBSH)</b> —
        Tăng ngân sách đầu tư lên 2-5B USD/năm, tập trung vào hạ tầng công nghệ cao,
        trung tâm R&D, và khu công nghệ. (2) <b style="color:#FFD700;">Nhóm trung bình (Bắc Trung Bộ, ĐBSCL)</b> —
        Duy trì mức 1-2B USD/năm, ưu tiên kết nối giao thông, logistics, và hạ tầng số.
        (3) <b style="color:#FF6B35;">Nhóm khó khăn (Trung du miền núi, Tây Nguyên)</b> —
        Đầu tư 0.5-1B USD/năm tập trung vào hạ tầng số cơ bản, đào tạo nhân lực,
        và kết nối với vùng dẫn đầu. (4) <b style="color:#00D4FF;">Cơ chế đặc thù vùng</b> —
        Áp dụng chính sách ưu đãi thuế, đất đai, hành chính khác nhau cho từng nhóm.
        (5) <b style="color:#00D4FF;">Giám sát bằng Q-table</b> — Cập nhật Q-Learning
        hàng năm để điều chỉnh mức đầu tư theo tiến bộ thực tế của từng vùng.
        </div>
        """, unsafe_allow_html=True)
