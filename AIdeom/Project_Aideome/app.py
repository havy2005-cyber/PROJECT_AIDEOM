import streamlit as st
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.styles import apply_page_config, apply_global_style
from utils.data_loader import get_dataset_info

apply_page_config()
apply_global_style()

# ============================================================
# PAGE REGISTRY
# ============================================================
PAGES = {
    "Trang chủ":         "modules.home",
    "Bài 1 Cobb-Douglas": "modules.bai1_cobb_douglas",
    "Bài 2 LP Ngân sách":  "modules.bai2_lp_budget",
    "Bài 3 Xếp hạng ngành": "modules.bai3_priority_ranking",
    "Bài 4 LP Vùng kinh tế": "modules.bai4_lp_region",
    "Bài 5 MIP Dự án":    "modules.bai5_mip_projects",
    "Bài 6 TOPSIS Vùng":  "modules.bai6_topsis_region",
    "Bài 7 NSGA-II":      "modules.bai7_nsga2",
    "Bài 8 Mô hình động": "modules.bai8_dynamic_model",
    "Bài 9 AI và Lao động": "modules.bai9_ai_labor",
    "Bài 10 Stochastic":   "modules.bai10_stochastic",
    "Bài 11 Q-Learning":   "modules.bai11_qlearning",
    "Bài 12 Dashboard tổng hợp": "modules.bai12_dashboard",
}

SECTIONS = [
    ("🏠", "Trang chủ",            ["Trang chủ"]),
    ("📊", "Phân tích kinh tế",      ["Bài 1 Cobb-Douglas", "Bài 2 LP Ngân sách", "Bài 3 Xếp hạng ngành"]),
    ("⚙️", "Tối ưu hóa",           ["Bài 4 LP Vùng kinh tế", "Bài 5 MIP Dự án"]),
    ("🤖", "AI & Mô phỏng",        ["Bài 6 TOPSIS Vùng", "Bài 7 NSGA-II", "Bài 9 AI và Lao động", "Bài 11 Q-Learning"]),
    ("📈", "Dự báo",               ["Bài 8 Mô hình động", "Bài 10 Stochastic"]),
    ("🚀", "Dashboard tổng hợp",    ["Bài 12 Dashboard tổng hợp"]),
]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="text-align:center; padding:0.3rem 0 0.8rem;">
    <div style="font-size:1.3rem; margin-bottom:0.2rem;">🇻🇳 <span style="font-family:'Orbitron',monospace; font-weight:900;
        background:linear-gradient(135deg,#00D4FF,#BF00FF,#00FF88);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
        AIDEOM-VN
    </span></div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:0.68rem; color:#8090C0; line-height:1.6;">
        Hệ thống hỗ trợ phân tích,<br>
        mô phỏng và tối ưu hóa quyết định<br>
        phát triển kinh tế Việt Nam
    </div>
    <div style="width:100%; height:1px; background:linear-gradient(90deg,#00D4FF,#BF00FF); margin:0.6rem 0; border-radius:2px;"></div>
</div>
""", unsafe_allow_html=True)

# CSS
st.sidebar.markdown("""
<style>
.sidebar-nav { font-family:'Rajdhani',sans-serif; }
.nav-section {
    font-family:'Orbitron',monospace; font-size:0.58rem; color:#4A5A8A;
    text-transform:uppercase; letter-spacing:1.5px; padding:0.4rem 0.6rem 0.2rem; margin-top:0.3rem;
}
.nav-divider { border-top:1px solid rgba(0,212,255,0.08); margin:0.4rem 0; }
</style>
""", unsafe_allow_html=True)

# Session state for page persistence
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Trang chủ"

# Navigation — ONE element per nav item, no duplicates
for icon, section_name, items in SECTIONS:
    st.sidebar.markdown(f'<div class="nav-section">{icon} {section_name}</div>', unsafe_allow_html=True)
    for item in items:
        is_active = item == st.session_state.selected_page
        btn_style = "border-color:#00D4FF;color:#00D4FF;font-weight:600;" if is_active else ""
        if st.sidebar.button(item, key=f"nav_{section_name}_{item}", width='stretch'):
            st.session_state.selected_page = item
            st.rerun()
        if is_active:
            st.sidebar.markdown(
                f'<style>button[data-testid="stBaseButton-secondary"][key="nav_{section_name}_{item}"]'
                f'{{border-color:#00D4FF;color:#00D4FF;font-weight:600;}}</style>',
                unsafe_allow_html=True
            )

# Student info — placed with the navigation menu
st.sidebar.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="nav-section">🎓 Sinh viên thực hiện</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="padding:0.5rem 0.7rem; margin:0.2rem 0 0.4rem;
    background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.18);
    border-radius:8px; font-family:'Rajdhani',sans-serif; font-size:0.82rem; line-height:1.55;">
    <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:#8090C0;">
        <span style="color:#6070A0;">Tên sinh viên:</span>
        <b style="color:#E0E0FF; font-weight:600;">Đào Thị Hà Vy</b>
    </div>
    <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem; color:#8090C0;">
        <span style="color:#6070A0;">Mã sinh viên:</span>
        <b style="color:#E0E0FF; font-weight:600; letter-spacing:0.5px;">23051471</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset info
st.sidebar.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="nav-section">📂 Dữ liệu</div>', unsafe_allow_html=True)
info = get_dataset_info()
for name, data in info.items():
    status = "OK" if data["available"] else "Thiếu"
    color = "#00FF88" if data["available"] else "#FF6B35"
    st.sidebar.markdown(
        f'<div style="padding:0.25rem 0.6rem; font-family:Rajdhani,sans-serif; font-size:0.80rem;">'
        f'<span style="color:{color};">&#9679;</span> {name}: {data["rows"]} dòng {status}'
        f'</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("""
<div style="padding:0.6rem 0.8rem; margin-top:0.6rem; background:rgba(0,212,255,0.04);
    border:1px solid rgba(0,212,255,0.12); border-radius:8px;">
    <div style="font-family:Orbitron,monospace; font-size:0.62rem; color:#00D4FF; text-transform:uppercase;">
        Thông tin hệ thống
    </div>
    <div style="font-family:Rajdhani,sans-serif; font-size:0.78rem; color:#8090C0; margin-top:0.3rem; line-height:1.6;">
        Phiên bản: AIDEOM-VN v1.0<br>
        Số mô hình: 12<br>
        Dữ liệu: Việt Nam 2020-2025
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE ROUTER
# ============================================================
selected = st.session_state.selected_page
module_name = PAGES.get(selected)
if module_name:
    try:
        # Hiển thị tiêu đề trang lớn ở đầu mỗi trang
        if selected != "Trang chủ":
            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding: 1.2rem 0 0.8rem;
                    border-bottom: 2px solid rgba(0,212,255,0.25);
                    margin-bottom: 1.2rem;
                ">
                    <div style="
                        font-family:'Orbitron',monospace;
                        font-size:2rem;
                        font-weight:900;
                        letter-spacing:3px;
                        background:linear-gradient(90deg,#00D4FF,#BF00FF,#00FF88);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;
                        background-clip:text;
                        color:#00D4FF;
                    ">{selected.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        module = importlib.import_module(module_name)
        module.run()
    except Exception as e:
        st.error(f"Lỗi tại {selected}: {e}")
        import traceback
        with st.expander("Chi tiết lỗi"):
            st.code(traceback.format_exc())
