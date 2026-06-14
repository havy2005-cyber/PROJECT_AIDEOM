<div align="center">

# 🇻🇳 AIDEOM-VN

### *AI-Driven Decision Optimization Model for Vietnam*
*Hệ thống hỗ trợ phân tích, mô phỏng và tối ưu hóa quyết định phát triển kinh tế Việt Nam*

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-00D4FF?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-BF00FF?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Research-00FF88?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-FF6B35?style=for-the-badge)

<br/>

**12 mô hình tối ưu** &nbsp;·&nbsp; **3 bộ dữ liệu thực** &nbsp;·&nbsp; **6 vùng kinh tế** &nbsp;·&nbsp; **10 ngành** &nbsp;·&nbsp; **6 năm vĩ mô (2020–2025)**

<br/>

```text
  ___    _  _____ ___  __  __   __  __  ___  __  __ ___ ___
 / _ \  / \|_   _| _ \/  \|  \ |  \/  || __||  \/  | __| _ \
| (_) |/ _ \ | | |   / /\ | |) | |\/| || _| | |\/| | _||   /
 \___//_/ \_\|_| |_|_\/_/\_\___/|_|  |_|___|_|  |_|___|_|_\
  Vietnamese Economic Decision Support & Optimization Suite
```

</div>

---

## 📑 Mục lục

1. [Giới thiệu](#-giới-thiệu)
2. [Tính năng nổi bật](#-tính-năng-nổi-bật)
3. [12 mô hình trong hệ thống](#-12-mô-hình-trong-hệ-thống)
4. [Bộ dữ liệu](#-bộ-dữ-liệu)
5. [Cấu trúc dự án](#-cấu-trúc-dự-án)
6. [Cài đặt & Chạy](#-cài-đặt--chạy)
7. [Luồng hoạt động](#-luồng-hoạt-động)
8. [Thiết kế giao diện](#-thiết-kế-giao-diện)
9. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
10. [Hướng phát triển](#-hướng-phát-triển)
11. [Giấy phép & Trích dẫn](#-giấy-phép--trích-dẫn)

---

## 🇻🇳 Giới thiệu

**AIDEOM-VN** là bộ công cụ phân tích & tối ưu hóa quyết định phát triển kinh tế Việt Nam,
xây dựng trên nền tảng **Python · Streamlit · AI · Data Science**.

Hệ thống tích hợp **12 mô hình tối ưu hóa** trải dài từ **Hàm Sản Xuất Cobb–Douglas**,
**Quy Hoạch Tuyến Tính (LP)**, **MIP**, **TOPSIS**, **NSGA-II**, **Mô hình động lực học**,
đến **Monte Carlo** và **Q-Learning** — tất cả nhằm cung cấp **khuyến nghị chính sách**
dựa trên dữ liệu thực tế giai đoạn **2020–2025**.

> 🎯 Mục tiêu: biến dữ liệu kinh tế vĩ mô & vi mô của Việt Nam thành **quyết định có căn cứ khoa học**.

---

## ✨ Tính năng nổi bật

| Tính năng | Mô tả |
|---|---|
| 🧮 **12 mô hình tối ưu** | Từ hồi quy tuyến tính đến NSGA-II đa mục tiêu & Q-Learning |
| 🇻🇳 **Dữ liệu thực VN** | Vĩ mô 6 năm (2020–2025), 6 vùng kinh tế, 10 ngành trọng điểm |
| 🎨 **Giao diện Neon Dark** | Glassmorphism + gradient cyan–purple–green, font Orbitron & Rajdhani |
| 📊 **Biểu đồ Plotly tương tác** | 12+ template: line, bar, pie, scatter, heatmap, radar, Pareto, funnel, Pareto Front |
| 🔄 **Sidebar 6 mục phân loại** | Trang chủ · Phân tích kinh tế · Tối ưu hóa · AI & Mô phỏng · Dự báo · Dashboard |
| 🧰 **Utility kit đầy đủ** | `data_loader`, `helpers` (TOPSIS, MAPE, RMSE, monte carlo, …), `charts`, `report_generator`, `styles` |
| 🛡️ **Robust** | Tự phát hiện sklearn/pulp, fallback graceful, bảng tên cột tự động |
| 📥 **Báo cáo tự động** | Sinh báo cáo text / HTML / LaTeX từ kết quả mô hình |

---

## 🧬 12 mô hình trong hệ thống

> Mỗi bài là một module độc lập trong `modules/`, có giao diện Streamlit riêng và dùng dữ liệu từ `data/`.

| # | Bài | Tên mô hình | Loại | Kết quả chính |
|---|---|---|---|---|
| 01 | [Bài 01](modules/bai1_cobb_douglas.py) | **Cobb-Douglas + AI** | Hồi quy | Dự báo GDP, TFP, Growth Accounting |
| 02 | [Bài 02](modules/bai2_lp_budget.py) | **LP Ngân sách** | Tuyến tính | Phân bổ ngân sách 10 ngành |
| 03 | [Bài 03](modules/bai3_priority_ranking.py) | **TOPSIS Ngành** | Đa tiêu chí | Xếp hạng ưu tiên 10 ngành |
| 04 | [Bài 04](modules/bai4_lp_region.py) | **LP Vùng** | Tuyến tính | Phân bổ nguồn lực 6 vùng |
| 05 | [Bài 05](modules/bai5_mip_projects.py) | **MIP Dự án** | Số nguyên hỗn hợp | Chọn 15/22 dự án tối ưu |
| 06 | [Bài 06](modules/bai6_topsis_region.py) | **TOPSIS Vùng** | Đa tiêu chí | Xếp hạng 6 vùng kinh tế |
| 07 | [Bài 07](modules/bai7_nsga2.py) | **NSGA-II** | Đa mục tiêu | Pareto Front GDP – Lao động – AI |
| 08 | [Bài 08](modules/bai8_dynamic_model.py) | **Hệ thống động** | Phương trình vi phân | Dự báo GDP 2026 – 2035 |
| 09 | [Bài 09](modules/bai9_ai_labor.py) | **AI & Lao động** | Phân tích tác động | Thay thế & tạo việc làm AI |
| 10 | [Bài 10](modules/bai10_stochastic.py) | **Monte Carlo** | Stochastic | Khoảng dự báo GDP 90% CI |
| 11 | [Bài 11](modules/bai11_qlearning.py) | **Q-Learning** | Học tăng cường | Chính sách đầu tư vùng |
| 12 | [Bài 12](modules/bai12_dashboard.py) | **Dashboard** | Tổng hợp | Tổng hợp 12 mô hình |

### Phân nhóm thuật toán

```text
┌─────────────────────────────────────────────────────────────────┐
│  REGRESSION           │ LP/MIP              │  MULTI-CRITERIA   │
│  • Bài 1  Cobb-Douglas│ • Bài 2  LP Budget  │  • Bài 3 TOPSIS  │
│                       │ • Bài 4  LP Region  │    (Sector)       │
│  DYNAMICS             │ • Bài 5  MIP Project│  • Bài 6 TOPSIS  │
│  • Bài 8  System Dyn. │                     │    (Region)       │
│                       │ METAHEURISTIC       │                   │
│  STOCHASTIC           │ • Bài 7  NSGA-II    │  RL               │
│  • Bài 10 Monte Carlo │   (Pareto Front)    │  • Bài 11 Q-Learn │
│                       │                     │                   │
│  ML IMPACT            │ DASHBOARD           │                   │
│  • Bài 9  AI & Labor  │ • Bài 12 Tổng hợp  │                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Bộ dữ liệu

Dữ liệu thực nằm trong thư mục [`data/`](data/), tự động load bởi `utils/data_loader.py`.

| File | Mô tả | Dòng | Cột chính |
|---|---|---|---|
| [`vietnam_macro_2020_2025.csv`](data/vietnam_macro_2020_2025.csv) | Vĩ mô VN 2020–2025 | 6 năm | GDP, tăng trưởng, FDI, xuất/nhập khẩu, lạm phát, kinh tế số, năng suất lao động |
| [`vietnam_regions_2024.csv`](data/vietnam_regions_2024.csv) | 6 vùng kinh tế | 6 vùng | GRDP, dân số, FDI, số hóa, AI Readiness, R&D, Gini |
| [`vietnam_sectors_2024.csv`](data/vietnam_sectors_2024.csv) | 10 ngành kinh tế | 10 ngành | GDP share, xuất khẩu, số hóa, AI Readiness, FDI, rủi ro tự động hóa, R&D |

> ⚙️ Tên cột được **tự động chuẩn hóa** theo canonical name → an toàn khi CSV đổi thứ tự hoặc khoảng trắng.

### 6 vùng kinh tế

```text
1. Trung du và miền núi phía Bắc        (Northern Midlands & Mountains)
2. Đồng bằng Sông Hồng                  (Red River Delta)
3. Bắc Trung Bộ & Duyên hải Trung Bộ   (North Central & South Central Coast)
4. Tây Nguyên                           (Central Highlands)
5. Đông Nam Bộ                          (Southeast)
6. Đồng bằng Sông Cửu Long              (Mekong Delta)
```

### 10 ngành trọng điểm

```text
Nông-Lâm-Thủy sản · Công nghiệp chế biến · Xây dựng · Khai khoáng ·
Bán buôn–Bán lẻ · Tài chính–Ngân hàng–Bảo hiểm ·
Logistics–Vận tải–Kho bãi · CNTT–Truyền thông · Giáo dục–Đào tạo · Y tế
```

---

## 🏗 Cấu trúc dự án

```text
F fix/
│
├── app.py                          # 🚪 Entry point: registry trang + sidebar + router
├── requirements.txt                # 📦 Thư viện cần cài
├── README.md                       # 📖 File này
│
├── data/                           # 💾 Dữ liệu thực (CSV)
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_regions_2024.csv
│   └── vietnam_sectors_2024.csv
│
├── modules/                        # 🧩 12 mô hình + trang chủ
│   ├── home.py                     # 🏠 Dashboard tổng quan
│   ├── bai1_cobb_douglas.py
│   ├── bai2_lp_budget.py
│   ├── bai3_priority_ranking.py
│   ├── bai4_lp_region.py
│   ├── bai5_mip_projects.py
│   ├── bai6_topsis_region.py
│   ├── bai7_nsga2.py
│   ├── bai8_dynamic_model.py
│   ├── bai9_ai_labor.py
│   ├── bai10_stochastic.py
│   ├── bai11_qlearning.py
│   └── bai12_dashboard.py
│
└── utils/                          # 🔧 Bộ tiện ích dùng chung
    ├── styles.py                   # 🎨 Neon theme + KPI card + section header
    ├── data_loader.py              # 📥 Load & chuẩn hóa CSV
    ├── helpers.py                  # 🧮 MAPE, RMSE, TOPSIS, monte carlo, normalize, …
    ├── charts.py                   # 📊 12+ template Plotly (line/bar/radar/pareto/…)
    └── report_generator.py         # 📄 Báo cáo text/HTML/LaTeX
```

### Vai trò từng file

| File | Vai trò |
|---|---|
| [`app.py`](app.py) | Khai báo `PAGES`, `SECTIONS`, `PAGE_TITLES` · Render sidebar · Router module |
| `modules/home.py` | KPI tổng quan, navigation cards, biểu đồ GDP 2020–2025, bảng 12 mô hình |
| `modules/baiN_*.py` | Mỗi file export `def run():` — render UI Streamlit cho bài tương ứng |
| `utils/styles.py` | Theme neon dark (CSS) + factory `kpi_card`, `section_header`, `glass_card`, `highlight_box` |
| `utils/data_loader.py` | `load_macro()` · `load_regions()` · `load_sectors()` + auto column mapping |
| `utils/helpers.py` | TOPSIS, MAPE/RMSE, normalize (minmax/zscore), monte carlo, format VND, dịch tên vùng/ngành |
| `utils/charts.py` | 12+ hàm `line_chart`, `bar_chart`, `pie_chart`, `scatter_chart`, `heatmap_chart`, `radar_chart`, `pareto_front_chart`, `reward_curve_chart`, … |
| `utils/report_generator.py` | Sinh báo cáo text/HTML/LaTeX, summary stats, correlation insights, export CSV |

---

## ⚡ Cài đặt & Chạy

### 1. Yêu cầu

- **Python ≥ 3.10**
- Hệ điều hành: Windows / macOS / Linux

### 2. Cài đặt thư viện

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501` 🚀

### 4. Tuỳ chọn môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `STREAMLIT_THEME` | `dark` | Streamlit tự nhận từ CSS global |
| `STREAMLIT_SERVER_PORT` | `8501` | Đổi cổng nếu cần |

---

## 🔄 Luồng hoạt động

```text
       ┌────────────────────────────────────────────────────┐
       │                   app.py  (Entry)                  │
       │                                                    │
       │  1. apply_page_config()    → set wide + icon 📊   │
       │  2. apply_global_style()   → inject CSS neon dark   │
       │  3. st.session_state.selected_page ← sidebar pick │
       │  4. render sidebar (6 sections, 12 bài)            │
       │  5. lookup PAGES[selected] → importlib.module      │
       │  6. module.run()  → render nội dung bài           │
       └────────────────┬───────────────────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   load_macro()   load_regions()   load_sectors()       (utils/data_loader.py)
        │               │                │
        └───────────────┼────────────────┘
                        ▼
              ┌─────────────────────┐
              │  pd.DataFrame chuẩn  │
              │  hóa tên cột         │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  modules/bai*.py    │
              │  • Hồi quy / LP /   │
              │    MIP / TOPSIS /   │
              │    NSGA-II / Q-Lrn  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  utils/charts.py    │
              │  Plotly figures     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  utils/styles.py    │
              │  KPI card / glass / │
              │  section header     │
              └──────────┬──────────┘
                         ▼
                    🖥️  Streamlit UI
```

---

## 🎨 Thiết kế giao diện

Hệ thống dùng **neon dark theme** với 3 nhóm màu chính:

| Token | Hex | Sử dụng |
|---|---|---|
| `NEON_BLUE` | `#00D4FF` | Tiêu đề, KPI chính, gradient chính |
| `NEON_PURPLE` | `#BF00FF` | Nhấn phụ, AI/mô phỏng, gradient |
| `NEON_GREEN` | `#00FF88` | Tăng trưởng, kịch bản tích cực |
| `NEON_ORANGE` | `#FF6B35` | Cảnh báo, kịch bản bi quan |
| `BG_DARK` | `#0A0E27` | Nền chính, sidebar |
| `CARD_BG` | `rgba(15,20,50,0.75)` | Glass card backdrop blur |

### Thành phần UI tái sử dụng (factory trong `utils/styles.py`)

```python
section_header("GDP VIET NAM", "blue")      # 4 variant: blue/purple/green/orange
kpi_card("12847.6T", "GDP 2025", "Tỷ VND", NEON_BLUE)
glass_card("<nội dung html>")
highlight_box("Câu khẳng định nổi bật")
```

### Fonts

- **`Orbitron`** — tiêu đề, badge, con số KPI (sci-fi, futuristic)
- **`Rajdhani`** — body, mô tả, table (modern, dễ đọc tiếng Việt)

### Plotly template

Mọi biểu đồ tự động dùng `PLOTLY_TEMPLATE` trong `utils/styles.py` — nền trong suốt, lưới tối, font neon, colorway cyan→purple→green→orange→gold.

---

## 🛠 Công nghệ sử dụng

| Lĩnh vực | Thư viện | Vai trò |
|---|---|---|
| **UI framework** | `streamlit ≥ 1.35` | Multi-page app, widget, layout |
| **Data** | `pandas ≥ 2.0`, `numpy ≥ 1.26` | Xử lý dữ liệu, mảng số |
| **Visualization** | `plotly ≥ 5.20` | Biểu đồ tương tác |
| **Optimization** | `pulp ≥ 2.8` | LP / MIP solver |
| **ML & Hồi quy** | `scikit-learn ≥ 1.5` | Linear regression cho Cobb-Douglas |
| **Stats & Sci** | `scipy ≥ 1.13` | Tích phân, ODE cho mô hình động lực học |
| **Plots phụ** | `matplotlib ≥ 3.8`, `seaborn ≥ 0.13` | Biểu đồ phụ trợ |

### Tiêu chuẩn code

- Mỗi bài có cấu trúc 4 phần: **MỤC TIÊU · DỮ LIỆU · PHƯƠNG PHÁP · KẾT QUẢ · KHUYẾN NGHỊ**
- Mọi module export `def run():` để router `app.py` gọi
- `sys.path.insert(0, ...)` ở đầu mỗi module `bai*` để chạy độc lập
- Tên cột CSV được chuẩn hoá tự động — robust khi đổi thứ tự cột
- Hằng số màu tập trung ở `utils/styles.py` → dễ re-theme
- `try/except ImportError` cho `pulp`, `sklearn` → fallback an toàn

---

## 🚀 Hướng phát triển

```text
[ ]  Tích hợp dữ liệu thời gian thực từ GSO/World Bank API
[ ]  Thêm mô hình Bayesian / VAR cho dự báo chuỗi thời gian
[ ]  Dashboard tương tác nâng cao (filter năm/vùng/ngành)
[ ]  Xuất báo cáo PDF/Word tự động
[ ]  Đa ngôn ngữ (Anh / Việt)
[ ]  Triển khai Docker + CI/CD
[ ]  Bổ sung thuật toán Deep RL (DQN, PPO) cho bài 11
[ ]  So sánh nhiều kịch bản chính sách cùng lúc (S1–S5)
```

---

## 📜 Giấy phép & Trích dẫn

Dự án phát triển phục vụ **nghiên cứu & giáo dục**. Mọi dữ liệu trong `data/`
là dữ liệu thực từ **Tổng cục Thống kê Việt Nam (GSO)**, **World Bank**, và **IMF**
giai đoạn 2020–2025.

Nếu sử dụng cho nghiên cứu, vui lòng trích dẫn:

```bibtex
@software{aideom_vn_2026,
  title  = {AIDEOM-VN: AI-Driven Decision Optimization Model for Vietnam},
  author = {AIDEOM-VN Team},
  year   = {2026},
  url    = {https://github.com/your-org/aideom-vn}
}
```

---
<div align="center">

**🇻🇳 Xây dựng trên Python · Streamlit · Scikit-learn · PuLP · NumPy · SciPy · Plotly 🇻🇳**

*Được tạo với ❤️ cho cộng đồng phân tích kinh tế Việt Nam*

</div>
