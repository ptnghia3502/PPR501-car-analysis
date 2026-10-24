"""
=============================================================================
APP - GIAO DIỆN NGƯỜI DÙNG ĐỂ DỰ BÁO GIÁ XE Ô TÔ CŨ TẠI TP. HCM
=============================================================================

ỨNG DỤNG:
    Giao diện web cho phép người dân TP.HCM nhập mô tả chiếc xe ô tô cũ
    bằng tiếng Việt và nhận về giá bán dự báo.

ĐỐI TƯỢNG SỬ DỤNG:
    - Người dân TP.HCM muốn định giá nhanh chiếc xe mình đang quan tâm.
    - Chủ salon xe cũ tại TP.HCM cần tham khảo giá thị trường.

PHƯƠNG PHÁP BÊN DƯỚI:
    Mô hình Transformer (DistilBERT đa ngôn ngữ) đã được fine-tune
    trên dữ liệu mô tả xe cũ tại TP.HCM.

CÁCH CHẠY:
    py -m streamlit run app.py
    Sau đó mở trình duyệt tại http://localhost:8501
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt


# =============================================================================
# 1. CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(
    page_title="Dự Báo Giá Xe TP.HCM",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Dự Báo Giá Xe Ô Tô Cũ Tại TP. Hồ Chí Minh")
st.caption("Mô hình Transformer (DistilBERT đa ngôn ngữ) - "
           "Fine-tuned trên dữ liệu xe cũ tại TP.HCM")

# Sidebar: thông tin đề tài (phục vụ báo cáo môn học)
with st.sidebar:
    st.header("📋 Thông Tin Đề Tài")
    st.markdown("""
    **Ứng dụng:** Dự báo giá xe ô tô cũ  
    **Khu vực:** TP. Hồ Chí Minh  
    **Phương pháp:** Transformer (DistilBERT)  
    **Input:** Mô tả tiếng Việt  
    **Output:** Giá dự báo (triệu VNĐ)  
    **Đối tượng:** Người dân TP.HCM
    """)
    st.divider()
    st.caption("Dự án môn PPR501 - Nhóm sinh viên")


# =============================================================================
# 2. ĐƯỜNG DẪN & HÀM TẢI MODEL
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"
CSV_PATH = BASE_DIR.parent / "Dataset" / "car-price.csv"


@st.cache_resource
def load_model():
    """Tải model Transformer đã train + Linear model cho ensemble."""
    if not MODEL_DIR.exists():
        return None, None, False, None, None
    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    # Đọc cấu hình log-price (nếu model được train với log(price))
    use_log = False
    config_file = MODEL_DIR / "app_config.json"
    if config_file.exists():
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            use_log = cfg.get("use_log_price", False)
        except Exception:
            use_log = False

    # Load Linear Regression model cho ensemble (nếu có)
    linear_model = None
    vectorizer = None
    linear_path = MODEL_DIR / "linear_model.joblib"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    if linear_path.exists() and vectorizer_path.exists():
        try:
            from joblib import load
            linear_model = load(linear_path)
            vectorizer = load(vectorizer_path)
        except Exception:
            linear_model = None
            vectorizer = None

    return tokenizer, model, use_log, linear_model, vectorizer


@st.cache_data
def load_dataset_info():
    """Tải dataset để hiển thị thống kê cho người dùng."""
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH)
    df_hcm = df[df["Location"].str.contains("TP HCM", case=False, na=False)]
    return df_hcm


def clean_text(text):
    """Làm sạch văn bản đầu vào (giống lúc train)."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s,.]",
                  " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_price(model, tokenizer, text, max_length=128, use_log=False,
                  linear_model=None, vectorizer=None):
    """Dự báo giá cho 1 mô tả văn bản (kết hợp Transformer + Linear ensemble).

    Args:
        use_log: nếu True, model output ở log-scale -> chuyển ngược về giá thực.
        linear_model: model Linear Regression (optional, dùng cho ensemble).
        vectorizer: TF-IDF vectorizer (optional, dùng cho ensemble).

    Returns:
        dict: {"transformer", "linear", "ensemble"} - giá dự báo (triệu VNĐ)
    """
    text_clean = clean_text(text)
    encoding = tokenizer(
        text_clean,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )
        raw = output.logits.squeeze().item()

    # Chuyển từ log-scale về giá thực (nếu cần)
    transformer_price = np.expm1(raw) if use_log else raw
    transformer_price = max(0, transformer_price)

    # Dự đoán từ Linear Regression (nếu có)
    linear_price = None
    if linear_model is not None and vectorizer is not None:
        try:
            X = vectorizer.transform([text_clean])
            log_pred = linear_model.predict(X)[0]
            linear_price = max(0, np.expm1(log_pred) if use_log else log_pred)
        except Exception:
            linear_price = None

    # Ensemble = 0.7 * Transformer + 0.3 * Linear (nếu cả 2 đều có)
    if linear_price is not None:
        ensemble_price = 0.7 * transformer_price + 0.3 * linear_price
    else:
        ensemble_price = transformer_price

    return {
        "transformer": transformer_price,
        "linear": linear_price,
        "ensemble": ensemble_price,
    }


# =============================================================================
# 3. TẢI MODEL & DATASET
# =============================================================================
tokenizer, model, USE_LOG_PRICE, linear_model, vectorizer = load_model()
df_hcm = load_dataset_info()

if model is None:
    st.error("⚠️ Chưa có model. Vui lòng chạy `py transformer_train.py` trước.")
    st.stop()


# =============================================================================
# 4. PHẦN 1 - GIỚI THIỆU BÀI TOÁN
# =============================================================================
st.subheader("📌 Bài Toán")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Mô tả bài toán:** Dự báo giá bán xe ô tô cũ tại thị trường TP.HCM
    dựa trên mô tả văn bản tiếng Việt của người dùng.
    """)
with col2:
    if df_hcm is not None and len(df_hcm) > 0:
        st.metric("Số mẫu TP.HCM", f"{len(df_hcm):,}")
        st.metric("Phương pháp", "Transformer")
        st.metric("Loại bài toán", "Hồi quy (Regression)")


# =============================================================================
# 5. PHẦN 2 - NHẬP MÔ TẢ & DỰ BÁO
# =============================================================================
st.subheader("✍️ Nhập Mô Tả Chiếc Xe Của Bạn")

# Khởi tạo session state cho ô nhập
if "car_description" not in st.session_state:
    st.session_state["car_description"] = ""

# Các ví dụ mẫu để người dùng tham khảo
st.markdown("**💡 Ví dụ mẫu (click để dùng):**")
example_cols = st.columns(3)
examples = [
    "Toyota Vios 2020, đi 30000 km, màu trắng, máy xăng 1.5L, số tự động, "
    "5 chỗ, chính chủ",
    "Honda City 2019, màu đen, máy xăng 1.5L, số tay, đi 50000 km, "
    "còn mới",
    "Mazda 3 2021, màu đỏ, máy xăng 1.5L, số tự động, đi 20000 km, "
    "lắp ráp trong nước",
]
for col, ex in zip(example_cols, examples):
    with col:
        if st.button(f"📋 {ex[:30]}...", key=ex):
            st.session_state["car_description"] = ex

# Ô nhập văn bản chính
user_input = st.text_area(
    "Mô tả chiếc xe (tiếng Việt):",
    value=st.session_state.get("car_description", ""),
    height=120,
    placeholder="Ví dụ: Toyota Vios 2020, đi 30000 km, màu trắng, "
                "máy xăng 1.5L, số tự động, 5 chỗ",
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    predict_btn = st.button("🔮 Dự Báo Giá", type="primary", use_container_width=True)
with col_btn2:
    st.caption("Mô hình đã học từ hàng nghìn mô tả xe thực tế tại TP.HCM")


# =============================================================================
# 6. PHẦN 3 - HIỂN THỊ KẾT QUẢ
# =============================================================================
if predict_btn and user_input.strip():
    with st.spinner("Đang phân tích mô tả và dự báo giá..."):
        result = predict_price(
            model, tokenizer, user_input,
            use_log=USE_LOG_PRICE,
            linear_model=linear_model,
            vectorizer=vectorizer,
        )
        # Ưu tiên dùng ensemble nếu có, ngược lại dùng Transformer
        predicted_price = result["ensemble"]

    st.success("✅ Dự báo hoàn tất!")

    # Hiển thị kết quả chính
    st.markdown("### 💰 Kết Quả Dự Báo")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric(
            label="Giá dự báo (Ensemble)",
            value=f"{predicted_price:,.0f} triệu VNĐ",
        )
    with col_r2:
        st.metric(
            label="Tương đương",
            value=f"{predicted_price/1000:,.2f} tỷ VNĐ",
        )
    with col_r3:
        # Khoảng giá ước lượng (±15% sai số trung bình của model)
        lower = predicted_price * 0.85
        upper = predicted_price * 1.15
        st.metric(
            label="Khoảng giá ước lượng",
            value=f"{lower:,.0f} - {upper:,.0f}",
            delta="triệu VNĐ",
        )

    # Nếu có cả 3 model, hiển thị chi tiết để so sánh
    if result["linear"] is not None:
        st.markdown("#### 🔍 Chi tiết từng mô hình")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Transformer", f"{result['transformer']:,.0f} triệu")
        with c2:
            st.metric("Linear Regression", f"{result['linear']:,.0f} triệu")
        with c3:
            st.metric("Ensemble (kết hợp)", f"{result['ensemble']:,.0f} triệu",
                      delta=f"{result['ensemble'] - result['transformer']:+,.0f}")

    st.info(
        f"📌 **Ghi chú:** Mức giá trên được dự báo dựa trên phân tích "
        f"{len(df_hcm):,}+ mẫu tin đăng bán xe tại TP.HCM. "
        f"Kết quả mang tính chất tham khảo, giá thực tế có thể chênh lệch "
        f"tuỳ tình trạng xe, thời điểm bán và thương lượng giữa hai bên."
    )

    # Lưu lịch sử dự đoán trong session
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "Mô tả": user_input[:80] + "..." if len(user_input) > 80 else user_input,
        "Giá dự báo (triệu VNĐ)": f"{predicted_price:,.0f}",
    })


# =============================================================================
# 7. PHẦN 4 - LỊCH SỬ DỰ BÁO
# =============================================================================
if "history" in st.session_state and st.session_state["history"]:
    st.subheader("📜 Lịch Sử Dự Báo Trong Phiên")
    st.dataframe(
        pd.DataFrame(st.session_state["history"][-10:]),
        use_container_width=True,
    )


# =============================================================================
# 8. PHẦN 5 - THỐNG KÊ THỊ TRƯỜNG XE TP.HCM
# =============================================================================
if df_hcm is not None and len(df_hcm) > 0:
    st.subheader("📊 Thống Kê Thị Trường Xe Cũ TP.HCM")

    # Chuẩn bị dữ liệu giá
    def to_million(price_str):
        if pd.isna(price_str):
            return np.nan
        s = str(price_str).strip()
        ty = re.search(r"([\d.,]+)\s*Tỷ", s, re.IGNORECASE)
        tr = re.search(r"([\d.,]+)\s*Triệu", s, re.IGNORECASE)

        def f(m):
            return float(m.group(1).replace(",", "."))

        return (f(ty) * 1000 if ty else 0) + (f(tr) if tr else 0)

    df_hcm["Price_Million"] = df_hcm["Price"].apply(to_million)
    df_valid = df_hcm.dropna(subset=["Price_Million"])
    df_valid = df_valid[df_valid["Price_Million"] > 0]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Tổng tin đăng", f"{len(df_valid):,}")
    with col_s2:
        st.metric("Giá trung vị",
                  f"{df_valid['Price_Million'].median():,.0f} triệu")
    with col_s3:
        st.metric("Giá trung bình",
                  f"{df_valid['Price_Million'].mean():,.0f} triệu")
    with col_s4:
        st.metric("Giá cao nhất",
                  f"{df_valid['Price_Million'].max():,.0f} triệu")

    # Biểu đồ phân bố giá
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(df_valid["Price_Million"], bins=40, color="#4ECDC4",
            edgecolor="black", alpha=0.8)
    ax.set_title("Phân bố giá xe ô tô cũ tại TP. Hồ Chí Minh",
                 fontweight="bold")
    ax.set_xlabel("Giá (triệu VNĐ)")
    ax.set_ylabel("Số lượng tin đăng")
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)


# =============================================================================
# 9. FOOTER
# =============================================================================
st.divider()
st.caption(
    "📚 Đồ án môn PPR501 | Mô hình: DistilBERT Multilingual + Linear Ensemble | "
    "Dữ liệu: car-price.csv | Khu vực: TP. Hồ Chí Minh"
)
