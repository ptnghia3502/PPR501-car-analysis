"""
=============================================================================
TRANSFORMER - DỰ BÁO GIÁ XE Ô TÔ CŨ TẠI TP. HỒ CHÍ MINH
=============================================================================

ỨNG DỤNG (Application):
    Dự báo (Prediction) giá bán xe ô tô cũ tại thị trường TP. Hồ Chí Minh
    dựa trên mô tả văn bản tiếng Việt của người dùng.

ĐỐI TƯỢNG SỬ DỤNG (Target Users):
    - Người dân TP.HCM có nhu cầu mua/bán xe cũ trên các sàn
      (Chợ Tốt, Bonbanh, Oto.com.vn).
    - Chủ salon xe cũ tại TP.HCM cần tham khảo giá thị trường.
    - Người mới lái xe lần đầu muốn định giá nhanh chiếc xe mình quan tâm.

PHƯƠNG PHÁP (Method):
    Fine-tune mô hình Transformer đa ngôn ngữ (DistilBERT Multilingual)
    cho bài toán hồi quy (regression) trên cột "Description" (tiếng Việt)
    của dataset xe cũ Việt Nam. So sánh với baseline Linear Regression.

INPUT (Đầu vào):
    - Văn bản mô tả tiếng Việt về chiếc xe, ví dụ:
      "Toyota Vios 2020, đi 3 vạn km, màu trắng, máy xăng 1.5L, số tự động"

OUTPUT (Đầu ra):
    - Giá bán dự đoán của chiếc xe (đơn vị: triệu VNĐ).

CÁCH CHẠY:
    py transformer_train.py
"""

# =============================================================================
# 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT
# =============================================================================
# Lý do: Tách phần import ra đầu file giúp dễ kiểm tra thư viện nào cần cài.
import re
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thư viện Transformer (Hugging Face)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# DistilBERT đa ngôn ngữ - đã được huấn luyện sẵn trên 104 ngôn ngữ,
# bao gồm tiếng Việt. Phù hợp để fine-tune cho bài toán hồi quy văn bản.
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# Scikit-learn dùng để chia tập train/test và tính metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")  # Tắt cảnh báo không quan trọng


# =============================================================================
# 2. CẤU HÌNH CHUNG (CONFIGURATION)
# =============================================================================
# Lý do: Gom tất cả hằng số vào 1 chỗ để dễ tinh chỉnh.

# Đường dẫn tới dataset CSV (cùng cấu trúc với LinearReg/linear_regression.py)
CSV_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "car-price.csv"

# Tên thành phố mà project này tập trung vào (phục vụ cho yêu cầu bài toán)
TARGET_CITY = "TP HCM"

# Model Transformer đa ngôn ngữ nhẹ (66M tham số), phù hợp máy cá nhân
MODEL_NAME = "distilbert-base-multilingual-cased"

# Cấu hình huấn luyện
MAX_LENGTH = 128        # Độ dài tối đa của mỗi mô tả (token)
BATCH_SIZE = 16         # Số mẫu xử lý mỗi lần (giảm nếu máy yếu)
EPOCHS = 3              # Số lần duyệt qua toàn bộ dữ liệu (3 là đủ - loss hội tụ ổn)
LEARNING_RATE = 5e-5    # Tốc độ học (tăng từ 2e-5 -> 5e-5 để hội tụ nhanh hơn)
SEED = 42               # Để kết quả có thể tái lập

# Chuẩn hoá giá: dùng log(price) thay vì price trực tiếp
# Lý do: giá xe dao động 100 triệu - 10 tỷ (chênh 100 lần), log sẽ nén
# về khoảng [4.6, 9.2]. Nếu không chuẩn hoá, MSE loss sẽ rất lớn và
# Transformer không hội tụ được.
USE_LOG_PRICE = True


def set_seed(seed):
    """Đảm bảo kết quả chạy ổn định giữa các lần (quan trọng cho báo cáo)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3. HÀM TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================

def convert_price_to_million(price_str):
    """
    CHUYỂN ĐỔI GIÁ TỪ CHUỖI TIẾNG VIỆT SANG SỐ TRIỆU VNĐ.

    Ví dụ:
        "1 Tỷ 118 Triệu" -> 1118.0
        "646 Triệu"      -> 646.0
        "2 Tỷ"           -> 2000.0

    Phương pháp: dùng regex tách phần "Tỷ" và phần "Triệu", sau đó cộng lại.
    Trả về NaN nếu không tách được (dữ liệu lỗi).
    """
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).strip()

    # Tìm phần "Tỷ" (1 tỷ = 1000 triệu)
    ty = re.search(r"([\d.,]+)\s*Tỷ", price_str, re.IGNORECASE)
    # Tìm phần "Triệu"
    trieu = re.search(r"([\d.,]+)\s*Triệu", price_str, re.IGNORECASE)

    # Chuyển "1,5" -> 1.5 (xử lý dấu phẩy thập phân kiểu Việt Nam)
    def to_float(match):
        return float(match.group(1).replace(",", "."))

    ty_value = to_float(ty) * 1000 if ty else 0
    trieu_value = to_float(trieu) if trieu else 0

    return ty_value + trieu_value


def clean_text(text):
    """
    LÀM SẠCH MÔ TẢ TIẾNG VIỆT TRƯỚC KHI ĐƯA VÀO TRANSFORMER.

    Các bước:
    1. Chuyển về chữ thường (giúp model không phân biệt hoa/thường).
    2. Bỏ ký tự đặc biệt thừa (giữ lại chữ cái tiếng Việt, số, dấu cách).
    3. Rút gọn nhiều khoảng trắng liên tiếp về 1 khoảng.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Giữ chữ cái (kể cả tiếng Việt có dấu), số, dấu cách và dấu phẩy
    text = re.sub(r"[^a-zA-Z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(csv_path, target_city):
    """
    TẢI VÀ CHUẨN BỊ DỮ LIỆU CHO BÀI TOÁN DỰ BÁO GIÁ XE TẠI TP.HCM.

    Quy trình:
    1. Đọc file CSV.
    2. Lọc các dòng thuộc TP.HCM (vì bài toán tập trung thị trường này).
    3. Chuyển cột "Price" từ chuỗi tiếng Việt sang số triệu VNĐ.
    4. Làm sạch cột "Description" (mô tả tiếng Việt).
    5. Loại bỏ các dòng thiếu dữ liệu quan trọng.

    Trả về: DataFrame đã sẵn sàng cho việc train model.
    """
    print("=" * 70)
    print(f"BƯỚC 1: TẢI DỮ LIỆU - LỌC XE TẠI {target_city}")
    print("=" * 70)

    dataset = pd.read_csv(csv_path)
    print(f"   Tổng số dòng trong dataset gốc: {len(dataset):,}")

    # Bước 1: Lọc theo thành phố - đây là yêu cầu cốt lõi của bài toán
    if "Location" in dataset.columns:
        dataset_city = dataset[dataset["Location"].str.contains(
            target_city, case=False, na=False
        )].copy()
        print(f"   Số dòng sau khi lọc theo {target_city}: {len(dataset_city):,}")
    else:
        dataset_city = dataset.copy()
        print("   Cảnh báo: Không thấy cột Location, dùng toàn bộ dataset.")

    # Bước 2: Chuyển cột Price sang số (triệu VNĐ)
    dataset_city["Price_Million"] = dataset_city["Price"].apply(
        convert_price_to_million
    )

    # Bước 3: Làm sạch mô tả tiếng Việt
    dataset_city["Description_Clean"] = dataset_city["Description"].apply(
        clean_text
    )

    # Bước 4: Loại bỏ dòng thiếu dữ liệu (giá hoặc mô tả)
    before = len(dataset_city)
    dataset_city = dataset_city.dropna(
        subset=["Price_Million", "Description_Clean"]
    )
    # Giữ lại chỉ những dòng có mô tả hợp lệ (ít nhất 10 ký tự)
    dataset_city = dataset_city[
        dataset_city["Description_Clean"].str.len() >= 10
    ]
    # Loại giá bất thường (≤ 0 triệu hoặc > 50 tỷ - outlier rõ ràng)
    dataset_city = dataset_city[
        (dataset_city["Price_Million"] > 0)
        & (dataset_city["Price_Million"] < 50000)
    ]
    after = len(dataset_city)
    print(f"   Số dòng hợp lệ sau khi lọc: {after:,} "
          f"(đã loại {before - after:,} dòng lỗi)")

    return dataset_city


# =============================================================================
# 4. ĐỊNH NGHĨA DATASET CHO PYTORCH
# =============================================================================
# Lý do: PyTorch cần dữ liệu ở dạng Dataset object để nạp theo batch.

class CarPriceDataset(Dataset):
    """
    Dataset ánh xạ (mô tả văn bản) -> (giá xe triệu VNĐ).

    Mỗi mẫu dữ liệu sẽ được tokenizer (DistilBERT) chuyển thành
    chuỗi ID số để đưa vào mô hình Transformer.
    """

    def __init__(self, texts, prices, tokenizer, max_length, use_log=True):
        """
        Args:
            texts: danh sách mô tả tiếng Việt
            prices: danh sách giá (triệu VNĐ)
            tokenizer: DistilBERT tokenizer
            max_length: độ dài tối đa token
            use_log: nếu True, sẽ chuyển giá sang log(price) - GIÚP MODEL HỘI TỤ
        """
        self.texts = texts
        # QUAN TRỌNG: nếu use_log=True, lưu log(price) thay vì price trực tiếp
        # Lý do: giá xe dao động 100 - 10000 (triệu). MSE trên scale này quá
        # lớn, model không học được. log(price) nén về khoảng [4.6, 9.2].
        if use_log:
            self.prices = np.log1p(np.array(prices, dtype=np.float64))
        else:
            self.prices = np.array(prices, dtype=np.float64)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_log = use_log

    def __len__(self):
        """Trả về số mẫu trong dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Trả về 1 mẫu (đã được mã hoá) tại vị trí idx."""
        text = str(self.texts[idx])
        price = float(self.prices[idx])

        # Bước QUAN TRỌNG nhất: chuyển văn bản thành tensor cho Transformer
        # padding='max_length': câu ngắn được đệm bằng token 0 cho đủ độ dài
        # truncation=True: câu dài bị cắt bớt
        # return_tensors='pt': trả về PyTorch tensor
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),       # Mã số của token
            "attention_mask": encoding["attention_mask"].squeeze(),  # Đánh dấu token thật
            "price": torch.tensor(price, dtype=torch.float32),  # Giá (label - log scale nếu use_log)
        }


# =============================================================================
# 5. HÀM TRAIN VÀ ĐÁNH GIÁ TRANSFORMER
# =============================================================================

def train_transformer_model(train_loader, val_loader, device, epochs, lr):
    """
    HUẤN LUYỆN (FINE-TUNE) MÔ HÌNH TRANSFORMER CHO BÀI TOÁN DỰ BÁO GIÁ.

    Phương pháp:
    - Dùng DistilBERT đã train sẵn, thêm 1 lớp linear ở cuối để ra 1 số (giá).
    - Đây là kỹ thuật "Fine-tuning" - tận dụng kiến thức ngôn ngữ đã học.

    Trả về: model đã train, danh sách loss mỗi epoch.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH TRANSFORMER (FINE-TUNING)")
    print("=" * 70)

    # Tải model đã train sẵn, cấu hình 1 output (regression)
    # num_labels=1 vì bài toán dự báo 1 số liên tục (giá)
    # problem_type="regression" để loss là MSE thay vì CrossEntropy
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
    )
    model.to(device)

    # Optimizer AdamW - tiêu chuẩn cho fine-tuning Transformer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Scheduler giảm learning rate dần theo thời gian
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(epochs):
        # ----------- PHA TRAIN -----------
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prices = batch["price"].to(device)

            # Reset gradient mỗi batch
            optimizer.zero_grad()

            # Forward pass: đưa văn bản qua Transformer
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=prices,  # Model tự tính MSE loss
            )
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass + cập nhật trọng số
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Tránh gradient nổ
            optimizer.step()
            scheduler.step()

            # In tiến trình mỗi 10 batch
            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/"
                      f"{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ----------- PHA ĐÁNH GIÁ -----------
        val_rmse, _, _ = evaluate_model(model, val_loader, device, USE_LOG_PRICE)
        history["val_rmse"].append(val_rmse)
        print(f"   Epoch {epoch+1}/{epochs} hoàn tất - "
              f"Train Loss: {avg_train_loss:.4f} - Val RMSE: {val_rmse:.2f} triệu VNĐ")

    return model, history


def evaluate_model(model, data_loader, device, use_log=True):
    """
    ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU (validation hoặc test).

    Args:
        use_log: nếu True, sẽ chuyển ngược từ log(price) về price (triệu VNĐ)
                 trước khi tính metric. Metric sẽ ở scale thực (triệu VNĐ).

    Các metric (đơn vị triệu VNĐ):
    - RMSE: sai số bình phương trung bình
    - MAE: sai số tuyệt đối trung bình
    - R²: hệ số xác định (1.0 = hoàn hảo, càng gần 1 càng tốt)
    """
    model.eval()  # Chuyển sang chế độ đánh giá
    predictions = []
    actuals = []

    with torch.no_grad():  # Không tính gradient khi đánh giá (tiết kiệm bộ nhớ)
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prices = batch["price"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            preds = outputs.logits.squeeze().cpu().numpy()
            predictions.extend(preds.tolist())
            actuals.extend(prices.cpu().numpy().tolist())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Nếu model dự đoán ở log-scale, chuyển ngược về giá thực (triệu VNĐ)
    # để metric RMSE/MAE/R² có ý nghĩa cho người dùng.
    if use_log:
        predictions = np.expm1(predictions)   # exp(x) - 1 (nghịch của log1p)
        actuals = np.expm1(actuals)

    # Đảm bảo giá trị hợp lệ (không âm)
    predictions = np.maximum(predictions, 0)
    actuals = np.maximum(actuals, 0)

    rmse = math.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return rmse, mae, r2


# =============================================================================
# 6. BASELINE: LINEAR REGRESSION (để so sánh)
# =============================================================================

def train_linear_regression_baseline(train_texts, train_prices, val_texts, val_prices):
    """
    MÔ HÌNH BASELINE - LINEAR REGRESSION.

    Mục đích: So sánh hiệu quả của Transformer với phương pháp truyền thống
    (Linear Regression - đã làm ở folder LinearReg trước đó).

    Vì Linear Regression không xử lý trực tiếp văn bản, ta dùng kỹ thuật
    TF-IDF đơn giản để chuyển mô tả thành vector số.

    Returns:
        dict: {"rmse", "mae", "r2", "val_preds", "model", "vectorizer"}
    """
    print("\n" + "=" * 70)
    print("BƯỚC 2 (PHỤ): BASELINE - LINEAR REGRESSION + TF-IDF")
    print("=" * 70)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from joblib import dump

    # Chuẩn hoá giá sang log-scale (giống Transformer) để so sánh công bằng
    train_prices_log = np.log1p(train_prices)
    val_prices_log = np.log1p(val_prices)

    # TF-IDF: chuyển văn bản thành vector số dựa trên tần suất từ
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    model = LinearRegression()
    model.fit(X_train, train_prices_log)  # Train trên log-price

    # Dự đoán ở log-scale rồi chuyển ngược
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)   # exp(x) - 1
    val_preds = np.maximum(val_preds, 0)

    rmse = math.sqrt(mean_squared_error(val_prices, val_preds))
    mae = mean_absolute_error(val_prices, val_preds)
    r2 = r2_score(val_prices, val_preds)

    print(f"   Baseline Linear Regression RMSE: {rmse:.2f} triệu VNĐ")
    print(f"   Baseline Linear Regression MAE:  {mae:.2f} triệu VNĐ")
    print(f"   Baseline Linear Regression R²:   {r2:.4f}")

    # Lưu model + vectorizer để dùng cho ensemble sau này
    save_dir = Path(__file__).resolve().parent / "saved_model"
    save_dir.mkdir(exist_ok=True)
    dump(model, save_dir / "linear_model.joblib")
    dump(vectorizer, save_dir / "tfidf_vectorizer.joblib")

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "val_preds": val_preds, "model": model, "vectorizer": vectorizer,
    }


# =============================================================================
# 7. VẼ BIỂU ĐỒ ĐÁNH GIÁ
# =============================================================================

def plot_results(history, transformer_metrics, baseline_metrics, val_actuals,
                 val_predictions, save_path):
    """
    VẼ BIỂU ĐỒ ĐÁNH GIÁ MÔ HÌNH (4 biểu đồ trong 1 figure).

    Biểu đồ gồm:
    1. Loss qua các epoch (đường học của Transformer)
    2. So sánh RMSE giữa Transformer và Baseline
    3. So sánh R² giữa Transformer và Baseline
    4. Scatter plot: Giá thực tế vs Giá dự đoán (đường chéo = dự đoán hoàn hảo)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Biểu đồ 1: Training Loss ---
    axes[0, 0].plot(range(1, len(history["train_loss"]) + 1),
                    history["train_loss"], marker="o", color="blue",
                    linewidth=2, label="Train Loss")
    axes[0, 0].set_title("Training Loss theo Epoch", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # --- Biểu đồ 2: So sánh RMSE ---
    models = ["Linear Regression\n(Baseline + TF-IDF)", "Transformer\n(DistilBERT)",
              "Ensemble\n(0.7 Trans + 0.3 Lin)"]
    rmses = [baseline_metrics["rmse"], transformer_metrics["rmse"],
             transformer_metrics.get("ensemble_rmse", transformer_metrics["rmse"])]
    colors_bar = ["#FF6B6B", "#4ECDC4", "#FFD93D"]
    bars = axes[0, 1].bar(models, rmses, color=colors_bar)
    axes[0, 1].set_title("So sánh RMSE (triệu VNĐ - càng thấp càng tốt)",
                         fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("RMSE (triệu VNĐ)")
    for bar, value in zip(bars, rmses):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5, f"{value:.0f}",
                        ha="center", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # --- Biểu đồ 3: So sánh R² ---
    r2s = [baseline_metrics["r2"], transformer_metrics["r2"],
           transformer_metrics.get("ensemble_r2", transformer_metrics["r2"])]
    bars = axes[1, 0].bar(models, r2s, color=colors_bar)
    axes[1, 0].set_title("So sánh R² (càng gần 1.0 càng tốt)",
                         fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("R² Score")
    axes[1, 0].set_ylim(min(0, min(r2s) - 0.05), 1)
    for bar, value in zip(bars, r2s):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{value:.4f}",
                        ha="center", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # --- Biểu đồ 4: Scatter thực tế vs dự đoán ---
    axes[1, 1].scatter(val_actuals, val_predictions, alpha=0.5, color="green", s=20)
    max_val = max(val_actuals.max(), val_predictions.max())
    axes[1, 1].plot([0, max_val], [0, max_val], "r--", linewidth=2,
                    label="Dự đoán hoàn hảo")
    axes[1, 1].set_title("Giá thực tế vs Giá dự đoán (tập Validation)",
                         fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Giá thực tế (triệu VNĐ)")
    axes[1, 1].set_ylabel("Giá dự đoán (triệu VNĐ)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("ĐÁNH GIÁ MÔ HÌNH DỰ BÁO GIÁ XE TẠI TP. HỒ CHÍ MINH",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"\n   Đã lưu biểu đồ đánh giá: {save_path}")
    plt.show()


# =============================================================================
# 8. HÀM CHÍNH - PIPELINE HOÀN CHỈNH
# =============================================================================

def main():
    """Pipeline hoàn chỉnh: Load → Train → Đánh giá → Vẽ biểu đồ."""

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Thiết bị train: {device}")

    # ----- BƯỚC 1: Load và chuẩn bị dữ liệu -----
    df = load_and_prepare_data(CSV_PATH, TARGET_CITY)
    if len(df) < 50:
        print("   Cảnh báo: Quá ít dữ liệu để train. Kiểm tra lại dataset.")
        return

    # Chia tập train / validation theo tỉ lệ 80/20
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED
    )
    print(f"   Tập train: {len(train_df):,} mẫu")
    print(f"   Tập validation: {len(val_df):,} mẫu")

    train_texts = train_df["Description_Clean"].values
    train_prices = train_df["Price_Million"].values
    val_texts = val_df["Description_Clean"].values
    val_prices = val_df["Price_Million"].values

    # ----- BƯỚC 2: Train baseline Linear Regression (để so sánh) -----
    baseline_metrics = train_linear_regression_baseline(
        train_texts, train_prices, val_texts, val_prices
    )

    # ----- BƯỚC 3: Train Transformer -----
    print(f"\n   Đang tải tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # Tạo PyTorch Dataset (dùng log-price để model hội tụ tốt hơn)
    train_dataset = CarPriceDataset(
        train_texts, train_prices, tokenizer, MAX_LENGTH, USE_LOG_PRICE
    )
    val_dataset = CarPriceDataset(
        val_texts, val_prices, tokenizer, MAX_LENGTH, USE_LOG_PRICE
    )

    # DataLoader: cung cấp dữ liệu theo batch
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Train model
    model, history = train_transformer_model(
        train_loader, val_loader, device, EPOCHS, LEARNING_RATE
    )

    # ----- BƯỚC 4: Đánh giá chi tiết trên tập validation -----
    print("\n" + "=" * 70)
    print("BƯỚC 4: ĐÁNH GIÁ CHI TIẾT TRANSFORMER TRÊN TẬP VALIDATION")
    print("=" * 70)
    val_rmse, val_mae, val_r2 = evaluate_model(model, val_loader, device, USE_LOG_PRICE)
    transformer_metrics = {"rmse": val_rmse, "mae": val_mae, "r2": val_r2}

    print(f"\n   KẾT QUẢ CUỐI CÙNG:")
    print(f"   ┌─────────────────────────────────────────────┐")
    print(f"   │  TRANSFORMER (DistilBERT)                   │")
    print(f"   │  RMSE: {val_rmse:>10.2f} triệu VNĐ          │")
    print(f"   │  MAE:  {val_mae:>10.2f} triệu VNĐ          │")
    print(f"   │  R²:   {val_r2:>10.4f}                      │")
    print(f"   └─────────────────────────────────────────────┘")
    print(f"   ┌─────────────────────────────────────────────┐")
    print(f"   │  BASELINE (Linear Regression + TF-IDF)      │")
    print(f"   │  RMSE: {baseline_metrics['rmse']:>10.2f} triệu VNĐ       │")
    print(f"   │  MAE:  {baseline_metrics['mae']:>10.2f} triệu VNĐ       │")
    print(f"   │  R²:   {baseline_metrics['r2']:>10.4f}                      │")
    print(f"   └─────────────────────────────────────────────┘")

    # Phần trăm cải thiện
    improvement = (
        (baseline_metrics["rmse"] - val_rmse) / baseline_metrics["rmse"] * 100
    )
    print(f"\n   Transformer cải thiện RMSE: {improvement:+.2f}% so với Baseline")

    # ----- BƯỚC 4.5: ENSEMBLE - KẾT HỢP TRANSFORMER + LINEAR REGRESSION -----
    # Ý tưởng: lấy trung bình có trọng số giữa 2 model để tăng độ chính xác.
    # Transformer hiểu ngữ nghĩa tốt, Linear bắt được pattern tuyến tính.
    print("\n" + "=" * 70)
    print("BƯỚC 4.5: ENSEMBLE - KẾT HỢP TRANSFORMER + LINEAR")
    print("=" * 70)

    # Lấy dự đoán từ Transformer trên tập val
    transformer_val_preds = []
    val_actuals_list = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            transformer_val_preds.extend(
                outputs.logits.squeeze().cpu().numpy().tolist()
            )
            val_actuals_list.extend(batch["price"].cpu().numpy().tolist())

    # Chuyển từ log-scale về giá thực
    transformer_val_preds = np.expm1(np.array(transformer_val_preds))
    val_actuals_arr = np.expm1(np.array(val_actuals_list))
    transformer_val_preds = np.maximum(transformer_val_preds, 0)
    val_actuals_arr = np.maximum(val_actuals_arr, 0)

    # Dự đoán từ Linear Regression (đã có sẵn trong baseline_metrics)
    linear_val_preds = np.array(baseline_metrics["val_preds"])

    # Ensemble = 0.7 * Transformer + 0.3 * Linear (trọng số ưu tiên Transformer)
    ensemble_preds = 0.7 * transformer_val_preds + 0.3 * linear_val_preds
    ensemble_rmse = math.sqrt(mean_squared_error(val_actuals_arr, ensemble_preds))
    ensemble_mae = mean_absolute_error(val_actuals_arr, ensemble_preds)
    ensemble_r2 = r2_score(val_actuals_arr, ensemble_preds)

    print(f"   Ensemble (0.7*Transformer + 0.3*Linear) RMSE: {ensemble_rmse:.2f} triệu VNĐ")
    print(f"   Ensemble MAE: {ensemble_mae:.2f} triệu VNĐ")
    print(f"   Ensemble R²:  {ensemble_r2:.4f}")

    # So sánh cải thiện
    best_rmse = min(val_rmse, baseline_metrics["rmse"], ensemble_rmse)
    if ensemble_rmse == best_rmse:
        print(f"   ✓ ENSEMBLE TỐT NHẤT!")
    elif val_rmse == best_rmse:
        print(f"   ✓ Transformer tốt nhất (Ensemble chưa cải thiện thêm)")
    transformer_metrics["ensemble_rmse"] = ensemble_rmse
    transformer_metrics["ensemble_mae"] = ensemble_mae
    transformer_metrics["ensemble_r2"] = ensemble_r2

    # ----- BƯỚC 5: Lấy dự đoán để vẽ scatter -----
    # Ưu tiên dùng ensemble nếu tốt hơn, ngược lại dùng Transformer
    val_predictions = ensemble_preds if ensemble_rmse < val_rmse else transformer_val_preds
    val_actuals = val_actuals_arr

    # ----- BƯỚC 6: Vẽ biểu đồ -----
    save_path = Path(__file__).resolve().parent / "transformer_result.png"
    plot_results(
        history, transformer_metrics, baseline_metrics,
        val_actuals, val_predictions, save_path
    )

    # ----- BƯỚC 7: Lưu model + metadata để dùng cho app -----
    save_dir = Path(__file__).resolve().parent / "saved_model"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # QUAN TRỌNG: lưu file cấu hình để app biết cách chuyển ngược log -> giá
    import json
    config = {"use_log_price": USE_LOG_PRICE}
    with open(save_dir / "app_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n   Đã lưu model vào: {save_dir}")
    print("\n   Hoàn tất! Bây giờ có thể chạy app bằng lệnh:")
    print("   py -m streamlit run app.py")

    # ----- BƯỚC 8 (tuỳ chọn): Hỏi user có muốn chạy app luôn không -----
    # Mục đích: giúp user khỏi phải mở terminal mới.
    # Nếu user gõ 'y' -> tự động khởi động Streamlit app trong subprocess.
    # Nếu user gõ 'n' hoặc Enter -> kết thúc bình thường.
    try:
        answer = input(
            "\n   Bạn có muốn chạy app ngay bây giờ không? (y/n, mặc định=n): "
        ).strip().lower()
    except EOFError:
        # Trường hợp chạy trong IDE không hỗ trợ input (ví dụ: Run cell)
        answer = "n"

    if answer == "y":
        print("\n   Đang khởi động Streamlit app...")
        print("   Mở trình duyệt tại: http://localhost:8501")
        print("   Nhấn Ctrl+C để dừng app.\n")
        import subprocess
        import sys
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve().parent / "app.py")]
        )
    else:
        print("\n   Bạn có thể chạy app sau bằng lệnh:")
        print("   py -m streamlit run app.py")


if __name__ == "__main__":
    main()
