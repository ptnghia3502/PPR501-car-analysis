# Transformer - Chatbot Gợi Ý Xe & Nhận Diện Ảnh

---

## 📌 Ứng dụng

Chatbot trò chuyện giúp **tìm xe ô tô cũ phù hợp** và **nhận diện xe từ ảnh** trên
thị trường Việt Nam. Người dùng gõ mô tả nhu cầu bằng tiếng Việt (ví dụ
*"Toyota Vios 2020, 5 chỗ, dưới 500 triệu"*) hoặc tải ảnh xe lên, chatbot sẽ:

1. **Lọc và xếp hạng xe** từ `car-price.csv` bằng **Sentence-Transformers embedding**
   (độ tương đồng ngữ nghĩa) kết hợp bộ lọc theo năm / số chỗ / hãng / loại xe.
2. **Nhận diện hãng xe từ ảnh** bằng CNN (TensorFlow/Keras) huấn luyện trực tiếp
   trên thư mục `sources/Dataset/Train` và `Test`.
3. **Gợi ý giá bán** bằng **Linear Regression** trên các đặc trưng của xe, kèm
   biểu đồ hồi quy và chỉ số đánh giá (R²).

## 👥 Đối tượng sử dụng

- **Người dân** muốn tìm mẫu xe cũ phù hợp ngân sách và nhu cầu.
- **Người bán xe** cần tham khảo giá thị trường để định giá nhanh chiếc xe.
- **Người mới mua xe** muốn kiểm tra xe đã chọn có hợp lý với giá niêm yết hay không.

## 🔬 Phương pháp

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| **Text ranking** | `sentence-transformers` — `Qwen/Qwen3-Embedding-0.6B` | Mã hóa câu hỏi và mô tả xe thành vector, xếp hạng theo độ tương đồng |
| **Bộ lọc điều kiện** | Regex + chuẩn hóa dấu tiếng Việt | Trích xuất năm, số chỗ, hãng, loại xe, khoảng giá từ câu hỏi |
| **Nhận diện ảnh** | CNN (TensorFlow/Keras) | Phân loại ảnh xe thuộc 7 nhóm hãng |
| **Gợi ý giá** | `sklearn` Linear Regression | Dự báo giá bán và trực quan hóa quan hệ giá – đặc trưng |

> **Lưu ý:** lần chạy đầu tiên sẽ tải mô hình embedding từ Hugging Face
> (`Qwen/Qwen3-Embedding-0.6B`), nên **cần kết nối Internet**. Các lần chạy sau
> dùng lại bản cache trong `~/.cache/huggingface`.

## 📥 Input / 📤 Output

**Input:**
- Văn bản tiếng Việt mô tả nhu cầu, ví dụ:
  > *"Xe Toyota 7 chỗ năm 2020 dưới 1 tỷ"*
- Hoặc ảnh xe (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) tải lên trực tiếp trong ô chat.

**Output:**
- Bảng **Top-N xe phù hợp** (tên, hãng, năm, giá, số km, đặc trưng) kèm khoảng giá.
- Với ảnh: **tên xe nhận diện** + **độ tin cậy** dự đoán.
- Biểu đồ Linear Regression và chỉ số R² của mô hình giá.

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài đặt thư viện

Cài đặt chung từ thư mục gốc repository (xem [`SETUP.md`](../../SETUP.md)):

```bash
python -m pip install -r requirements.txt
```

### 2. Chạy giao diện web (Streamlit)

```bash
cd sources/Transformer
streamlit run app.py --server.port 8502
```

Mở trình duyệt tại **http://localhost:8502**.

### 3. Chạy notebook

```bash
cd sources/Transformer
jupyter lab main.ipynb
```

Notebook thực hiện cùng quy trình lọc + xếp hạng Transformer và hiển thị kết quả
dạng bảng.

---

## 📁 Cấu trúc thư mục

```
sources/Transformer/
├── README.md                       # File này
├── app.py                          # Chatbot Streamlit (tìm xe + nhận diện ảnh + gợi ý giá)
├── main.ipynb                     # Notebook cùng quy trình, hiển thị kết quả dạng bảng
├── price_prediction_app.py         # App dự báo giá bằng DistilBERT (fine-tune hồi quy)
├── transformer_train.py            # Script train DistilBERT + baseline Linear Regression
├── transformer_result.png          # Biểu đồ kết quả (tạo sau khi train)
├── run_app.bat / run_app.sh        # Script phục vụ chạy nhanh app
├── saved_model/                    # Model DistilBERT đã train (tạo sau khi train)
└── models/                         # Model CNN nhận diện ảnh (tạo ở lần chạy đầu)
```

---

## 🖥️ Giao diện App (Streamlit)

App gồm các phần chính:

1. **Header:** tiêu đề + tên mô hình embedding đang dùng.
2. **Ô chat:** nhập câu hỏi tiếng Việt **hoặc** tải ảnh xe lên (hỗ trợ nhiều file).
3. **Lịch sử hội thoại:** hiển thị lại các lượt trao đổi trong phiên làm việc.
4. **Kết quả tìm xe:** bảng xe phù hợp + biểu đồ Linear Regression + chỉ số R².
5. **Kết quả nhận diện ảnh:** tên xe + độ tin cậy dự đoán.

---

## 🛠️ Khắc phục lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-----------|-----------|
| `Không tìm thấy car-price.csv` | Sai đường dẫn dataset | Đảm bảo có `sources/Dataset/car-price.csv` |
| `ModuleNotFoundError: sentence_transformers` | Thiếu thư viện | `python -m pip install sentence-transformers` |
| `ConnectionError` khi tải model embedding | Mất mạng lần tải đầu | Cần Internet lần đầu, sau đó dùng cache |
| `Chưa thể khởi tạo CNN` | Thiếu TensorFlow hoặc thiếu thư mục ảnh | Cài `tensorflow`, kiểm tra `sources/Dataset/Train` & `Test` |
| App khởi động chậm | Đang tải/train mô hình embedding hoặc CNN | Đợi hoàn tất lần đầu, các lần sau dùng cache |
| `Chưa có model` khi chạy `price_prediction_app.py` | Chưa chạy `transformer_train.py` | Chạy file train trước |
| `Out of memory` khi train DistilBERT | RAM/GPU không đủ | Giảm `BATCH_SIZE` xuống 8 hoặc 4 trong `transformer_train.py` |

---

## 📚 Tài liệu tham khảo

- [Sentence Transformers docs](https://www.sbert.net/)
- [Qwen3 Embedding - Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [DistilBERT paper](https://arxiv.org/abs/1910.01108) - Sanh et al., 2019
- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)
- Dataset: [car-price.csv trên Kaggle](https://www.kaggle.com/datasets/qucvinhdng/automobile-dataset)

---

## 👨‍🎓 Thông tin đồ án môn học

- **Môn học:** PPR501 - Phân tích & Trực quan hoá Dữ liệu
- **Phương pháp:** Transformer (embedding + CNN + Linear Regression)
- **Khu vực:** Việt Nam
- **Đối tượng:** Người dân, người mua/bán xe cũ

---

For the shared environment setup and commands to run this application, see the
repository-level [SETUP.md](../../SETUP.md). Install dependencies from the
repository-root `requirements.txt`; do not install a separate per-app package
set.
