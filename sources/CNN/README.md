# CNN — Phân loại ảnh xe thuộc 7 nhóm

> Phần CNN của dự án **PPR501 Car Analysis** — hệ thống nhận diện, định giá
> và gợi ý xe hơi tại Việt Nam.

---

## Mục lục

1. [Tóm tắt project](#1-tóm-tắt-project)
2. [Bài toán & dữ liệu](#2-bài-toán--dữ-liệu)
3. [Vấn đề "dự đoán dồn về Audi" và cách sửa](#3-vấn-đề-dự-đoán-dồn-về-audi-và-cách-sửa)
4. [Kiến trúc model](#4-kiến-trúc-model)
5. [Cài đặt môi trường](#5-cài-đặt-môi-trường)
6. [Cách chạy (terminal)](#6-cách-chạy-terminal)
7. [Cách chạy bằng Jupyter Notebook](#7-cách-chạy-bằng-jupyter-notebook)
8. [Output sinh ra](#8-output-sinh-ra)
9. [Đọc kết quả & đánh giá](#9-đọc-kết-quả--đánh-giá)
10. [Tuỳ chọn CLI đầy đủ](#10-tuỳ-chọn-cli-đầy-đủ)
11. [Troubleshooting](#11-troubleshooting)
12. [Cấu trúc thư mục](#12-cấu-trúc-thư-mục)

---

## 1. Tóm tắt project

**PPR501 Car Analysis** là project cuối kỳ môn *Phân tích & Trực quan hóa dữ
liệu*, tập trung vào **phân tích thị trường ô tô Việt Nam (1990–2023)**.
Project gồm **3 ứng dụng độc lập**, tất cả dùng chung một môi trường ảo
`.venv` ở thư mục gốc:

| Thư mục | Ứng dụng | Công nghệ | Lệnh khởi chạy |
|---|---|---|---|
| `sources/CNN` | Phân loại ảnh xe (7 nhóm) | TensorFlow / Keras / MobileNetV2 | `python train_cnn.py --mode serve` |
| `sources/LinearReg` | Dự đoán giá xe (hồi quy tuyến tính) | Scikit-learn / Streamlit | `streamlit run app.py --server.port 8501` |
| `sources/Transformer` | Chatbot gợi ý xe theo câu hỏi | Sentence-Transformers / Streamlit | `streamlit run app.py --server.port 8502` |

**Phần CNN** (thư mục này) là một ứng dụng web: người dùng tải ảnh xe lên,
model trả về **xác suất thuộc 7 nhóm xe** kèm biểu đồ phân bố xác suất.

### Ba nguồn dữ liệu

| Nguồn | Nội dung | Dùng cho |
|---|---|---|
| Kaggle — [Ảnh xe](https://www.kaggle.com/code/kshitij192/car-images-classification-using-cnn/data) | 3.352 ảnh train + 813 ảnh test, 7 lớp | CNN |
| Kaggle — [Giá xe](https://www.kaggle.com/datasets/qucvinhdng/automobile-dataset) | CSV giá xe (tên, năm, km, công suất, vị trí…) | LinearReg, Transformer |
| Repo nội bộ `sources/Dataset/` | Ảnh + CSV đã tải sẵn về | Tất cả |

---

## 2. Bài toán & dữ liệu

**Đầu vào:** một ảnh xe. **Đầu ra:** xác suất thuộc một trong 7 nhóm.

### 7 lớp và số ảnh

| Lớp | Thư mục | Train | Test |
|---|---|---|---|
| Audi | `Audi` | 814 | 199 |
| Hyundai Creta | `Hyundai Creta` | 271 | 67 |
| Mahindra Scorpio | `Mahindra Scorpio` | 316 | 75 |
| Rolls Royce | `Rolls Royce` | 311 | 74 |
| Maruti Suzuki Swift | `Swift` | 424 | 102 |
| Tata Safari | `Tata Safari` | 441 | 106 |
| Toyota Innova | `Toyota Innova` | 775 | 190 |
| **Tổng** | | **3.352** | **813** |

> Lưu ý tên thư mục trên đĩa là `Swift` nhưng tên lớp hiển thị là
> `Maruti Suzuki Swift`. Mapping này do Keras tự suy ra từ thứ tự alphabet của
> thư mục và được ghi lại trong `labels.json` / `meta.json` — **không** hard-code.

### Cấu trúc dataset (bắt buộc giữ nguyên)

```text
sources/Dataset/
├── car-price.csv          # dùng bởi LinearReg + Transformer
├── Train/
│   ├── Audi/              # *.jpg
│   ├── Hyundai Creta/
│   ├── Mahindra Scorpio/
│   ├── Rolls Royce/
│   ├── Swift/
│   ├── Tata Safari/
│   └── Toyota Innova/
└── Test/
    └── ... (cùng 7 thư mục)
```

### Chia tập dữ liệu

```text
Dataset/Train (3.352 ảnh)
   ├── 80% (2.682) → tập train      (có augmentation)
   └── 20% ( 670) → tập validation   (KHÔNG augmentation)
Dataset/Test  ( 813 ảnh) → chỉ dùng để đánh giá lần cuối (KHÔNG augmentation)
```

**Quan trọng:** tập test **không bao giờ** được dùng làm `validation_data`,
và **không bao giờ** được dùng để chọn siêu tham số. Mọi quyết định (early
stopping, chọn learning rate, chọn hệ số prior correction) đều dựa trên tập
validation.

---

## 3. Vấn đề "dự đoán dồn về Audi" và cách sửa

### Triệu chứng

Phiên bản cũ cho ra kết quả kiểu:

```text
Audi: 99.5%   Hyundai Creta: 0.2%   Mahindra Scorpio: 0.1%
Rolls Royce: 0.1%   Swift: 0.05%   Tata Safari: 0.03%   Toyota Innova: 0.02%
```

Nói cách khác: **gần như mọi ảnh đều bị dự đoán là Audi**, kể cả ảnh Rolls
Royce hay Tata Safari.

### Nguyên nhân gốc rễ

| # | Nguyên nhân | Giải thích |
|---|---|---|
| 1 | **Dataset mất cân bằng** | Audi 814 ảnh vs Hyundai Creta 271 ảnh — tỉ lệ 3:1 |
| 2 | **CNN train từ đầu trên 3.352 ảnh** | Quá ít dữ liệu để học đặc trưng "lưới xe Rolls Royce". Model chỉ học được "đây là ảnh xe" và dùng prior của lớp đông nhất |
| 3 | **`Flatten(123008) → Dense(96)`** | 11,8M tham số (99,9% model) nằm ở tầng FC → overfit nặng, model ghi nhớ tập train |
| 4 | **`class_weight` inverse-frequency quá mạnh** | Đẩy lớp hiếm lên quá nhiều → recall lớp hiếm = 0, model mất ổn định |
| 5 | **Prior bias còn sót lại** | Dù đã dùng `class_weight`, xác suất đầu ra vẫn nghiêng về lớp đa số |

### Giải pháp đã áp dụng

| # | Sửa gì | Cách làm |
|---|---|---|
| 1 | **Transfer learning** | MobileNetV2 pre-trained trên **ImageNet** (1,2M ảnh). Model đã biết sẵn về hình khối, texture, vật liệu → chỉ cần vài epoch fine-tune |
| 2 | **Head gọn** | `GlobalAveragePooling2D` + `BatchNorm` + `Dropout` thay cho `Flatten` + `Dense` khổng lồ. 11,8M → 165K tham số trainable ở stage 1 |
| 3 | **Effective-number class weights** | Công thức Cui et al. 2019: `w_c = (1-β)/(1-β^n_c)`, β=0.999. Mềm hơn inverse-frequency nhiều, **đã normalize về trung bình 1.0** |
| 4 | **Prior correction (logit adjustment)** | `q_c ∝ p_c / π_c^α`. Cỡ α được chọn tự động trên **validation** theo **balanced accuracy** |
| 5 | **Đánh giá bằng balanced accuracy** | Accuracy thường bị lớp lớn chi phối; balanced accuracy = trung bình recall 7 lớp mới phản ánh đúng chất lượng |
| 6 | **Theo dõi phân bố nhãn dự đoán** | In ra số lượng dự đoán mỗi lớp → phát hiện ngay model có bị "dồn về một lớp" hay không |

### Prior correction hoạt động thế nào

```text
p_c  = xác suất thô từ mô hình (softmax)
π_c  = tần suất lớp c trong tập train
α    = cường độ hiệu chỉnh, chọn tự động trên validation

q_c ∝ p_c / π_c^α
```

- `α = 0` → giữ nguyên output của mô hình.
- `α = 1` → loại bỏ hoàn toàn ảnh hưởng của prior.

Vì chia cho `π_c^α` (với `π_c < 1`) làm **tăng** xác suất của các lớp hiếm
và **giảm** của lớp đông, nên mô hình cuối cùng có xác suất cân bằng hơn mà
vẫn giữ được thứ tự xếp hạng giữa các lớp.

### Kết quả sau khi sửa

Đo thực tế trên toàn bộ test set (813 ảnh), 20 epoch MobileNetV2:

| Chỉ số | Trước | Sau |
|---|---|---|
| **Test accuracy** | ~0.21 | **0.9705** |
| **Balanced accuracy** | thấp, lớp hiếm recall ≈ 0 | **0.9651** |
| **Recall thấp nhất** (Hyundai Creta) | 0.00 | **0.91** |
| **Số ảnh dự đoán là Audi** | gần như toàn bộ | **199 / 813** (đúng tỉ lệ thật) |
| **Recall trung bình 7 lớp** | không ổn định | **0.97 (macro)** |

> Số liệu "trước" lấy từ log training cũ (`logs/train_full.log`): accuracy
> dao động quanh **0.20–0.22** và không hề tăng — dấu hiệu điển hình của model
> bị lớp đa số nuốt chửng.

Classification report sau khi sửa:

```text
                     precision    recall    f1-score   support
               Audi       0.99      0.99      0.99       199
      Hyundai Creta       0.91      0.91      0.91        67
      Mahindra Scorpio       0.93      0.99      0.95        75
        Rolls Royce       0.97      0.99      0.98        74
Maruti Suzuki Swift       0.97      0.94      0.96       102
        Tata Safari       0.97      0.96      0.97       106
      Toyota Innova       0.99      0.98      0.98       190
           accuracy                           0.97       813
      macro avg           0.96      0.97      0.96       813
   weighted avg           0.97      0.97      0.97       813
```

**Phân bố nhãn dự đoán** — bằng chứng rõ ràng nhất là model đã cân bằng:

```text
Audi=199  Hyundai Creta=67  Mahindra Scorpio=80  Rolls Royce=75
Maruti Suzuki Swift=99  Tata Safari=105  Toyota Innova=188
```

So với số ảnh thật (199 / 67 / 75 / 74 / 102 / 106 / 190) — model dự đoán gần
như **khớp hoàn toàn** với phân bố thật. Trước đây toàn bộ 813 ảnh đều bị
dồn về Audi.

> **Về prior correction:** với MobileNetV2 đã đạt 97% accuracy, hệ số `α`
> được chọn tự động là **0.00** — tức là mô hình đã tự cân bằng được nhờ
> class weights + transfer learning, không cần can thiệp thêm. Cơ chế prior
> correction vẫn được giữ lại và **tự động kích hoạt** nếu model yếu hơn
> (ví dụ khi dùng `--arch scratch` hoặc train ít epoch). Đây là lý do nên
> **luôn để script tự chọn α** thay vì hard-code.

> Con số chính xác nằm trong `outputs/car_cnn/metrics.json`
> (`test_accuracy`, `balanced_accuracy`,
> `predicted_label_distribution`) và trong
> `outputs/car_cnn/confusion_matrix.png`.

---

## 4. Kiến trúc model

### Mặc định — MobileNetV2 (transfer learning)

```text
Input (224, 224, 3)
  │
  ├─ MobileNetV2 (ImageNet weights, include_top=False)   → (7, 7, 1280)
  │    ├─ Stage 1: trainable = False   (đóng băng toàn bộ)
  │    └─ Stage 2: trainable = True    (mở băng, BatchNorm vẫn đóng băng)
  ├─ Dropout(0.30)                                        → (1280,)
  ├─ Dense(128, relu)                                     → (128,)
  ├─ Dropout(0.20)                                        → (128,)
  └─ Dense(7, softmax) "predictions"                      → (7,)

Total params: 2,422,855 (9.24 MB)
Trainable (stage 1):  164,871
Trainable (stage 2):  2,422,855 − BatchNorm
```

**Hai stage huấn luyện:**

| Stage | Nội dung | LR | Mặc định |
|---|---|---|---|
| 1 | Đóng băng backbone, chỉ train phần đầu | 1e-3 | 5 epoch |
| 2 | Mở băng toàn bộ backbone, fine-tune | 1e-4 | 15 epoch |

Trong stage 2, các lớp **BatchNorm vẫn được giữ đóng băng** — thống kê chạy
của chúng được ước lượng trên ImageNet, để chúng trôi trên 2.682 ảnh sẽ phá
hỏng hoàn toàn.

### Tuỳ chọn — `scratch` (CNN train tương đương notebook cũ)

```text
Input (256, 256, 3)
  ├─ Conv2D(32, 3×3, same) + BatchNorm → MaxPool(2) → Dropout(0.10)  → (128,128,32)
  ├─ Conv2D(64, 3×3, same) + BatchNorm → MaxPool(2) → Dropout(0.15)  → ( 64, 64,64)
  ├─ Conv2D(128,3×3, same) + BatchNorm → MaxPool(2) → Dropout(0.20)  → ( 32, 32,128)
  ├─ GlobalAveragePooling2D → Dropout(0.50)                         → (128,)
  ├─ Dense(64, relu) → Dropout(0.30)                                 → (64,)
  └─ Dense(7, softmax)                                                → (7,)

Total params: ~103K
```

Chạy bằng: `python train_cnn.py --mode train --arch scratch`

### Augmentation (chỉ áp dụng cho tập train)

| Phép biến đổi | Tham số | Vì sao |
|---|---|---|
| `RandomFlip("horizontal")` | — | Xe gần đối xứng, hợp lệ |
| `RandomRotation` | 0.10 rad (~6°) | Góc nghiêng nhẹ, không cắt mất đặc trưng |
| `RandomZoom` | 0.10 | Mô phỏng khung hình khác nhau |
| `RandomTranslation` | 0.05 | Xe không luôn ở giữa ảnh |
| `RandomBrightness` | 0.10 | Ánh sáng ngoài trời khác nhau |

Các phép biến đổi được giữ **nhẹ** có chủ đích: đặc trưng phân biệt 7 hãng xe
(lưới xe, logo, hình dáng đèn pha) nằm ở các chi tiết nhỏ — xoay/zoom mạnh
sẽ xoá mất chúng.

---

## 5. Cài đặt môi trường

### Yêu cầu

- **Python 3.10 – 3.12** (khuyến nghị 3.11)
- ~4 GB đĩa cho `.venv` + weights MobileNetV2
- Internet **chỉ cần lần đầu** để tải weights ImageNet (~9 MB, được cache tại
  `~/.keras/models/`)

### Các lệnh cài đặt

#### macOS / Linux

```bash
git clone https://github.com/ptnghia3502/PPR501-car-analysis.git
cd PPR501-car-analysis

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppr501-car-analysis --display-name "Python (PPR501 Car Analysis)"
```

#### Windows (PowerShell)

```powershell
git clone https://github.com/ptnghia3502/PPR501-car-analysis.git
cd PPR501-car-analysis

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppr501-car-analysis --display-name "Python (PPR501 Car Analysis)"
```

### Kiểm tra môi trường

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import gradio, sklearn, PIL; print('deps OK')"
```

### Lưu ý macOS

Nếu Matplotlib báo lỗi không ghi được cache, đặt biến môi trường:

```bash
export MPLCONFIGDIR=/tmp/mplcache
```

### Kiểm tra dataset

```bash
cd sources/CNN
python -c "
from pathlib import Path
for d in sorted(Path('../Dataset/Train').iterdir()):
    if d.is_dir():
        print(f'{d.name:<20} {len(list(d.iterdir()))}')
"
```

---

## 6. Cách chạy (terminal)

> **Luôn `cd` vào `sources/CNN`** trước khi chạy — script tự tính đường dẫn
> tương đối tới dataset.

### 6.1 Kiểm tra nhanh (vài giây)

```bash
cd sources/CNN
python train_cnn.py --mode train --smoke --epochs 2 --freeze-epochs 1
```

Chỉ dùng vài batch, chạy xong là biết pipeline còn hoạt động.

### 6.2 Train đầy đủ + đánh giá

```bash
cd sources/CNN
python train_cnn.py --mode train --epochs 20 --freeze-epochs 5
```

Sau khi chạy xong, terminal sẽ in:

```text
Classification report
                     precision    recall    f1-score   support
               Audi       0.99      0.99      0.99       199
      Hyundai Creta       0.91      0.91      0.91        67
      Mahindra Scorpio       0.93      0.99      0.95        75
        Rolls Royce       0.97      0.99      0.98        74
Maruti Suzuki Swift       0.97      0.94      0.96       102
        Tata Safari       0.97      0.96      0.97       106
      Toyota Innova       0.99      0.98      0.98       190

           accuracy                           0.97       813
          macro avg           0.96      0.97      0.96       813
       weighted avg           0.97      0.97      0.97       813

[16:23:21] Test loss: 0.0994 | Test accuracy: 0.9705 | Balanced accuracy: 0.9651
[16:23:21] Predicted label distribution: Audi=199, Hyundai Creta=67,
           Mahindra Scorpio=80, Rolls Royce=75, Maruti Suzuki Swift=99,
           Tata Safari=105, Toyota Innova=188
```

Nếu muốn lưu lại log, tự pipe ra file:

```bash
mkdir -p logs
python train_cnn.py --mode train 2>&1 | tee logs/train_cnn_full.log
```

> **Thời gian chạy (CPU, MacBook):** khoảng **11 phút** cho 20 epoch
> (~13 giây/epoch, 84 batch/epoch). Trên GPU: vài giây/epoch.

### 6.3 Đánh giá lại model đã lưu

```bash
cd sources/CNN
python train_cnn.py --mode evaluate
```

In classification report (precision/recall/F1 từng lớp), test loss, test
accuracy, **balanced accuracy**, phân bố nhãn dự đoán, và vẽ lại confusion
matrix.

### 6.4 Dự đoán một ảnh

```bash
cd sources/CNN
python train_cnn.py --mode predict --image "../Dataset/Test/Tata Safari/116.jpg"
```

```text
Probabilities:
  Mahindra Scorpio        50.76%
  Tata Safari             48.77%
  Hyundai Creta            0.38%
  Maruti Suzuki Swift      0.05%
  Toyota Innova            0.04%
  Audi                     0.00%
  Rolls Royce              0.00%

Predicted class: Mahindra Scorpio  (50.76%)
```

> Lưu ý đường dẫn có khoảng trắng → **phải bọc trong dấu nháy kép**.
> Ảnh này là trường hợp khó: Tata Safari và Mahindra Scorpio là 2 SUV 7 chỗ
> gần như giống nhau về ngoại hình, nên model phân biệt chỉ được 50/50 —
> vẫn tốt hơn hẳn so với việc mọi ảnh đều ra Audi.

### 6.5 Mở giao diện web Gradio

```bash
cd sources/CNN
python train_cnn.py --mode serve
```

Mở trình duyệt tại **http://localhost:7860**.

| Tuỳ chọn | Tác dụng |
|---|---|
| `--port 7861` | Đổi cổng |
| `--share` | Tạo link public tạm thời (cần mạng) |

> ⚠️ **Không upload ảnh riêng tư khi bật `--share`** — ảnh sẽ đi qua server
> public của Gradio.

Dừng server: `Ctrl+C`.

**Chạy nền (đóng terminal vẫn chạy):**

```bash
nohup python train_cnn.py --mode serve --port 7860 > logs/serve.log 2>&1 < /dev/null &
disown
tail -f logs/serve.log
```

Dừng: `pkill -f "train_cnn.py --mode serve"`

### 6.6 Tổng hợp một lệnh cho người mới

```bash
# Từ thư mục gốc repo, làm đúng 3 bước:
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

cd sources/CNN
python train_cnn.py --mode train --epochs 20 && python train_cnn.py --mode serve
```

---

## 7. Cách chạy bằng Jupyter Notebook

`main.ipynb` là **wrapper mỏng** quanh `train_cnn.py` — nó `import` script đó
rồi gọi lại đúng hàm, nên kết quả luôn **giống hệt** khi chạy bằng terminal.

```bash
cd sources/CNN
jupyter lab main.ipynb
```

Trong JupyterLab:

1. Chọn kernel **Python (PPR501 Car Analysis)** (đã cài ở bước 5).
2. Chạy các cell theo thứ tự từ trên xuống (hoặc **Run All**).
3. Cell cuối cùng có sẵn dòng `demo.launch(...)` — bỏ comment để mở giao diện.

> **Lưu ý:** notebook cần train thật (không có `--smoke`), mất khoảng
> 30–40 phút trên CPU cho 20 epoch MobileNetV2.

### Notebook làm gì

| Cell | Nội dung |
|---|---|
| 1 | Import `train_cnn.py` bằng `importlib` + kiểm tra dataset |
| 2 | Bảng số ảnh mỗi lớp + bảng class weights |
| 3 | Chia dữ liệu, kiểm tra pixel range, vẽ ảnh augmented |
| 4 | Dựng model (`ARCH = "mobilenetv2"`) |
| 5 | Train 2 stage, giữ checkpoint tốt nhất |
| 6 | Chọn hệ số prior correction `α` trên validation |
| 7 | Lưu model + đánh giá test set |
| 8 | Vẽ biểu đồ loss / accuracy |
| 9 | Dự đoán một ảnh mẫu |
| 10 | Giao diện Gradio |

---

## 8. Output sinh ra

Tất cả nằm trong `sources/CNN/outputs/car_cnn/`:

| File | Nội dung |
|---|---|
| `model.keras` | Model đã train (kiến trúc + trọng số của epoch tốt nhất) |
| `labels.json` | Thứ tự lớp — dùng để map output số → tên |
| `meta.json` | Kiến trúc, image size, prior của mỗi lớp, hệ số `α` đã chọn |
| `history.json` | Lịch sử loss/accuracy mỗi epoch |
| `metrics.json` | Test loss, accuracy, balanced accuracy, confusion matrix, phân bố nhãn dự đoán |
| `confusion_matrix.png` | Ma trận nhầm lẫn (heat-map) |
| `training_curves.png` | Biểu đồ accuracy & loss train vs validation |

Log training không được tự sinh — nếu cần thì pipe stdout ra file
(xem [mục 5](#5-chạy-train)).

> Thư mục `outputs/` và `logs/` được `.gitignore` — model được train lại từ
> dataset mỗi lần, không commit binary lên git.

---

## 9. Đọc kết quả & đánh giá

### Các chỉ số trong `metrics.json`

| Key | Ý nghĩa |
|---|---|
| `test_accuracy` | Tỉ lệ dự đoán đúng trên test set |
| `balanced_accuracy` | **Trung bình recall của 7 lớp** — chỉ số quan trọng nhất khi dữ liệu mất cân bằng |
| `test_loss` | Cross-entropy trên test set (sau prior correction) |
| `best_epoch` | Epoch có validation accuracy cao nhất |
| `predicted_label_distribution` | Số ảnh dự đoán thuộc mỗi lớp — **kiểm tra ngay model có bị dồn lớp không** |
| `confusion_matrix` | Ma trận 7×7 (hàng = thật, cột = dự đoán) |
| `prior_alpha` | Hệ số prior correction đã chọn |
| `num_parameters` | Tổng số tham số của model |

### Đọc `training_curves.png`

- **`loss` giảm đều** → model đang học.
- **`val_loss` giảm rồi tăng** → bắt đầu **overfitting**; đó là lúc EarlyStopping dừng.
- **Gap giữa `accuracy` và `val_accuracy` lớn** → model ghi nhớ tập train.
- **Stage 1 → stage 2**: đường val accuracy thường đi lên nhẹ ở đầu stage 2 rồi bão hoà.

### Ma trận nhầm lẫn

- Đường chéo càng sáng → model càng tốt.
- Ô tối nằm **ngoài** đường chéo cho biết model hay nhầm xe nào với xe nào
  (ví dụ: Hyundai Creta ↔ Maruti Suzuki Swift — cùng là SUV hatchback cỡ nhỏ).
- Nếu một **cột** bị tối bất thường → model đang thiên về lớp đó → tăng `α`.

### Đánh giá nhanh chất lượng cân bằng

```bash
cd sources/CNN
python -c "
import json
m = json.load(open('outputs/car_cnn/metrics.json'))
print('accuracy          :', round(m['test_accuracy'], 4))
print('balanced accuracy :', round(m['balanced_accuracy'], 4))
print('predicted counts  :', m['predicted_label_distribution'])
"
```

---

## 10. Tuỳ chọn CLI đầy đủ

```bash
python train_cnn.py --help
```

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--mode {train,evaluate,predict,serve}` | `train` | Chế độ chạy |
| `--arch {mobilenetv2,scratch}` | `mobilenetv2` | Kiến trúc model |
| `--image-size N` | 224 / 256 | Kích thước ảnh đầu vào (theo arch) |
| `--epochs N` | 20 | Tổng số epoch (cả 2 stage) |
| `--freeze-epochs N` | 5 | Số epoch đóng băng backbone (chỉ mobilenetv2) |
| `--fine-tune-lr F` | 1e-4 | Learning rate của stage 2 |
| `--learning-rate F` | 1e-3 | Learning rate của stage 1 |
| `--batch-size N` | 32 | Kích thước batch |
| `--patience N` | 6 | Số epoch chờ `val_accuracy` không tăng trước khi dừng |
| `--validation-split F` | 0.2 | Tỉ lệ tách validation từ `Dataset/Train` |
| `--class-weight {effective,balanced,none}` | `effective` | Chiến lược cân bằng lớp khi train |
| `--seed N` | 42 | Seed để tái lập kết quả |
| `--image PATH` | — | Đường dẫn ảnh (dùng với `--mode predict`) |
| `--port N` | 7860 | Cổng Gradio |
| `--share` | off | Tạo link public Gradio |
| `--smoke` | off | Chạy thử siêu nhanh (vài batch) |

### Ví dụ kết hợp

```bash
# Train dài hơn, kiên nhẫn hơn
python train_cnn.py --mode train --epochs 30 --freeze-epochs 8 --patience 8

# So sánh với CNN train từ đầu
python train_cnn.py --mode train --arch scratch --epochs 30

# Dùng class_weight mạnh (balanced)
python train_cnn.py --mode train --class-weight balanced

# Không dùng class_weight (để đo baseline)
python train_cnn.py --mode train --class-weight none
```

---

## 11. Troubleshooting

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `FileNotFoundError: Training folder not found` | Chạy sai thư mục | `cd sources/CNN` trước khi chạy |
| `ModuleNotFoundError: No module named 'tensorflow'` | Chưa kích hoạt `.venv` | `source .venv/bin/activate` rồi `pip install -r requirements.txt` |
| `CERTIFICATE_VERIFY_FAILED` khi tải weights | CA store của Python bị hỏng (hay gặp trên macOS) | Script đã tự dùng `certifi`; nếu vẫn lỗi: `export SSL_CERT_FILE=$(python -m certifi)` |
| `AttributeError: module 'gradio' has no attribute 'Bar'` | Gradio ≥ 5 đã xoá `gr.Bar` | Dùng `gr.BarPlot` (script đã xử lý) |
| Matplotlib lỗi ghi cache (macOS) | Sandbox ghi `/Users/.../.config` bị chặn | `export MPLCONFIGDIR=/tmp/mplcache` |
| Kernel Jupyter không import được TensorFlow | Chọn sai kernel | Chọn kernel **Python (PPR501 Car Analysis)** |
| Model vẫn dồn về một lớp | Thiếu dữ liệu / LR quá cao | Tăng `--epochs`, giảm `--learning-rate`, thử `--class-weight balanced` |
| `UnicodeEncodeError` khi in tên lớp | Console Windows cp1252 | `set PYTHONIOENCODING=utf-8` |
| Gradio không mở được | Cổng bị chiếm | `python train_cnn.py --mode serve --port 7861` |

### Kiểm tra chẩn đoán nhanh

```bash
# Số ảnh mỗi lớp
python - <<'PY'
from pathlib import Path
for d in sorted(Path("../Dataset/Train").iterdir()):
    if d.is_dir():
        print(f"{d.name:<20} {len(list(d.iterdir()))}")
PY

# Thứ tự lớp Keras suy ra (phải khớp labels.json)
python - <<'PY'
import tensorflow as tf
ds = tf.keras.utils.image_dataset_from_directory("../Dataset/Train", labels="inferred")
print(ds.class_names)
PY
```

---

## 12. Cấu trúc thư mục

```text
sources/CNN/
├── README.md              # file này
├── train_cnn.py           # nguồn sự thật duy nhất — CLI + train + eval + predict + serve
├── main.ipynb             # wrapper mỏng quanh train_cnn.py
├── logs/                  # tự tạo khi bạn pipe output (tee) — không bắt buộc
└── outputs/               # tự sinh khi train, không commit
    └── car_cnn/
        ├── model.keras           # trọng số cuối cùng (dùng cho serve)
        ├── labels.json           # thứ tự lớp
        ├── meta.json             # arch, image_size, prior_alpha, priors
        ├── history.json          # loss/accuracy theo từng epoch
        ├── metrics.json          # kết quả đánh giá trên tập test
        ├── confusion_matrix.png
        └── training_curves.png
```

> **Quy ước:** `train_cnn.py` là nguồn sự thật. Notebook chỉ import và gọi lại
> các hàm từ đó. Nếu muốn sửa logic, sửa `train_cnn.py`; không copy-paste logic
> vào notebook — đó là cách hai bên lệch nhau.

---

## Nguồn tham khảo

- [Kaggle — Car images classification (dataset)](https://www.kaggle.com/code/kshitij192/car-images-classification-using-cnn/data)
- [Cui et al. 2019 — Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)
- [Menon et al. 2021 — Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314)
- [He et al. 2016 — Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [Sandler et al. 2018 — MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Source Code — github.com/ptnghia3502/PPR501-car-analysis](https://github.com/ptnghia3502/PPR501-car-analysis)

---

## Tác giả

- **Giảng viên:** Nguyễn Xuân Sâm
- **Thành viên:**
  - Phạm Trọng Nghĩa — 26MS23417
  - Trần Công Vinh — 26MS23406
  - Trần Quang Duy — 26MS23410
