# Hướng dẫn chạy Transformer Chatbot trên máy mới

Tài liệu này hướng dẫn cài đặt và chạy hai thành phần:

- `sources/Transformer/chatbot.ipynb`: notebook giải thích, kiểm tra pipeline xử lý dữ liệu, tìm kiếm Hybrid, Linear Regression và CNN.
- `sources/Transformer/app.py`: ứng dụng Streamlit gồm trang tìm xe và trang nhận diện ảnh.

Các lệnh trong tài liệu mặc định được chạy từ **thư mục gốc repository**, tức thư mục chứa `requirements.txt` và `sources/`.

---

## 1. Yêu cầu máy

### Bắt buộc

- Windows 10/11, macOS hoặc Linux.
- Git.
- Python 3.10–3.12; **khuyến nghị Python 3.11 64-bit**.
- Kết nối Internet trong lần cài thư viện và lần đầu tải MiniLM từ Hugging Face.
- Ít nhất khoảng 8 GB dung lượng trống cho repository, môi trường Python và cache model.
- Khuyến nghị từ 8 GB RAM.

### Cài công cụ trên Windows

1. Cài Git: <https://git-scm.com/download/win>
2. Cài Python 3.11: <https://www.python.org/downloads/>
3. Khi cài Python, bật tùy chọn **Add Python to PATH** và cài Python Launcher `py`.
4. Mở PowerShell mới và kiểm tra:

```powershell
git --version
py -3.11 --version
```

Nếu TensorFlow báo lỗi thiếu DLL trên Windows, cài **Microsoft Visual C++ Redistributable 2015–2022 (x64)** từ trang Microsoft rồi mở lại terminal.

> TensorFlow từ phiên bản 2.11 không hỗ trợ GPU trên Windows native. CNN vẫn chạy được bằng CPU. Muốn TensorFlow dùng GPU cần WSL2; đây là tùy chọn, không bắt buộc để chạy project.

---

## 2. Tải source code

Mở PowerShell hoặc Terminal tại thư mục muốn lưu project:

```powershell
git clone https://github.com/ptnghia3502/PPR501-car-analysis.git
cd PPR501-car-analysis
```

Kiểm tra hai file cần chạy:

```powershell
Test-Path .\sources\Transformer\chatbot.ipynb
Test-Path .\sources\Transformer\app.py
```

Cả hai lệnh phải trả về `True`.

> Nếu `chatbot.ipynb` trả về `False`, phiên bản trên remote chưa chứa notebook mới. Cần commit/push notebook từ máy phát triển hoặc sao chép notebook vào đúng `sources/Transformer/` trước khi tiếp tục.

### macOS/Linux

```bash
git clone https://github.com/ptnghia3502/PPR501-car-analysis.git
cd PPR501-car-analysis
test -f sources/Transformer/chatbot.ipynb && echo "notebook OK"
test -f sources/Transformer/app.py && echo "app OK"
```

---

## 3. Tạo môi trường Python riêng

Không cài thư viện trực tiếp vào Python hệ thống. Tạo một `.venv` ở thư mục gốc repository.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r .\requirements.txt
python -m ipykernel install --user --name ppr501-car-analysis --display-name "Python (PPR501 Car Analysis)"
```

Nếu PowerShell chặn `Activate.ps1`, chỉ cấp quyền cho phiên terminal hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Hoặc không activate, gọi Python trong `.venv` trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppr501-car-analysis --display-name "Python (PPR501 Car Analysis)"
```

Mỗi lần mở terminal mới, cần kích hoạt lại môi trường:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

---

## 4. Kiểm tra thư viện

Sau khi kích hoạt `.venv`, chạy:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import streamlit, tensorflow, sentence_transformers, sklearn, pandas, PIL; print('Dependencies OK')"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Kết quả `CUDA: False` không phải lỗi. Khi không có PyTorch CUDA, MiniLM tự chạy bằng CPU.

Nếu muốn chỉ định thiết bị cho MiniLM:

```powershell
# Windows PowerShell — CPU
$env:CAR_EMBEDDING_DEVICE = "cpu"

# Chỉ dùng khi PyTorch CUDA đã được cài và torch.cuda.is_available() là True
$env:CAR_EMBEDDING_DEVICE = "cuda"
```

macOS/Linux dùng:

```bash
export CAR_EMBEDDING_DEVICE=cpu
```

Không cần đặt biến này trong trường hợp bình thường; app tự chọn CUDA nếu PyTorch nhìn thấy GPU, nếu không sẽ chọn CPU.

---

## 5. Kiểm tra dữ liệu và artifact model

App cần các file sau:

```text
sources/
├── Dataset/
│   └── car-price.csv
├── CNN/
│   └── outputs/
│       └── car_cnn/
│           ├── model.keras
│           ├── labels.json
│           ├── meta.json
│           └── metrics.json
└── Transformer/
    ├── chatbot.ipynb
    └── app.py
```

Kiểm tra trên Windows:

```powershell
$required = @(
    ".\sources\Dataset\car-price.csv",
    ".\sources\CNN\outputs\car_cnn\model.keras",
    ".\sources\CNN\outputs\car_cnn\labels.json",
    ".\sources\CNN\outputs\car_cnn\meta.json",
    ".\sources\Transformer\chatbot.ipynb",
    ".\sources\Transformer\app.py"
)
$required | ForEach-Object { "$(Test-Path $_)`t$_" }
```

Tất cả dòng phải bắt đầu bằng `True`.

`model.keras` đã được lưu trong repository nên máy mới **không cần train lại CNN**. Chỉ khi artifact bị thiếu và có đầy đủ thư mục ảnh `Dataset/Train` cùng `Dataset/Test`, mới train lại bằng:

```powershell
python .\sources\CNN\train_cnn.py --mode train --epochs 20 --freeze-epochs 5
```

Việc train đầy đủ có thể mất nhiều thời gian trên CPU.

---

## 6. Chạy `chatbot.ipynb`

### Cách 1: JupyterLab

Từ thư mục gốc repository, sau khi kích hoạt `.venv`:

```powershell
python -m jupyter lab .\sources\Transformer\chatbot.ipynb
```

macOS/Linux:

```bash
python -m jupyter lab sources/Transformer/chatbot.ipynb
```

Sau khi trình duyệt mở:

1. Chọn kernel **Python (PPR501 Car Analysis)**.
2. Chọn **Kernel → Restart Kernel and Clear Outputs** nếu notebook còn output từ máy cũ.
3. Chạy cell từ trên xuống hoặc dùng **Run All**.
4. Lần đầu tải MiniLM có thể mất vài phút tùy mạng.

Notebook tự tìm `app.py` từ workspace hoặc repository root, vì vậy không cần sửa các đường dẫn tuyệt đối cũ xuất hiện trong output đã lưu. Sau khi chạy lại cell, đường dẫn sẽ được tính theo máy mới.

### Cách 2: Mở bằng Cursor/VS Code

1. Mở thư mục gốc repository trong Cursor/VS Code.
2. Mở `sources/Transformer/chatbot.ipynb`.
3. Bấm **Select Kernel**.
4. Chọn **Python (PPR501 Car Analysis)** hoặc chọn interpreter `.venv`:
   - Windows: `.venv\Scripts\python.exe`
   - macOS/Linux: `.venv/bin/python`
5. Chạy từng cell từ trên xuống.

### Cell mở Streamlit trong mục 11

Notebook chủ yếu dùng để phân tích và kiểm tra model. Cách ổn định nhất là giữ app chạy bằng terminal riêng theo mục 8 bên dưới.

Nếu dùng cell ở mục 11:

- Đặt `RUN_STREAMLIT = True` để notebook mở `app.py`.
- Không chạy đồng thời một Streamlit khác trên cùng cổng.
- Khi kết thúc, đặt `STOP_STREAMLIT = True` và chạy cell dừng.
- Nếu chỉ muốn **Run All** để xem phân tích mà không mở app, đặt `RUN_STREAMLIT = False` và `STOP_STREAMLIT = False` trước khi chạy toàn bộ.

---

## 7. Chuẩn bị cache trước khi chạy app

Từ thư mục gốc repository và trong `.venv`:

```powershell
python .\sources\Transformer\app.py --prepare
```

macOS/Linux:

```bash
python sources/Transformer/app.py --prepare
```

Lệnh này thực hiện:

1. Đọc `car-price.csv`.
2. Tạo chỉ mục TF-IDF.
3. Tải hoặc đọc MiniLM `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. Tạo/kiểm tra cache embedding tại `sources/Transformer/.cache/`.
5. Load thử `model.keras` và chạy một lần dự đoán CNN.

Kết quả thành công có dạng:

```text
Dataset: ... dòng (...s)
TF-IDF sẵn sàng (...s)
Embedding sẵn sàng: (..., 384) -> ...\.cache\embeddings-....npy (...s)
CNN sẵn sàng: model.keras, 7 lớp (...s)
Chuẩn bị xong (...s).
```

Lần đầu cần Internet để tải MiniLM. Các lần sau model và embedding được lấy từ cache local.

> `--prepare` chạy trong process riêng nên cache trên đĩa được giữ lại, còn app Streamlit vẫn warm-up dataset, TF-IDF, MiniLM và CNN một lần trong process của app. Màn hình tiến trình ban đầu là hành vi bình thường.

---

## 8. Chạy `app.py`

Từ thư mục gốc repository, sau khi `--prepare` thành công:

```powershell
python -m streamlit run .\sources\Transformer\app.py --server.port 8502
```

macOS/Linux:

```bash
python -m streamlit run sources/Transformer/app.py --server.port 8502
```

Mở trình duyệt tại:

- <http://localhost:8502>

App gồm:

- **Tìm xe**: tìm kiếm Hybrid, Semantic hoặc TF-IDF từ dữ liệu CSV.
- **Nhận diện ảnh**: MobileNetV2 nhận diện một trong 7 lớp xe đã học.
- **Xem phân tích giá (Linear Regression)**: chỉ vẽ biểu đồ khi bấm nút, không làm chậm câu chat đầu.

Ở lần mở đầu tiên của process Streamlit, chờ màn hình tiến trình hoàn tất. Khi mở tab thứ hai trong cùng process, tài nguyên đã nằm trong RAM nên không warm-up lại.

### Chạy chỉ bằng TF-IDF khi chưa có mạng/model MiniLM

App tự fallback sang TF-IDF nếu MiniLM không tải được. Trong sidebar sẽ hiện trạng thái tương ứng. Tìm xe vẫn hoạt động, nhưng Semantic/Hybrid sẽ không có embedding ngữ nghĩa.

---

## 9. Dừng app

Trong terminal đang chạy Streamlit, bấm:

```text
Ctrl+C
```

Nếu Windows đã đóng cổng nhưng terminal vẫn đứng ở `Stopping...` do TensorFlow chưa thoát, mở một PowerShell khác và chạy:

```powershell
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*streamlit*Transformer*app.py*" } |
    ForEach-Object { taskkill /PID $_.ProcessId /T /F }
```

Kiểm tra cổng 8502:

```powershell
Get-NetTCPConnection -LocalPort 8502 -ErrorAction SilentlyContinue
```

Nếu lệnh không trả kết quả thì cổng đã trống.

macOS/Linux:

```bash
pkill -f "streamlit run.*sources/Transformer/app.py"
```

---

## 10. Chạy lại vào những lần sau

Sau lần cài đầu tiên, không cần cài lại thư viện và thường không cần chạy lại `--prepare` nếu dataset/model không đổi.

### Windows

```powershell
cd đường-dẫn-tới\PPR501-car-analysis
.\.venv\Scripts\Activate.ps1
python -m streamlit run .\sources\Transformer\app.py --server.port 8502
```

### macOS/Linux

```bash
cd /duong-dan-toi/PPR501-car-analysis
source .venv/bin/activate
python -m streamlit run sources/Transformer/app.py --server.port 8502
```

Chạy lại `--prepare` khi:

- `car-price.csv` thay đổi.
- Đổi `CAR_EMBEDDING_MODEL`.
- Xóa thư mục `sources/Transformer/.cache/`.
- Thay `model.keras` hoặc metadata CNN.
- Muốn kiểm tra đầy đủ artifact trước khi demo.

---

## 11. Cập nhật source code

Trước khi cập nhật, dừng Streamlit và đóng kernel notebook. Sau đó:

```powershell
git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python .\sources\Transformer\app.py --prepare
```

Nếu code mới thay đổi cách tạo embedding, app sẽ dùng fingerprint mới và tạo file cache mới, không cần tự sửa file `.npy`.

---

## 12. Xử lý lỗi thường gặp

### `ModuleNotFoundError`

Nguyên nhân: chưa kích hoạt đúng `.venv` hoặc kernel dùng sai Python.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python -c "import sys; print(sys.executable)"
```

Trong notebook, chọn kernel **Python (PPR501 Car Analysis)**.

### `FileNotFoundError: car-price.csv`

Kiểm tra:

```powershell
Test-Path .\sources\Dataset\car-price.csv
```

Không đổi cấu trúc `sources/Dataset/` và luôn chạy lệnh từ thư mục gốc repository.

### Thiếu `model.keras`

Kiểm tra:

```powershell
Test-Path .\sources\CNN\outputs\car_cnn\model.keras
```

Nếu file thiếu, chạy `git status`, `git pull` hoặc clone lại repository. Chỉ train CNN lại khi repository thật sự không cung cấp artifact.

### Hugging Face lỗi mạng / `WinError 10054`

- Đảm bảo Internet hoạt động trong lần tải MiniLM đầu tiên.
- Thử chạy lại `python sources/Transformer/app.py --prepare`.
- Tắt VPN/proxy hoặc cho phép Python qua firewall nếu kết nối bị reset.
- Sau khi model đã cache, app ưu tiên `local_files_only=True` và có thể khởi động không cần mạng.
- Nếu MiniLM vẫn không tải được, app tự dùng TF-IDF.

### TensorFlow báo không dùng GPU trên Windows

Đây là cảnh báo, không phải lỗi. TensorFlow >= 2.11 trên Windows native chạy CPU. CNN vẫn hoạt động bình thường.

### `tf.reset_default_graph is deprecated`

Đây là cảnh báo tương thích bên trong Keras/TensorFlow, không làm hỏng dự đoán.

### Cổng 8502 đã được sử dụng

Tìm process giữ cổng:

```powershell
Get-NetTCPConnection -LocalPort 8502 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess
```

Dừng process hoặc chạy app trên cổng khác:

```powershell
python -m streamlit run .\sources\Transformer\app.py --server.port 8503
```

### Xóa và tạo lại cache embedding

Chỉ thực hiện khi cache hỏng hoặc cần tạo lại hoàn toàn:

```powershell
Remove-Item .\sources\Transformer\.cache -Recurse -Force -ErrorAction SilentlyContinue
python .\sources\Transformer\app.py --prepare
```

---

## 13. Checklist trước khi demo

- [ ] Đang dùng đúng Python trong `.venv`.
- [ ] `car-price.csv` tồn tại.
- [ ] `model.keras`, `labels.json` và `meta.json` tồn tại.
- [ ] `python sources/Transformer/app.py --prepare` chạy thành công.
- [ ] Streamlit mở được tại `http://localhost:8502`.
- [ ] Sidebar báo MiniLM đã tải hoặc báo fallback TF-IDF rõ ràng.
- [ ] Tìm thử: `Toyota 7 chỗ dưới 900 triệu`.
- [ ] Bấm **Xem phân tích giá (Linear Regression)** và thấy biểu đồ.
- [ ] Trang **Nhận diện ảnh** báo MobileNetV2 sẵn sàng.
- [ ] Upload một ảnh thuộc một trong 7 lớp và thấy top-3 dự đoán.

Sau khi hoàn tất checklist, máy mới đã sẵn sàng chạy cả notebook lẫn ứng dụng Streamlit.
