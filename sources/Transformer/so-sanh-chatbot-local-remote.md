# So sánh chatbot Transformer local và remote repository

## 1. Phạm vi và mốc đối chiếu

Tài liệu này so sánh:

- **Bản local:** `sources/Transformer/chatbot.ipynb`, đồng thời đối chiếu các hàm triển khai tương ứng trong `sources/Transformer/app.py`.
- **Bản remote:** [`sources/Transformer/main.ipynb`](https://github.com/ptnghia3502/PPR501-car-analysis/blob/0e09a00daa066b5082f1a250b8d22d9a01672d60/sources/Transformer/main.ipynb) và [`sources/Transformer/app.py`](https://github.com/ptnghia3502/PPR501-car-analysis/blob/0e09a00daa066b5082f1a250b8d22d9a01672d60/sources/Transformer/app.py) trên nhánh `main` của repository `ptnghia3502/PPR501-car-analysis`.
- **Commit remote được dùng:** `0e09a00daa066b5082f1a250b8d22d9a01672d60` — `Update bai 3`, ngày `2026-11-01T16:06:53+07:00`.
- **Ngày đối chiếu:** 03/11/2026.

Bản `sources/Transformer/main.ipynb` trong working tree local đang có thay đổi riêng, vì vậy tài liệu **không dùng file local cùng tên làm đại diện cho remote**. Nội dung remote được đọc trực tiếp từ `origin/main` tại commit nêu trên.

### Dấu vân tay để tái kiểm tra

| Thành phần | SHA-256 tại thời điểm so sánh |
| --- | --- |
| Local `chatbot.ipynb` | `3ac3b698bc71d7c4bcacebaccf79d6a59110eee21ad6b7826eeaaf9847fdb67b` |
| Remote `main.ipynb` | `5407e6fd0ea9f867486c796e67a326bc47c6274fa03cb0a0f775894a4cc3104d` |
| Remote `app.py` | `2fe6dab78b3b1037168572c943da928f4176fb91b24fca95b06897009fbd2d3d` |

Các hash trên chỉ có ý nghĩa nhận diện đúng phiên bản đã so sánh. Khi file thay đổi, cần chạy lại phép đối chiếu.

---

## 2. Kết luận ngắn

Hai bản có cùng mục tiêu tổng quát: nhận câu hỏi tiếng Việt, lọc tin đăng ô tô, xếp hạng kết quả, vẽ Linear Regression và nhận diện ảnh. **Cả hai đều là chatbot truy xuất, không phải chatbot sinh văn bản:** Qwen hoặc MiniLM chỉ tạo embedding để xếp hạng tin đăng; không có decoder hay Instruction LLM viết câu trả lời. Tuy nhiên, kiến trúc đã khác nhau đáng kể:

1. **Remote là bản demo gọn, phụ thuộc hoàn toàn vào Qwen cho xếp hạng ngữ nghĩa.** Bản này dùng `Qwen/Qwen3-Embedding-0.6B`, mã hóa lại tập ứng viên theo từng truy vấn và chỉ xếp hạng bằng cosine similarity.
2. **Local là hệ thống truy xuất nhiều tầng.** Bản này mặc định dùng multilingual MiniLM, lập chỉ mục toàn bộ dataset, hỗ trợ `Hybrid`, `Semantic`, `TF-IDF`, có cache embedding `.npy` và tự hạ xuống TF-IDF nếu model semantic lỗi.
3. **Local hiểu nhiều tiêu chí hơn và giải thích từng kết quả rõ hơn.** Ngoài giá, năm, hãng, kiểu dáng và số chỗ, local còn xử lý km, nhiên liệu, hộp số, xuất xứ, tình trạng, số cửa, màu và địa điểm; mỗi xe có trạng thái `Khớp`, `Khác`, hoặc `Chưa rõ`.
4. **CNN remote được huấn luyện trực tiếp trong notebook/app từ đầu.** Đáng chú ý, remote dùng `Test` làm `validation_data` rồi lại đánh giá trên chính tập đó, gây leakage khi chọn epoch.
5. **CNN local dùng MobileNetV2 transfer learning và artifact đã huấn luyện riêng.** App chỉ suy luận, không train khi người dùng mở giao diện; notebook đọc metrics, confusion matrix và lịch sử huấn luyện đã lưu.
6. **Local có đánh giá hồi quy nghiêm túc hơn trong notebook.** Ngoài đường hồi quy mô tả trong app, notebook có holdout theo nhóm, baseline trung bình và GroupKFold 5-fold. Remote chỉ fit và tính $R^2$ trên toàn bộ các điểm đang hiển thị.
7. **Local phù hợp hơn cho demo ổn định và phân tích học thuật**, nhưng phức tạp hơn, có nguy cơ trùng logic giữa notebook và `app.py`, chưa có bộ nhãn relevance để đánh giá retrieval tổng quát, và còn một số điểm cần sửa được nêu ở cuối tài liệu.

Không thể kết luận MiniLM tốt hơn Qwen hoặc ngược lại chỉ từ mã nguồn. Muốn chọn model theo chất lượng, cần một tập câu hỏi có nhãn và các metric như Precision@K, Recall@K, MRR hoặc nDCG.

---

## 3. Bảng so sánh tổng quan

| Hạng mục | Remote `main.ipynb` / `app.py` | Local `chatbot.ipynb` / `app.py` | Nhận xét |
| --- | --- | --- | --- |
| Embedding mặc định | `Qwen/Qwen3-Embedding-0.6B` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Qwen lớn hơn; MiniLM nhẹ hơn |
| Thiết bị embedding | Tự chọn `cuda` nếu `torch.cuda.is_available()`, ngược lại CPU | Có thể đặt `CAR_EMBEDDING_DEVICE`; nếu không đặt thì để `SentenceTransformer` tự chọn | Local cấu hình được bằng môi trường, remote thể hiện lựa chọn rõ trong code |
| Chiều embedding mặc định | Qwen 0.6B mặc định 1024 chiều | MiniLM 384 chiều, đã được notebook kiểm tra | Local cần ít RAM hơn cho chỉ mục |
| Prompt khi encode query | Luôn dùng `prompt_name="query"` | Chỉ dùng query prompt nếu model khai báo prompt `query`; MiniLM encode trực tiếp | Local cho phép đổi sang Qwen mà vẫn tương thích prompt |
| Phương pháp retrieval | Semantic cosine | Hybrid, Semantic, TF-IDF | Local linh hoạt và có đường dự phòng |
| Cache embedding | `st.cache_data` theo từng tuple batch ứng viên; chủ yếu trong runtime | Cache resource trong phiên và file `.npy` cho toàn dataset | Local tái sử dụng qua lần chạy app |
| Phạm vi encode | Chỉ ứng viên còn lại sau hard filter, theo từng câu hỏi | Toàn dataset một lần, rồi lấy các dòng ứng viên theo `_row_id` | Remote có thể nhanh khi tập lọc rất nhỏ; local ổn định hơn qua nhiều truy vấn |
| Fallback khi model lỗi | Không có retrieval thay thế | Tự chuyển sang TF-IDF và ghi cảnh báo | Local vẫn tìm được xe khi offline/model lỗi |
| Tiêu chí lọc | Giá, một năm chính xác, số chỗ, hãng, một số kiểu dáng | Giá, khoảng/cận năm, hãng, kiểu dáng, chỗ, cửa, km, nhiên liệu, hộp số, xuất xứ, tình trạng, màu, địa điểm | Local bao phủ nhiều ngôn ngữ truy vấn hơn |
| Nới điều kiện | Bỏ giá, rồi năm, rồi chỗ/kiểu dáng; cố giữ hãng | Năm giai đoạn tích lũy, giữ hãng; ghi rõ tiêu chí đã nới | Local giải thích được quá trình fallback |
| Đối chiếu từng xe | Không | `matched` / `unmatched` / `unknown` cho từng tiêu chí | Local minh bạch hơn |
| Top K | Cố định 5 | Người dùng chọn 3–10, mặc định 5 | Local linh hoạt hơn |
| Linear Regression trong app | Fit toàn bộ dữ liệu thị trường đang chọn | Vẫn có đường mô tả fit toàn bộ, nhưng parser km và giới hạn 0–100.000 được sửa | $R^2$ trên biểu đồ app vẫn là in-sample, không phải test score |
| Đánh giá hồi quy notebook | Không có holdout/baseline/CV | GroupShuffleSplit 20%, DummyRegressor, GroupKFold 5-fold | Local có quy trình đánh giá hợp lý hơn |
| CNN | Conv2D custom, train from scratch, 180×180 | MobileNetV2 ImageNet, 224×224, transfer learning | Local tận dụng pretrained features |
| Nơi train CNN | Trong notebook/app nếu artifact thiếu hoặc dataset đổi | `sources/CNN/train_cnn.py`, tách khỏi app | Local không làm người dùng chờ train khi mở app |
| Chia dữ liệu CNN | `Train` để train, `Test` vừa validation vừa test | `Train` tách train/validation; `Test` chỉ dùng đánh giá cuối | Remote có data leakage |
| Suy luận CNN | Top-1 và confidence | Top-3, ngưỡng từ chối 0,60 | Local thận trọng hơn với dự đoán yếu |
| Đánh giá CNN | Chỉ lưu `test_accuracy` | Accuracy, balanced accuracy, loss, confusion matrix, per-class precision/recall/F1, history | Local cung cấp nhiều bằng chứng hơn |
| UI | Một trang chat, text và ảnh cùng input | Hai trang `Tìm xe` và `Nhận diện ảnh`, sidebar điều khiển retrieval | Local rõ luồng tác vụ hơn |
| Dark mode/responsive | CSS rất ngắn, width 75% | Biến màu `light-dark()`, responsive cards, focus state, reduced motion | Local hoàn thiện UI hơn nhưng CSS lớn hơn |
| Loading | Progress bar khi encode candidate; spinner cho ảnh/kết quả | Pending query và spinner trong bong bóng assistant | Local tránh cảm giác nút gợi ý không phản hồi |
| Tổ chức notebook | Gần như chứa toàn bộ app, gồm cả UI và train CNN | Tập trung giải thích/thí nghiệm; mã UI đọc từ `app.py` trong `<details>` | Local dễ đọc theo mục tiêu học tập hơn |
| Chạy server | Cell cuối tự chạy Streamlit, không có cell dừng | Có cờ chạy và cell dừng | Local kiểm soát được hơn, nhưng giá trị cờ đang lưu chưa khớp hướng dẫn |

---

## 4. Khác biệt về cấu trúc notebook

### 4.1. Quy mô

| Chỉ số | Remote `main.ipynb` | Local `chatbot.ipynb` |
| --- | ---: | ---: |
| Tổng cell | 26 | 34 |
| Markdown cell | 13 | 14 |
| Code cell | 13 | 20 |
| Dòng source trong cell | 975 | 1.723 |
| Dòng code source | 910 | 1.538 |
| Code cell đã có execution count | 13/13 | 20/20 |
| Output object đã lưu | 6 | 29 |
| Kích thước file lúc đối chiếu | khoảng 90,6 KB | khoảng 815,3 KB |

Local lớn hơn chủ yếu vì:

- Có diễn giải công thức Transformer, TF-IDF, hồi quy và CNN.
- Có bảng kết quả retrieval, regression, CNN và ảnh biểu đồ đã nhúng.
- Có output HTML của phần xem mã Streamlit thu gọn.
- Có nhiều kiểm tra thực nghiệm hơn remote.

Kích thước file local không đồng nghĩa code triển khai tốt hơn; output ảnh nhúng làm notebook tăng kích thước và khiến diff Git khó đọc hơn.

### 4.2. Vai trò của notebook

**Remote:** notebook gần với một bản sao chạy được của `app.py`. Nó định nghĩa CSS, hàm truy xuất, biểu đồ, train CNN, hàm `main()` và cell mở Streamlit. Khi chạy cell import/cấu hình trong Jupyter, Streamlit có thể phát cảnh báo `missing ScriptRunContext` vì code giao diện đang chạy ở chế độ notebook/bare mode.

**Local:** notebook được tổ chức như tài liệu thực hành và kiểm chứng:

- Mục 1–5: chuẩn hóa dữ liệu, parser, embedding, TF-IDF và retrieval.
- Mục 6: đánh giá Linear Regression.
- Mục 7–9: đọc artifact CNN, giải thích transfer learning, đánh giá và suy luận ảnh.
- Mục 10: dùng AST để đọc riêng phần UI từ `app.py`, escape HTML và đặt trong `<details>` mặc định đóng; không `exec` mã UI.
- Mục 11: chạy Streamlit tùy chọn.

Local dùng `NotebookRuntime` để thay decorator cache trong notebook, thay vì monkey-patch module `streamlit`. Cách này giảm cảnh báo và tránh làm thay đổi hành vi Streamlit dùng chung.

### 4.3. Rủi ro đồng bộ code

Dù phần UI local được đọc trực tiếp từ `app.py`, nhiều hàm backend vẫn được định nghĩa lại trong `chatbot.ipynb`, ví dụ parser, dataset loader, retrieval, regression và CNN inference. Vì vậy vẫn có nguy cơ:

- Sửa `app.py` nhưng quên cập nhật cell tương ứng.
- Notebook và app trả kết quả khác nhau dù tên hàm giống nhau.
- Fix indentation/cú pháp ở một nơi nhưng còn lỗi ở nơi kia.

Giải pháp tốt hơn là chuyển logic dùng chung sang module như `car_chatbot_core.py`, sau đó cả notebook và app cùng import. Notebook chỉ nên giữ code thí nghiệm và cách trình bày kết quả.

---

## 5. Cài đặt, phụ thuộc và thiết bị tính toán

### 5.1. Remote

Remote chạy một lệnh `%pip install` gồm TensorFlow, NumPy, pandas, Matplotlib, scikit-learn, Gradio, Pillow, Streamlit, sentence-transformers, JupyterLab và ipykernel. Sau đó code import trực tiếp `torch` để chọn thiết bị:

- Có CUDA theo PyTorch: Qwen chạy trên GPU.
- Không có CUDA: Qwen chạy CPU.

Nhược điểm:

- Mỗi lần chạy cell vẫn gọi pip kiểm tra toàn bộ danh sách.
- Có `gradio` trong lệnh cài nhưng notebook đang dùng Streamlit.
- Không pin phiên bản nên kết quả có thể thay đổi theo thời gian.
- Qwen 0.6B nặng hơn MiniLM; lần tải model đầu và suy luận CPU thường chậm hơn.

### 5.2. Local

Local chỉ gọi pip với các package đang thiếu. Model embedding được import lazy trong `load_text_model`; TensorFlow được import khi cần tải CNN.

Biến môi trường:

- `CAR_EMBEDDING_MODEL`: thay model mặc định.
- `CAR_EMBEDDING_DEVICE`: ép `cpu`, `cuda` hoặc thiết bị phù hợp với môi trường.

Nếu không đặt device, `SentenceTransformer` tự chọn theo backend hiện có. Cách này linh hoạt, nhưng trạng thái thiết bị không được hiển thị rõ ngay trong UI.

Notebook local đã ghi nhận cảnh báo rằng TensorFlow bản mới trên native Windows không dùng CUDA theo cách cũ; muốn dùng GPU ổn định thường cần WSL2 hoặc backend tương thích khác. Điều này áp dụng cho CNN TensorFlow, không đồng nghĩa embedding PyTorch chắc chắn không dùng GPU.

### 5.3. Chênh lệch tài nguyên model

Ước lượng theo số tham số và dtype, không phải số đo chính xác của toàn bộ runtime:

- Qwen3-Embedding-0.6B: khoảng 0,6 tỷ tham số; riêng trọng số tương đương khoảng 1,2 GB ở 16-bit hoặc 2,4 GB ở 32-bit.
- Multilingual MiniLM-L12-v2: khoảng 118 triệu tham số; riêng trọng số 32-bit xấp xỉ 472 MB.
- Embedding full dataset local: $15.442 \times 384 \times 4$ byte, xấp xỉ 23,7 MB theo hệ thập phân.
- Nếu dùng Qwen mặc định 1024 chiều cho toàn bộ 15.442 dòng: khoảng 63,3 MB cho ma trận float32.

Runtime còn dùng thêm RAM cho tokenizer, activations, framework, dataframe và TF-IDF. Vì vậy không nên suy ra tổng RAM chỉ từ các số trên.

---

## 6. Chuẩn hóa dữ liệu và lập corpus tìm kiếm

### 6.1. Điểm giống nhau

Hai bản đều:

- Đọc `sources/Dataset/car-price.csv`.
- Chuyển giá về đơn vị triệu VNĐ.
- Loại giá rỗng/không hợp lệ và chỉ giữ giá dương.
- Ghép nhiều cột thành chuỗi mô tả `_SearchText`.
- Dùng tên cột tương đương để chịu được một số thay đổi schema.

### 6.2. Remote

Remote chỉ thêm `Price_Million` khi load dataset. `_SearchText` được tạo lại từ tập `candidates` sau hard filter ở mỗi truy vấn. Các trường như km, hộp số hoặc nhiên liệu chưa được chuẩn hóa thành cột riêng phục vụ parser.

Ưu điểm là code ngắn. Nhược điểm là mỗi lượt tìm kiếm phải lặp lại việc tạo mô tả và có ít trường chuẩn hóa để lọc chính xác.

### 6.3. Local

Local chuẩn bị nhiều trường ngay khi load dataset:

- `_row_id` để ánh xạ ứng viên về chỉ mục embedding/TF-IDF toàn dataset.
- `_Km` sau khi parse dấu chấm/phẩy phân tách hàng nghìn.
- `Hộp số` suy ra từ description.
- `Nhiên liệu` suy ra từ cột động cơ.
- `Tình trạng` được bổ sung từ `Condition` nếu cần.
- `_SearchTextNorm` bỏ dấu và dấu câu cho TF-IDF/rule matching.

Notebook local còn báo chất lượng dữ liệu trước khi diễn giải model. Output đã lưu cho thấy:

- 15.442 tin đăng.
- Giá trung bình khoảng 1.231,7 triệu, trung vị 659 triệu.
- 3.533 dòng trùng cặp `Name + Price_Million`; đây chỉ là dấu hiệu trùng, chưa chứng minh cùng một tin đăng.
- Tỷ lệ thiếu `_Km` khoảng 22,95%; năm và kiểu dáng khoảng 3,57%.

### 6.4. Fingerprint cache

Local gọi giá trị cache dataset là fingerprint, nhưng implementation hiện tại là:

- phiên bản schema thủ công `specs-v2`;
- kích thước file;
- `st_mtime_ns`.

Sau đó tên file embedding lấy SHA-256 của `model_name | fingerprint`. Như vậy đây **không phải SHA-256 nội dung CSV**. Nếu nội dung đổi mà kích thước và timestamp bị giữ nguyên, cache có thể không bị vô hiệu hóa. Muốn chặt chẽ hơn nên hash bytes của file hoặc hash theo nội dung các cột thực sự đi vào `_SearchText`.

Remote không có fingerprint cho text embedding. CNN remote có SHA-256 trên chuỗi `relative_path:size:mtime`, cũng không hash bytes từng ảnh.

---

## 7. Parser tiếng Việt và hard filter

### 7.1. Remote

Remote có các parser độc lập:

- `extract_price_filter`.
- `extract_year`: lấy một năm đầu tiên và lọc bằng chính xác.
- `extract_seats`.
- `extract_brand`: exact match, fuzzy match và full model name.
- `extract_body_type`: một nhóm nhỏ SUV, Sedan, MPV, Hatchback, Pickup/Bán tải và Crossover.

Hard filter remote chỉ có các tham số bật/tắt cho giá, năm, số chỗ và kiểu dáng. Hãng luôn được giữ nếu đã nhận diện.

### 7.2. Local

Local gom parser thành cấu trúc tiêu chí thống nhất:

- Mỗi tiêu chí có `field`, `op`, `value`, `label`.
- Các operator gồm `range`, `le`, `ge`, `eq`, `contains`.
- Cùng một hàm chuẩn hóa được dùng cho hard filter và trạng thái đối chiếu từng xe.
- Giá trị thiếu không được coi là khớp.

Các nhóm được hỗ trợ gồm:

1. Giá: cận dưới, cận trên, khoảng và xấp xỉ ±10%.
2. Năm: chính xác, khoảng, trước/sau, từ/đến.
3. Hãng và tên phiên bản.
4. Kiểu dáng mở rộng: SUV, Sedan, Crossover, Hatchback, Pickup, Van/Minivan, Coupe, Convertible/Cabriolet, Wagon và Truck.
5. Số chỗ và số cửa.
6. Số km với `k`, `nghìn`, `ngàn`.
7. Nhiên liệu.
8. Hộp số.
9. Xuất xứ.
10. Tình trạng xe.
11. Màu nội thất và ngoại thất.
12. Địa điểm và alias như HCM, TPHCM, Sài Gòn, Hà Nội.

Local xử lý km trước năm để `2000 km` không bị hiểu nhầm là đời 2000, và xử lý địa điểm trước màu/nhiên liệu để các tên tỉnh không bị nhận sai thành thuộc tính xe.

### 7.3. Nới điều kiện

Remote thử lần lượt:

1. Đủ điều kiện.
2. Bỏ giá.
3. Bỏ thêm năm.
4. Bỏ thêm số chỗ và kiểu dáng.

Local dùng năm giai đoạn tích lũy:

1. Giữ toàn bộ tiêu chí.
2. Bỏ giá.
3. Bỏ thêm năm.
4. Bỏ màu, địa điểm, km, nhiên liệu, hộp số, xuất xứ, tình trạng và số cửa.
5. Bỏ số chỗ và kiểu dáng.

Hãng không nằm trong danh sách bỏ của local. Khi giá bị nới, local đo khoảng cách tới **khoảng ngân sách**, thay vì chỉ lấy khoảng cách tới một scalar; xe còn nằm trong khoảng có khoảng cách bằng 0.

### 7.4. Hạn chế parser còn tồn tại ở cả hai bản

Các cụm `bé hơn` hoặc `nhỏ hơn` chưa được định nghĩa như synonym cận trên. Do regex kiểm tra từ `hơn` cho cận dưới trước, câu như `bé hơn 500 triệu` có thể bị hiểu thành **từ 500 triệu**. Cụm không tự nhiên `dưới hơn 500 triệu` cũng có nguy cơ tương tự vì phần ngay trước số kết thúc bằng `hơn`.

Cần sửa bằng cách:

- Chuẩn hóa `bé hơn`, `nhỏ hơn`, `rẻ hơn` thành nhóm cận trên trước khi kiểm tra `hơn` đứng độc lập.
- Dùng boundary cho toàn cụm, không để từ `hơn` khớp bên trong synonym dài hơn.
- Thêm unit test cho các câu phủ định, khoảng, cận trên và cận dưới.

Đây là vấn đề rule parsing; đổi sang embedding mạnh hơn không tự sửa hard filter vì model embedding chỉ xếp hạng sau khi parser đã tạo tập ứng viên.

---

## 8. Transformer, embedding và xếp hạng

### 8.1. Remote: Qwen semantic retrieval

Luồng remote:

1. Hard filter dataset.
2. Tạo `_SearchText` cho ứng viên.
3. Chia ứng viên thành batch 32.
4. Encode từng batch bằng Qwen và `normalize_embeddings=True`.
5. Encode query với `prompt_name="query"`.
6. Tính:

$$
s_i = e_i^{\mathsf T}e_q.
$$

Vì hai vector đã chuẩn hóa $L_2$, tích vô hướng bằng cosine similarity. Kết quả được sắp giảm dần theo `Similarity`, trừ khi đang fallback ngân sách thì khoảng cách giá đứng trước similarity.

`cached_car_embedding_batch` dùng `st.cache_data` với tuple nội dung batch. Cache có thể tái sử dụng nếu batch giống hệt, nhưng:

- không có file embedding bền vững qua lần chạy app;
- thay đổi tập ứng viên có thể làm ranh giới batch thay đổi;
- truy vấn rộng vẫn phải encode nhiều ứng viên ở câu hỏi đầu;
- lỗi tải model hoặc lỗi Hugging Face làm toàn bộ semantic retrieval thất bại.

Điểm mạnh của remote là Qwen là model embedding lớn hơn và có query instruction. Tuy nhiên mã nguồn không có benchmark chứng minh Qwen cho kết quả tốt hơn trên corpus tin xe tiếng Việt cụ thể này.

### 8.2. Local: MiniLM + TF-IDF + Hybrid

Local lập hai chỉ mục:

1. **Semantic:** encode toàn bộ `_SearchText`, chuẩn hóa $L_2$, lưu float32 vào `.npy`.
2. **Lexical:** `TfidfVectorizer` với unigram/bigram, `min_df=2`, `max_features=40.000`, `sublinear_tf=True`.

Ba chế độ:

- `Semantic`: dùng cosine.
- `TF-IDF`: dùng tích vô hướng sparse giữa query và corpus đã chuẩn hóa.
- `Hybrid`: đưa từng loại điểm trong tập ứng viên về $[0,1]`, rồi cộng có trọng số:

$$
s_{\mathrm{hybrid}}
= w\,\operatorname{minmax}(s_{\mathrm{semantic}})
+ (1-w)\,\operatorname{minmax}(s_{\mathrm{tfidf}}).
$$

Mặc định $w=0{,}7$. UI cho phép điều chỉnh từ 0 đến 1.

Lợi ích:

- Semantic bắt các cách diễn đạt gần nghĩa.
- TF-IDF giữ tín hiệu từ khóa cụ thể như model xe, năm, phiên bản.
- Hybrid giảm việc phụ thuộc hoàn toàn vào một nguồn điểm.
- Nếu model không tải được, app chuyển sang TF-IDF và chỉ thử model lỗi một lần trong phiên để tránh lặp HTTP traceback.

Hạn chế:

- Min-max được tính trong tập ứng viên của từng câu; vì vậy điểm Hybrid giữa hai câu hỏi không so trực tiếp được.
- Nếu mọi semantic score bằng nhau, thành phần semantic sau min-max trở thành 0.
- Hybrid score không phải xác suất hoặc confidence.
- Lần đầu vẫn phải tải MiniLM và encode toàn bộ 15.442 dòng nếu file cache chưa có.
- File cache nằm trong `.cache` và đang bị Git ignore; máy mới vẫn phải tạo lại.

### 8.3. Khác biệt latency

**Remote:** chi phí phụ thuộc số ứng viên sau hard filter. Một câu rất cụ thể có thể nhanh vì chỉ encode ít dòng. Một câu rộng có thể chậm; truy vấn mới tạo tập/batch khác có thể tiếp tục tốn encode.

**Local:** câu Semantic/Hybrid đầu tiên trên máy chưa có cache chậm hơn do encode toàn dataset. Sau khi `.npy` được tạo, mỗi câu chỉ encode query, lấy các hàng cần thiết và nhân ma trận; latency các câu sau ổn định hơn. TF-IDF không cần tải model embedding.

Đây là đánh đổi giữa **lazy candidate encoding** và **precomputed corpus index**.

### 8.4. Kết quả kiểm tra đã lưu trong notebook local

Notebook local chạy ba câu × ba chế độ. Ví dụ:

- `Toyota Innova 2020 7 chỗ`: cả ba chế độ đưa `Toyota Innova V 2.0 AT - 2020` lên đầu trong output đã lưu.
- `Mercedes Benz C200 2021`: Hybrid/Semantic đưa `C300 AMG` lên đầu, trong khi TF-IDF đưa `C200 Exclusive` lên đầu.

Điều này cho thấy semantic và lexical có thể ưu tiên khác nhau, nhưng **không đủ để kết luận chế độ nào đúng hơn**. Notebook local đã ghi đúng rằng chỉ nên tính Precision@5/MRR khi có relevance label do người chấm.

### 8.5. Khoảng trống đánh giá retrieval

Remote không có benchmark retrieval. Local có bảng case study nhưng chưa có:

- tập truy vấn chuẩn hóa;
- danh sách xe liên quan cho từng truy vấn;
- inter-annotator agreement nếu có nhiều người gán nhãn;
- Precision@K, Recall@K, MRR, MAP hoặc nDCG;
- benchmark latency cold start/warm start và CPU/GPU;
- ablation giữa hard filter, semantic, TF-IDF và Hybrid.

Vì vậy chưa nên phát biểu rằng Hybrid, MiniLM hoặc Qwen “chính xác hơn” trên toàn hệ thống.

---

## 9. Linear Regression

### 9.1. Mục đích trong cả hai bản

Hai biểu đồ đều mô tả quan hệ:

- năm sản xuất và giá niêm yết;
- số km đã đi và giá niêm yết.

Nếu người dùng nêu hãng, thị trường được thu hẹp theo hãng nếu có dữ liệu; nếu không thì dùng toàn dataset. Xe được chatbot gợi ý chỉ được đánh dấu trên biểu đồ, không tham gia quyết định hệ số theo một cơ chế riêng.

### 9.2. Remote

Remote fit `LinearRegression` trên toàn bộ điểm đang vẽ rồi tính $R^2$ trên chính các điểm đó. Đây là **descriptive/in-sample score**, không phải hiệu năng dự đoán dữ liệu chưa thấy.

Không có:

- train/test split;
- baseline;
- RMSE/MSE test;
- cross-validation;
- kiểm soát dòng trùng giữa train/test.

Parser km remote dùng regex giữ cả chữ số và dấu chấm. Với dữ liệu Việt Nam như `10.000 km`, giá trị có thể thành `10.0` thay vì `10000`. Ngoài ra remote fit tất cả km không âm, nhưng đường vẽ và trục chỉ đặt 0–100.000 km. Outlier lớn hơn 100.000 vẫn có thể làm đổi hệ số dù không xuất hiện trong khung hình.

### 9.3. Local app

Local sửa parser km bằng cách bỏ mọi ký tự không phải chữ số, rồi chỉ fit các bản ghi từ 0 đến 100.000 km. Vì tập dùng để fit và vùng hiển thị nhất quán hơn, đường trên biểu đồ dễ diễn giải hơn remote.

Tuy nhiên, $R^2$ hiển thị trên biểu đồ app vẫn là in-sample score. Nó chỉ mô tả mức độ một đường thẳng giải thích các điểm hiện tại, không phải chứng nhận khả năng định giá xe.

### 9.4. Local notebook

Notebook thêm thí nghiệm đánh giá riêng:

1. Tạo group theo `Name + Price_Million` để các cặp trùng không xuất hiện ở cả train và test.
2. `GroupShuffleSplit` giữ 20% làm holdout.
3. Fit Linear Regression trên 80% train.
4. So với `DummyRegressor(strategy="mean")`.
5. Chạy `GroupKFold(n_splits=5)` chỉ trên train.
6. Báo MSE, RMSE và $R^2$ test.

Output đã lưu:

| Feature | n | RMSE test (triệu VNĐ) | $R^2$ test | RMSE baseline | CV train $R^2$ trung bình |
| --- | ---: | ---: | ---: | ---: | ---: |
| Năm sản xuất | 14.891 | 1.703,32 | 0,0466 | 1.744,92 | 0,0349 |
| Km, chỉ 0–100.000 | 10.728 | 1.686,11 | 0,0078 | 1.692,72 | 0,0074 |

Notebook cũng ghi $R^2$ mô tả cho mọi km không âm là khoảng `0,000287` trên 11.898 dòng.

Diễn giải:

- Một biến năm hoặc km giải thích rất ít biến thiên giá.
- $R^2$ gần 0 không phải do Hybrid/TF-IDF; retrieval chỉ quyết định xe đánh dấu, không quyết định quan hệ hồi quy của toàn thị trường.
- RMSE rất lớn cho thấy không nên dùng hai đường một biến này như công cụ định giá đáng tin cậy.
- Group theo tên + giá chỉ giảm một dạng leakage; nó chưa chứng minh các ảnh/tin/người bán thực sự độc lập.

Local tốt hơn remote ở quy trình đánh giá, nhưng vẫn chỉ là mô hình một biến. Muốn định giá thực tế cần mô hình đa biến, xử lý outlier, thời điểm tin đăng, phiên bản xe và đánh giá theo split phù hợp thời gian/nhóm xe.

---

## 10. CNN và nhận diện ảnh

### 10.1. Remote: custom CNN train trong app/notebook

Cấu hình remote:

- Ảnh 180×180.
- Batch size 32.
- Tối đa 20 epoch.
- Data augmentation: flip, rotation 0,04 và zoom 0,08.
- Bốn lớp Conv2D: 32, 64, 128, 128 filter.
- MaxPooling sau ba conv đầu.
- GlobalAveragePooling, Dropout, Dense 128, Dropout, softmax 7 lớp.
- Adam learning rate 0,001.
- Khoảng 258 nghìn tham số nếu có 7 lớp.

`load_or_train_cnn` kiểm tra chữ ký dataset. Nếu artifact thiếu hoặc chữ ký khác, app tự train. Điều này thuận tiện cho một notebook tự chứa, nhưng có các vấn đề:

1. Mở app có thể kích hoạt công việc train dài.
2. Cùng một quy trình nặng nằm trong luồng phục vụ người dùng.
3. Khó kiểm soát tài nguyên và timeout khi deploy.
4. Artifact remote không nằm trong `sources/Transformer` của commit; fresh clone có thể phải train lại.

### 10.2. Leakage trong CNN remote

Remote tạo:

- `train_ds` từ `Dataset/Train`.
- `test_ds` từ `Dataset/Test`.

Nhưng sau đó gọi `model.fit(..., validation_data=test_ds, ...)`, theo dõi `val_accuracy` bằng EarlyStopping, rồi gọi `model.evaluate(test_ds)` và báo `test_accuracy`.

Như vậy tập Test đã ảnh hưởng tới việc chọn epoch/best weights. Điểm test không còn là đánh giá hoàn toàn độc lập. Đây là khác biệt phương pháp quan trọng nhất của phần CNN.

### 10.3. Local: MobileNetV2 transfer learning

Local dùng artifact từ `sources/CNN/train_cnn.py`:

- MobileNetV2 pretrained ImageNet.
- Ảnh 224×224.
- Global average pooling từ backbone.
- Dropout, Dense 128, Dropout, softmax 7 lớp.
- Giai đoạn đầu khóa backbone.
- Giai đoạn sau fine-tune với learning rate nhỏ; BatchNorm tiếp tục được giữ frozen.
- Có class weight kiểu effective-number.
- Có EarlyStopping, ReduceLROnPlateau và checkpoint tốt nhất.
- Tập `Train` được chia riêng train/validation 80/20.
- `Test` chỉ dùng cho đánh giá cuối.

App local chỉ tải `model.keras`, `labels.json`, `meta.json` và `metrics.json`; không gọi `model.fit`.

### 10.4. Metrics local đã lưu

Artifact local tại thời điểm đối chiếu:

| Metric | Giá trị |
| --- | ---: |
| Kiến trúc | MobileNetV2 |
| Kích thước ảnh | 224×224 |
| Số lớp | 7 |
| Số tham số | 2.422.855 |
| Test loss | 0,1119 |
| Test accuracy | 97,05% |
| Balanced accuracy | 96,15% |
| Macro-F1 | 96,54% |
| Weighted-F1 | 97,03% |
| Epoch tốt nhất | 20 |

Bảy lớp gồm Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Maruti Suzuki Swift, Tata Safari và Toyota Innova.

Notebook local tính per-class precision/recall/F1 từ confusion matrix, đồng thời hiển thị:

- chuẩn hóa theo hàng để đọc recall;
- chuẩn hóa theo cột để đọc precision;
- đường train/validation accuracy;
- đường train/validation loss.

Điểm mạnh là không chỉ nhìn accuracy tổng. Tuy nhiên đây là số liệu đọc từ artifact đã lưu; notebook không evaluate lại toàn bộ test mỗi lần chạy.

### 10.5. Suy luận ảnh

**Remote:** trả top-1 class và confidence. Không có ngưỡng từ chối. Một ảnh ngoài bảy lớp vẫn bị ép vào một lớp.

**Local:** trả top-3 softmax và dùng ngưỡng 0,60:

- điểm cao nhất dưới 0,60: `Không chắc chắn`;
- từ 0,60: trả lớp dự đoán.

Ngưỡng từ chối an toàn hơn việc luôn khẳng định top-1, nhưng 0,60 hiện là hằng số. Notebook cũng ghi rõ softmax cao không chứng minh ảnh thuộc một trong bảy lớp và không đảm bảo xác suất đã được calibration.

### 10.6. Calibration chưa được dùng trong app Streamlit local

Thư mục artifact local có `calibration.json` và `prototypes.npy`; `app_gradio.py` có logic temperature scaling và kiểm tra prototype/OOD. Tuy nhiên `sources/Transformer/app.py` và `chatbot.ipynb` đang dùng softmax trực tiếp với ngưỡng 0,60, chưa áp dụng hai artifact này.

Do đó không nên mô tả app Streamlit hiện tại là đã calibration hoặc đã phát hiện OOD bằng prototype. Nếu muốn đồng nhất, cần port logic đã có trong `app_gradio.py` sang `app.py` và bổ sung test.

---

## 11. Giao diện và trải nghiệm người dùng

### 11.1. Remote

- Một màn hình chat chứa cả tìm xe và upload ảnh.
- `st.chat_input(..., accept_file="multiple")`, nhưng code chỉ xử lý file đầu tiên.
- Kết quả tìm xe được render trực tiếp bằng container, metric và Markdown.
- Progress bar phản ánh số batch embedding đã chạy.
- Lịch sử lưu `question`, sau đó mỗi lần rerun gọi lại `process_text_question` để dựng kết quả. Cache batch có thể giảm chi phí, nhưng biểu đồ và một phần xử lý vẫn được chạy lại.
- CSS chỉ giới hạn chiều rộng chính và bottom input ở 75%.

Ưu điểm là đơn giản và dễ theo dõi. Nhược điểm là text và image dùng chung luồng nhưng không kết hợp thật sự; code còn ghi rõ chỉ tìm xe khi không upload ảnh.

### 11.2. Local

- Dùng `st.navigation` với hai trang: `Tìm xe` và `Nhận diện ảnh`.
- Sidebar chọn Hybrid/Semantic/TF-IDF, trọng số semantic và top K.
- Câu hỏi được đưa vào `pending_query`; spinner xuất hiện trong bong bóng assistant rồi mới thêm kết quả vào session.
- Kết quả backend được chuyển thành record có tiêu chí, score và ảnh biểu đồ; lịch sử render lại từ dữ liệu đã lưu thay vì tìm lại toàn bộ truy vấn.
- Card xe có thông tin chính, mô tả, link, điểm semantic/TF-IDF/ranking và trạng thái từng tiêu chí.
- CSS dùng `light-dark()` cho light/dark theme, container query, media query mobile, focus-visible và reduced motion.
- Ảnh nhận diện nằm ở trang riêng, có preview, top-3 và metadata model.

Đây là bước tiến rõ về UX và explainability. Đổi lại, `app.py` đã dài 2.018 dòng và phụ thuộc nhiều CSS selector nội bộ của Streamlit; nâng Streamlit có thể làm một số selector mất hiệu lực.

### 11.3. An toàn khi render dữ liệu

Local escape dữ liệu động trước khi chèn vào HTML và chỉ tạo link nếu URL khớp `http/https`. Remote chủ yếu dùng widget/Markdown native nên bề mặt HTML động nhỏ hơn. Cả hai cách đều hợp lý, nhưng local cần tiếp tục duy trì nguyên tắc escape vì UI tùy biến nhiều hơn.

---

## 12. Cache, lỗi và khả năng chạy offline

### 12.1. Remote

Các lỗi có thể làm mất chức năng chính:

- Không tải được Qwen: không có TF-IDF fallback.
- Không có Internet và model chưa cache: semantic retrieval thất bại.
- Không có CNN artifact phù hợp: app thử train, có thể rất lâu hoặc lỗi tài nguyên.
- Dataset ảnh thiếu: CNN bị vô hiệu hóa bằng warning, nhưng phần text vẫn có thể chạy.

### 12.2. Local

- Dataset và TF-IDF được cache theo fingerprint.
- Embedding được lưu `.npy` và ghi nguyên tử qua file tạm trước khi replace.
- Model semantic lỗi: chuyển sang TF-IDF và ghi chế độ thực tế.
- CNN artifact thiếu: trang ảnh báo lỗi và hướng dẫn chạy `python sources/CNN/train_cnn.py --mode train`; app không tự train.
- Có lệnh `python sources/Transformer/app.py --prepare` để chuẩn bị dataset, TF-IDF và embedding trước khi demo.

Local phù hợp hơn cho môi trường mạng không ổn định sau khi đã chuẩn bị resource. Máy hoàn toàn mới, không Internet và chưa có model/cache vẫn chỉ dùng được TF-IDF; CNN cần artifact local.

---

## 13. Cách khởi chạy từ notebook

### 13.1. Remote

Cell cuối remote đặt `PROJECT_ROOT = Path.cwd()`, sau đó tìm `APP_PATH = PROJECT_ROOT / "app.py"`. Tên biến và thông báo nói chạy từ thư mục gốc project, nhưng `app.py` thực tế nằm trong `sources/Transformer`. Cell chỉ đúng khi current working directory chính là thư mục có `app.py`, hoặc khi có một file app khác ở root.

Cell tự tạo process Streamlit và ghi `streamlit.log`, nhưng không có cell dừng process. Chạy lại notebook có kiểm tra process cũ trong biến global.

### 13.2. Local

Local dùng `NOTEBOOK_DIR / "app.py"`, ít phụ thuộc current working directory hơn, và có cell terminate/kill process.

Tuy nhiên notebook hiện có mâu thuẫn:

- Markdown hướng dẫn giữ `RUN_STREAMLIT = False` khi Run All.
- Giá trị đã lưu trong cell là `RUN_STREAMLIT = True`.
- Markdown cell dừng nói chỉ bật khi kết thúc demo.
- Giá trị đã lưu là `STOP_STREAMLIT = True`.

Vì cell dừng nằm ngay sau cell mở, Run All có thể mở rồi dừng app ngay. Nên đổi mặc định cả hai cờ về `False`, hoặc gộp thành một helper rõ ràng với hành động `start/stop/status`.

---

## 14. Tính tái lập và chất lượng kỹ thuật

### 14.1. Điểm remote làm tốt

- Kiến trúc nhỏ, dễ đọc và dễ trình bày demo.
- Qwen hỗ trợ query prompt rõ ràng.
- Device Qwen tự chọn CUDA/CPU bằng PyTorch.
- Cache batch ứng viên tránh encode lại batch giống hệt trong cùng cache lifecycle.
- Có chữ ký dataset ảnh để biết khi nào artifact CNN cũ.
- Một file có thể chạy gần như đầy đủ từ dữ liệu tới UI.

### 14.2. Điểm local làm tốt

- Tách train CNN khỏi serving.
- Chống leakage Test/validation ở pipeline CNN.
- Có fallback retrieval khi embedding lỗi.
- Có cache embedding bền vững.
- Có nhiều tiêu chí và giải thích từng kết quả.
- Có holdout, baseline và CV cho hồi quy.
- Có balanced accuracy, per-class metrics và confusion matrix cho CNN.
- UI phân trang, hỗ trợ theme và loading rõ ràng.
- Notebook nêu đúng giới hạn: retrieval score không phải xác suất, $R^2$ mô tả không phải test score, một ảnh không thay thế đánh giá test.

### 14.3. Điểm local cần tiếp tục cải thiện

1. Backend vẫn trùng giữa notebook và app.
2. Chưa pin dependency/version model/dataset schema.
3. Fingerprint CSV dựa vào size + mtime, chưa hash nội dung.
4. Chưa có retrieval relevance dataset và metric tổng quát.
5. Parser còn lỗi với `bé hơn`, `nhỏ hơn`, `dưới hơn`.
6. Ngưỡng CNN 0,60 chưa được chọn/báo cáo bằng đường risk-coverage rõ ràng.
7. Streamlit app chưa dùng calibration/prototype dù artifact đã có.
8. Output ảnh/HTML làm notebook lớn và diff Git khó đọc.
9. Cờ start/stop Streamlit đang không khớp phần hướng dẫn.
10. `app.py` lớn; cần module hóa parser, retrieval, regression, vision và UI.

### 14.4. Điểm remote cần ưu tiên sửa nếu tiếp tục dùng

1. Không dùng `Test` làm `validation_data` cho CNN.
2. Không train CNN trong app serving.
3. Thêm fallback khi Qwen không tải được.
4. Tạo persistent embedding index hoặc vector store.
5. Sửa parser km có dấu phân cách hàng nghìn.
6. Đánh giá regression trên holdout thay vì chỉ in-sample.
7. Thêm evaluation cho retrieval.
8. Sửa đường dẫn cell chạy Streamlit.
9. Pin dependency và model revision.
10. Tách logic dùng chung khỏi notebook/app để tránh duplication.

---

## 15. Nên chọn kiến trúc nào?

### Chọn remote làm nền nếu

- Cần một demo nhỏ, ít file, dễ trình bày nhanh.
- Muốn thử trực tiếp chất lượng Qwen embedding.
- Dataset ứng viên sau hard filter thường rất nhỏ.
- Chấp nhận cần Internet/model cache và chưa cần fallback offline.
- Phần CNN chỉ là prototype và sẽ được tách/sửa trước khi đánh giá chính thức.

### Chọn local làm nền nếu

- Cần app demo ổn định qua nhiều truy vấn.
- Cần chạy được khi semantic model lỗi.
- Cần giải thích vì sao từng xe được chọn hoặc khác yêu cầu.
- Cần kiểm soát train/evaluation CNN và tránh train trong app.
- Cần báo cáo học thuật có baseline, holdout, confusion matrix và giới hạn phương pháp.
- Cần UI tốt hơn cho theme, mobile, loading và nhiều trang.

### Khuyến nghị

Dùng **kiến trúc local làm baseline chính**, nhưng không mặc định kết luận MiniLM là model cuối cùng. Giữ abstraction `CAR_EMBEDDING_MODEL` và benchmark ít nhất:

- MiniLM hiện tại.
- Qwen3-Embedding-0.6B.
- TF-IDF.
- Hybrid MiniLM + TF-IDF.
- Nếu tài nguyên cho phép, Hybrid Qwen + TF-IDF.

Việc chọn model nên dựa trên chất lượng relevance, cold/warm latency, RAM/VRAM, kích thước tải và khả năng deploy thực tế.

---

## 16. Kế hoạch cải thiện đề xuất

### Mức ưu tiên cao

1. **Tạo bộ test parser** cho giá, năm, km, hãng, màu, địa điểm và các cụm `bé hơn`/`nhỏ hơn`/`dưới hơn`.
2. **Tạo retrieval benchmark** khoảng 50–100 câu hỏi, mỗi câu có danh sách tin liên quan được gán nhãn.
3. **Tách core module** để notebook và Streamlit import chung.
4. **Đổi cờ Run All** trong notebook về an toàn: không tự mở hoặc tự dừng server.
5. **Pin môi trường** bằng file dependency có version và lưu model revision.

### Mức ưu tiên trung bình

6. Đổi fingerprint sang hash nội dung liên quan.
7. Thêm benchmark cold start/warm start trên CPU và GPU.
8. Port calibration/OOD từ `app_gradio.py` sang Streamlit hoặc bỏ artifact không sử dụng khỏi luồng mô tả.
9. Thêm kiểm thử dataset schema và thông báo trường nào thiếu.
10. Xóa output nặng trước khi commit hoặc dùng notebook cleaning tool; lưu biểu đồ quan trọng thành artifact riêng.

### Mức ưu tiên nghiên cứu

11. Thử reranker cross-encoder trên top-N sau retrieval hai tầng.
12. Dùng model hồi quy đa biến và split theo thời gian/nhóm xe.
13. Đánh giá CNN trên ảnh ngoài bảy lớp và báo risk-coverage theo ngưỡng từ chối.
14. So sánh transfer learning backbone nhẹ khác như EfficientNetV2B0/MobileNetV3 nếu cần tối ưu inference.

---

## 17. Kết luận cuối

Remote thể hiện một pipeline end-to-end đơn giản: Qwen semantic retrieval, Linear Regression mô tả và CNN custom tự train. Nó phù hợp để quan sát luồng tổng quát, nhưng còn thiếu fallback, đánh giá retrieval, đánh giá hồi quy ngoài mẫu và đặc biệt có leakage giữa validation/test của CNN.

Local đã chuyển từ demo nguyên khối sang hệ thống có chỉ mục, fallback, parser nhiều tiêu chí, explainability, đánh giá mô hình và artifact serving rõ hơn. Đây là cải tiến có ý nghĩa cả về kỹ thuật lẫn nội dung báo cáo. Tuy vậy, local chưa phải hệ thống hoàn chỉnh: retrieval chưa có ground truth, parser vẫn có edge case, notebook còn trùng backend với app, và calibration CNN chưa được dùng trong Streamlit.

Vì vậy, nhận định phù hợp nhất là:

> **Local mạnh hơn về độ ổn định, khả năng giải thích, quy trình đánh giá và UX; remote đơn giản hơn và dùng embedding model lớn hơn, nhưng chưa có bằng chứng thực nghiệm rằng chất lượng tìm kiếm tốt hơn.**
