# PPR501 Car Analysis

---

# Nhận Diện Và Phân Tích Giá Xe Hơi Ở Việt Nam
Eng: Support System for Car Recognition, Valuation, and Recommendation in Vietnam

Đây là dự án trực quan hóa dữ liệu, tập trung vào việc phân tích giá cả xe hơi ở Việt Nam (1990-2023).

Mục tiêu của dự án là cung cấp cái nhìn tổng quan về thị trường ô tô tại Việt Nam thông qua các biểu đồ.

**Library**:
* Python
* Numpy
* Pandas
* Keras
* TensorFlow
* Matplotlib
* Gradio
* Streamlit

---

# Tổng quan đề tài

## 1. Giới thiệu về dữ liệu

- **Nguồn dữ liệu**: Kaggle
- **License**: MIT
- **Thông tin dữ liệu**:
  - Gồm **3,352 ảnh** cho việc **Train** và **813 ảnh** cho việc **Test**
  - Bộ dataset CSV có thuộc tính bao gồm: tên, tình trạng, giá, vị trí, mô tả, năm sản xuất,mẫu mã, số km đã đi, loại động cơ, công suất, và nhiều thông tin liên quan khác

## 2. Convolutional Neural Network (CNN)

- Source: `sources/CNN`
- Tài liệu đầy đủ (kiến trúc, cách chạy, đánh giá, troubleshooting): [`sources/CNN/README.md`](./sources/CNN/README.md)

Phân loại ảnh xe thuộc 7 nhóm bằng **MobileNetV2 pre-trained trên ImageNet**
(2 stage: đóng băng backbone → fine-tune), kèm **prior correction** để xác suất
dự đoán cân bằng giữa các hãng.

```bash
cd sources/CNN
python train_cnn.py --mode train --epochs 20 --freeze-epochs 5
python train_cnn.py --mode serve        # http://localhost:7860
```

- Lenet 5:

![Lenet 5](./docs/imgs/lenet5-original.png)

![Lenet 5 Car](./docs/imgs/lenet5-car.png)

## 3. Linear Regression

- Source: `sources\LinearReg`

![Linear Regression](./docs/imgs/vietnam-car-price-regression.png)

## 4. Transformer (Chatbot gợi ý xe)

- Source: `sources/Transformer`

Ứng dụng chatbot Streamlit sử dụng **Sentence-Transformers embedding** để lọc và
xếp hạng xe theo mô tả tiếng Việt của người dùng, kèm nhận diện ảnh xe bằng CNN
và gợi ý giá bán bằng **Linear Regression**.

```bash
cd sources/Transformer
streamlit run app.py --server.port 8502   # http://localhost:8502
```

- Notebook: [`sources/Transformer/main.ipynb`](./sources/Transformer/main.ipynb)

---

# Nguồn tham khảo

- [Nguồn dữ liệu trên Kaggle - Giá xe](https://www.kaggle.com/datasets/qucvinhdng/automobile-dataset)

- [Nguồn dữ liệu trên Kaggle - Ảnh xe](https://www.kaggle.com/code/kshitij192/car-images-classification-using-cnn/data)

- [Source Code](https://github.com/ptnghia3502/PPR501-car-analysis)

---

# Tác giả
- **Giảng viên**: Nguyễn Xuân Sâm
- **Thành viên**:
  - Phạm Trọng Nghĩa - 26MS23417
  - Trần Công Vinh - 26MS23406 
  - Trần Quang Duy - 26MS23410
