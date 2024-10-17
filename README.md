# Human_Activity_Recognition

## 1. Giới thiệu về Mô hình HAR

Mục tiêu của mô hình Human Activity Recognition (HAR) là nhận diện và phân loại các hành động của con người dựa trên dữ
liệu từ video. Quy trình bao gồm việc trích xuất các đặc trưng chuyển động và sử dụng mô hình học sâu để dự đoán hành
động tương ứng.

## 2. Quy trình xây dựng mô hình

### 2.1. Load dữ liệu video

Video được đưa vào hệ thống để xử lý.
Sử dụng MediaPipe để phát hiện tư thế (pose detection) trong từng khung hình của video.

### 2.2. Phát hiện và xử lý tư thế (Pose Detection)

MediaPipe trích xuất các điểm đặc trưng (keypoints) từ cơ thể người qua từng khung hình.
Các điểm này đại diện cho các vị trí như vai, đầu gối, khuỷu tay, cổ tay, v.v.

### 2.3. Tiền xử lý dữ liệu

Sau khi có dữ liệu từ MediaPipe, thực hiện các bước sau:
Điều chỉnh kích thước dữ liệu:
Tăng hoặc giảm chiều dữ liệu để đạt kích thước chuẩn size = 35.
Điều này đảm bảo dữ liệu nhất quán và dễ xử lý cho các bước sau.
Xử lý dữ liệu thiếu (NaN):
Điền giá trị NaN vào các khoảng trống để đảm bảo dữ liệu không có lỗi trong quá trình huấn luyện.

### 2.4. Huấn luyện mô hình với Deep LSTM

Sau khi dữ liệu đã được xử lý, nó sẽ được đưa vào mô hình Deep LSTM (Long Short-Term Memory).
Mô hình LSTM được sử dụng vì khả năng ghi nhớ và phân tích các chuỗi thời gian, rất phù hợp với bài toán nhận diện hành
động từ video.

### 2.5. Phân loại hành động

Mô hình Deep LSTM sẽ dự đoán và phân loại các hành động cụ thể dựa trên chuỗi các tư thế đầu vào.

## Project Structure

```bash
har_project/
│
├── data/
│   ├── raw/                         # Video thô chưa xử lý
│   ├── processed/                   # Dữ liệu đã qua tiền xử lý (keypoints từ MediaPipe)
│   ├── interim/                     # Dữ liệu tạm thời cho việc thử nghiệm và prototyping
│
├── docs/
│   ├── api/                         # Tham chiếu về API (nếu dự án có endpoint)
│   ├── design_docs/                 # Tài liệu thiết kế mô hình và kiến trúc hệ thống
│   ├── tutorials/                   # Hướng dẫn từng bước (setup, huấn luyện, inference)
│   └── diagrams/                    # Sơ đồ quy trình huấn luyện và dự đoán của mô hình
│
├── notebooks/
│   ├── eda/                         # Phân tích dữ liệu ban đầu (EDA) với keypoints
│   ├── model_experiments/           # Các thí nghiệm thử các biến thể của mô hình
│   └── results/                     # Hình ảnh và bảng so sánh kết quả các mô hình
│
├── src/
│   ├── data_pipeline/
│   │   ├── sourcing/
│   │   │   ├── downloaders/         # Script tải video dữ liệu mẫu
│   │   │   ├── scrapers/            # Script để thu thập dữ liệu từ web (nếu cần)
│   │   │   ├── validators/          # Kiểm tra dữ liệu tải về
│   │   │   └── annotations/         # Công cụ dán nhãn (annotation) cho dữ liệu
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── cleaners/            # Xử lý NaN và làm sạch dữ liệu
│   │   │   ├── transformers/        # Điều chỉnh kích thước chuỗi pose thành 35 frame
│   │   │   └── splitters/           # Chia dữ liệu thành train/test/validation
│   │   │
│   │   ├── loaders/
│   │   │   ├── dataset_classes/     # Dataset class cho TensorFlow/PyTorch
│   │   │   └── batchers/            # Quản lý batch cho training
│   │   │
│   │   ├── validation/
│   │   │   ├── sanity_checks/       # Kiểm tra tính hợp lệ của dữ liệu đã qua xử lý
│   │   │   └── statistics/          # Tính toán và trực quan hóa thống kê
│   │   │
│   │   └── utils/
│   │       ├── visualization/       # Hiển thị các pose và video sau augmentation
│   │       └── logging/             # Ghi log quá trình xử lý dữ liệu
│   │
│   ├── models/
│   │   ├── architectures/           # Kiến trúc LSTM và các biến thể
│   │   ├── losses/                  # Loss function cho HAR
│   │   └── metrics/                 # Metrics (accuracy, F1) để đánh giá mô hình
│   │
│   ├── training/
│   │   ├── experiments/
│   │   │   ├── experiment1/         # Config và log cho thí nghiệm đầu tiên
│   │   │   ├── experiment2/         # Thí nghiệm thứ hai với hyperparameter khác
│   │   │   └── ...
│   │   │
│   │   ├── scripts/
│   │   │   ├── train_model.py       # Script huấn luyện mô hình
│   │   │   ├── validate_model.py    # Script chạy kiểm thử mô hình
│   │   │   └── resume_training.py   # Tiếp tục huấn luyện từ checkpoint
│   │   │
│   │   ├── hyperparameters/
│   │   │   └── search_algos/        # Thuật toán tìm kiếm hyperparameters (grid, random)
│   │   │
│   │   └── callbacks/
│   │       ├── early_stopping.py    # Callback để dừng sớm nếu không cải thiện
│   │       ├── model_checkpointing.py # Callback lưu checkpoint
│   │       └── tensorboard_logging.py # Log kết quả lên TensorBoard
│   │
│   ├── inference/
│   │   ├── deployment/
│   │   │   ├── api_endpoints/       # Endpoint API cho infer hành động
│   │   │   └── edge_devices/        # Triển khai trên thiết bị biên (Raspberry Pi, v.v.)
│   │   └── tools/
│   │       ├── model_converters/    # Chuyển đổi mô hình sang ONNX hoặc TFLite
│   │       └── benchmarking/        # Đánh giá hiệu năng của mô hình
│   │
│   └── testing/
│       ├── unit_tests/              # Unit tests cho từng module
│       └── integration_tests/       # Test tích hợp toàn bộ pipeline
│
├── results/
│   ├── models/                      # Checkpoint và mô hình đã huấn luyện
│   ├── plots/                       # Hình ảnh kết quả huấn luyện
│   └── tables/                      # Kết quả đánh giá (CSV, Excel)
│
├── config/
│   ├── environment.yml              # File cấu hình môi trường Conda
│   ├── requirements.txt             # Danh sách các gói Python cần thiết
│   └── model_config.yml             # Config cho mô hình HAR
│
└── README.md                        # Hướng dẫn tổng quan dự án
