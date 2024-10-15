# Human_Activity_Recognition
This project focuses on developing a Human Activity Recognition (HAR) system using machine learning and deep learning techniques. The goal is to classify and recognize human activities (e.g., walking, sitting, running) based on sensor data or multimedia inputs like images or videos. The system is built with modular components, making it flexible for experimentation, data processing, model training, and deployment.

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
