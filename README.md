# Hệ Thống Benchmark & Sinh Câu Hỏi Trắc Nghiệm Tự Động (Quiz Benchmark System)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Hệ thống này cung cấp một giải pháp toàn diện để sinh câu hỏi trắc nghiệm (MCQ) từ văn bản sử dụng nhiều phương pháp khác nhau, từ các giải thuật dựa trên quy tắc (Rule-based) đến các mô hình ngôn ngữ tiên tiến (T5). Điểm đặc biệt của dự án là khả năng **đánh giá và so sánh hiệu năng** giữa các mô hình thông qua các chỉ số NLP tiêu chuẩn.

---

## 🚀 Tính Năng Chính

*   **Sinh câu hỏi đa dạng**: Hỗ trợ 3 mô hình chính:
    *   **Rule-Based**: Trích xuất dựa trên thực thể (Noun Phrases) và cấu trúc câu.
    *   **TF-IDF**: Lựa chọn từ khóa quan trọng nhất trong văn bản để tạo câu hỏi.
    *   **T5 (Transformer)**: Sử dụng mô hình `valhalla/t5-base-qg-hl` được tinh chỉnh cho bài toán Question Generation.
*   **Đánh giá chuẩn xác (Benchmarking)**: Tích hợp các chỉ số:
    *   **BLEU & ROUGE**: Đánh giá độ tương đồng về ngôn ngữ.
    *   **F1 Score**: Đo lường độ chính xác của câu trả lời.
    *   **Runtime**: Đo lường thời gian xử lý của từng mô hình.
*   **Giao diện Web trực quan**: Thử nghiệm sinh câu hỏi thời gian thực qua Flask.
*   **Trực quan hóa dữ liệu**: Tự động vẽ biểu đồ so sánh (Bar charts, Combined charts).

---

## 🛠️ Cấu Trúc Dự Án

```text
quiz-benchmark-system/
├── app.py              # Web Application (Flask)
├── main.py             # Pipeline sinh dữ liệu hàng loạt
├── run_evaluation.py   # Script chạy benchmark và vẽ biểu đồ
├── src/                # Mã nguồn lõi
│   ├── data_loader.py  # Xử lý tập dữ liệu SQuAD
│   ├── load_model/     # Implementation của các mô hình (Rule, TF-IDF, T5)
│   └── evaluation/     # Các hàm tính toán chỉ số (BLEU, ROUGE, F1...)
├── data/               # Lưu trữ dataset và kết quả sinh
├── outputs/            # Lưu trữ báo cáo benchmark và biểu đồ
├── templates/          # Giao diện Web (HTML)
└── static/             # Assets cho Frontend (CSS, JS)
```

---

## 💻 Cài Đặt

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.9 trở lên.

```bash
# Clone dự án
git clone <repository-url>
cd quiz-benchmark-system

# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình
Tạo file `.env` để giới hạn số lượng dữ liệu chạy thử (tránh tốn tài nguyên):
```env
DATASET_LIMIT=20
```

---

## 📖 Hướng Dẫn Sử Dụng

### Bước 1: Sinh dữ liệu mẫu (Generation)
Chạy pipeline chính để sinh câu hỏi từ dataset SQuAD bằng cả 3 mô hình:
```bash
python main.py
```
*Kết quả sẽ được lưu tại `data/generated_mcqs.json`.*

### Bước 2: Đánh giá hiệu năng (Benchmark)
Sau khi đã có file kết quả ở Bước 1, chạy đánh giá để so sánh các mô hình:
```bash
python run_evaluation.py
```
*Báo cáo JSON và các biểu đồ so sánh (PNG) sẽ được tạo trong thư mục `outputs/`.*

### Bước 3: Chạy Demo Web (Interactive)
Trải nghiệm sinh câu hỏi trực tiếp từ văn bản bất kỳ:
```bash
python app.py
```
Truy cập: `http://localhost:5001` trên trình duyệt.

---

## 📊 Kết Quả Đánh Giá
Hệ thống sẽ tự động tạo các biểu đồ tại `outputs/`:
- `chart_avg_bleu.png`: So sánh độ tương đồng BLEU.
- `chart_total_runtime_seconds.png`: So sánh tốc độ xử lý.
- `benchmark_chart_combined.png`: Tổng quan tất cả các chỉ số.

---

## 📄 Tài Liệu Tham Khảo
- Mô hình T5: [valhalla/t5-base-qg-hl](https://huggingface.co/valhalla/t5-base-qg-hl)
- Dataset: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)

---
*Phát triển bởi Đội ngũ Quiz Benchmark System.*