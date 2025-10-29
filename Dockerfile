FROM python:3.11-slim

# --- Cài tất cả công cụ cần thiết để build dlib và face_recognition ---
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# --- Thiết lập thư mục làm việc ---
WORKDIR /app

# --- Sao chép mã nguồn ---
COPY server/ /app/
COPY requirements.txt /app/

# --- Cài thư viện Python ---
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Chạy ứng dụng ---
CMD ["python", "main.py"]
