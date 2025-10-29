# ===========================
# Base image
# ===========================
FROM python:3.11-slim

# Làm việc trong thư mục /app/server (vì main.py nằm ở đây)
WORKDIR /app/server

# ===========================
# Cài các thư viện hệ thống cần thiết
# ===========================
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# ===========================
# Sao chép file yêu cầu trước để tận dụng cache
# ===========================
COPY requirements.txt /app/requirements.txt

# ===========================
# Cài thư viện Python
# ===========================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# ===========================
# Sao chép toàn bộ mã nguồn
# ===========================
COPY . /app

# ===========================
# Mở cổng
# ===========================
EXPOSE 5000

# ===========================
# Chạy ứng dụng
# ===========================
CMD ["python", "main.py"]
