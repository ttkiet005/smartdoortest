# ===========================
# Base image
# ===========================
FROM python:3.11-slim

# Làm việc trong thư mục /app
WORKDIR /app

# ===========================
# Cài thư viện hệ thống cần thiết
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
# Copy và cài đặt thư viện Python
# ===========================
COPY server/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# Sao chép toàn bộ mã nguồn vào container
# ===========================
COPY server/ /app/

# ===========================
# Expose port
# ===========================
EXPOSE 5000

# ===========================
# Run ứng dụng
# ===========================
CMD ["python", "main.py"]
