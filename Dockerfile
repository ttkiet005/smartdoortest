FROM python:3.11-slim

# --- Cài đặt thư viện hệ thống cần cho dlib & face_recognition ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# --- Sao chép mã nguồn ---
WORKDIR /app
COPY server/ /app/
COPY requirements.txt /app/

# --- Cài thư viện Python ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Chạy chương trình ---
CMD ["python", "main.py"]
