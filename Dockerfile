# Base image
FROM python:3.11-slim

# Làm việc trong /app/server (vì main.py nằm trong server/)
WORKDIR /app/server

# Copy requirements và cài đặt
COPY server/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . /app

# Expose port
EXPOSE 5000

# Chạy app
CMD ["python", "server/main.py"]
