FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ❗ Chỉ sửa dòng này
COPY . /app/

EXPOSE 5000

CMD ["python", "server/main.py"]
