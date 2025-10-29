import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="SmartDoor Face Recognition API")

# ================================
# Cấu hình thư mục
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_FOLDER = os.path.join(BASE_DIR, "face_data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

# Static files (để xem ảnh upload)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Giao diện web
templates = Jinja2Templates(directory=TEMPLATE_FOLDER)

# ================================
# Tải khuôn mặt đã lưu
# ================================
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Đọc và mã hóa tất cả khuôn mặt trong thư mục face_data"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    print("=" * 50)
    print(f"[INFO] Loading known faces from: {FACE_FOLDER}")
    print("=" * 50)

    for file in os.listdir(FACE_FOLDER):
        path = os.path.join(FACE_FOLDER, file)
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    name = os.path.splitext(file)[0]
                    known_face_names.append(name)
                    print(f"   ✓ Loaded: {name}")
                else:
                    print(f"   ✗ No face found in: {file}")
            except Exception as e:
                print(f"   ✗ Error loading {file}: {e}")

    print("=" * 50)
    print(f"[INFO] Total faces loaded: {len(known_face_names)}")
    if known_face_names:
        print(f"[INFO] Names: {', '.join(known_face_names)}")
    else:
        print("⚠ WARNING: No faces loaded! Add images to 'face_data' folder.")
    print("=" * 50)

load_known_faces()

# ================================
# Giao diện upload ảnh
# ================================
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# ================================
# Xử lý upload từ giao diện web
# ================================
@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile):
    try:
        content = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")
        with open(image_path, "wb") as f:
            f.write(content)

        # So sánh khuôn mặt
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        matched_name = "Unknown"
        confidence = 0
        result = "no"

        if face_encodings and known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            if best_distance < 0.5:
                matched_name = known_face_names[best_match_index]
                confidence = (1 - best_distance) * 100
                result = "yes"

        # Trả về giao diện kết quả
        image_url = f"/uploads/{os.path.basename(image_path)}"
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result,
            "name": matched_name,
            "confidence": round(confidence, 2),
            "image_url": image_url,
            "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        })

    except Exception as e:
        return HTMLResponse(f"<h3>Error: {str(e)}</h3>")

# ================================
# API kiểm tra trạng thái
# ================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        "upload_page": "/upload",
        "docs": "/docs"
    }

# ================================
# Chạy server
# ================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
