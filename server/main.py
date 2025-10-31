import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request, UploadFile
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

for folder in [FACE_FOLDER, UPLOAD_FOLDER, LOG_FOLDER, TEMPLATE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
templates = Jinja2Templates(directory=TEMPLATE_FOLDER)

# ================================
# Tải khuôn mặt đã lưu
# ================================
known_face_encodings = []
known_face_names = []


def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    print("=" * 50)
    print(f"[INFO] Loading known faces from: {FACE_FOLDER}")
    print("=" * 50)

    for file in os.listdir(FACE_FOLDER):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(FACE_FOLDER, file)
        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])
                print(f"   ✓ Loaded: {file}")
            else:
                print(f"   ✗ No face in: {file}")
        except Exception as e:
            print(f"   ✗ Error loading {file}: {e}")

    print(f"[INFO] Total faces loaded: {len(known_face_names)}")
    print("=" * 50)


load_known_faces()

# ================================
# Trang upload web
# ================================
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile):
    """Upload ảnh từ form web"""
    try:
        content = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")

        with open(image_path, "wb") as f:
            f.write(content)

        # Nhận diện khuôn mặt
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return HTMLResponse("<h3>Không thể đọc ảnh</h3>")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        matched_name, confidence, result = "Unknown", 0, "no"

        if encodings and known_face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, encodings[0])
            best_match = np.argmin(distances)
            best_distance = distances[best_match]
            if best_distance < 0.5:
                matched_name = known_face_names[best_match]
                confidence = (1 - best_distance) * 100
                result = "yes"

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
# API cho ESP32: trả về "yes"/"no"
# ================================
@app.post("/recognize")
async def recognize_face(request: Request):
    """
    ESP32 gửi ảnh binary (application/octet-stream)
    -> Server trả về 'yes' nếu khớp, 'no' nếu không
    -> Ảnh vẫn được lưu vào thư mục uploads/ để xem trong /gallery
    """
    try:
        image_bytes = await request.body()
        if not image_bytes:
            return PlainTextResponse("no")

        # Giải mã ảnh
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("   ✗ Invalid image format")
            return PlainTextResponse("no")

        # ✅ Lưu ảnh lại trong thư mục uploads/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"{timestamp}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        cv2.imwrite(image_path, frame)
        print(f"   ✓ Saved uploaded image: {image_name}")

        # Nhận diện khuôn mặt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        result = "no"
        matched_name, confidence = None, 0

        if encodings and known_face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, encodings[0])
            best_match = np.argmin(distances)
            best_distance = distances[best_match]

            if best_distance < 0.5:
                matched_name = known_face_names[best_match]
                confidence = (1 - best_distance) * 100
                result = "yes"

        # Ghi log JSONL
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "name": matched_name,
            "confidence": round(confidence, 2),
            "image": f"/uploads/{image_name}"
        }
        with open(os.path.join(LOG_FOLDER, "recognition_log.jsonl"), "a", encoding="utf-8") as log:
            log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] → {result} ({matched_name or 'Unknown'})")
        return PlainTextResponse(result)

    except Exception as e:
        print(f"[ERROR] {e}")
        return PlainTextResponse("no")


# ================================
# Gallery ảnh upload
# ================================
@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    images_html = ""

    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            url = f"/uploads/{f}"
            images_html += f"""
            <div style='display:inline-block;margin:10px;text-align:center;'>
                <img src='{url}' width='200' style='border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.3)'>
                <p>{f}</p>
            </div>
            """

    return HTMLResponse(f"""
    <html>
      <head><title>Gallery</title></head>
      <body style='font-family:Arial;text-align:center;padding:30px;'>
        <h2>📸 Ảnh đã upload</h2>
        {images_html or '<p>Chưa có ảnh nào.</p>'}
        <br><a href="/upload">⬅ Quay lại upload</a>
      </body>
    </html>
    """)


# ================================
# Trang root kiểm tra server
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
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
