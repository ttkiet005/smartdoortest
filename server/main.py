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
from fastapi import Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from fastapi import Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
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
# Tải khuôn mặt
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
@app.post("/upload_panel_browser")
async def upload_panel_post(
    uid: str = Form(...),
    password: str = Form(...),
    file: UploadFile = Form(...)
):
    """
    Nhận file upload từ browser hoặc Blynk
    """
    UPLOAD_PANEL_PASSWORD = "123456"

    # 1️⃣ Kiểm tra mật khẩu
    if password != UPLOAD_PANEL_PASSWORD:
        raise HTTPException(status_code=403, detail="❌ Sai mật khẩu")

    # 2️⃣ Kiểm tra định dạng file
    if not file.filename.lower().endswith(".jpg"):
        raise HTTPException(status_code=400, detail="❌ Chỉ hỗ trợ file .jpg")

    # 3️⃣ Lưu file vào FACE_FOLDER
    save_path = os.path.join(FACE_FOLDER, f"{uid}.jpg")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 4️⃣ Reload dữ liệu khuôn mặt
    load_known_faces()

    # 5️⃣ Trả kết quả
    return HTMLResponse(f"""
    <h3>✅ Upload thành công!</h3>
    <p>UID: {uid}</p>
    <p>Ảnh đã lưu tại: {save_path}</p>
    <a href="/upload_panel_browser">⬅ Quay lại upload</a>
    """)
# ============================================================
# ✅ LOG SERIAL TỪ ESP32 — API /logs
# ============================================================


logs = []  # lưu 5000 dòng


@app.post("/logs")
async def logs_post(request: Request):
    """
    ESP32 gửi log vào đây bằng JSON:
    {"log": "nội dung log"}
    """
    global logs
    data = await request.json()
    line = data.get("log")

    if line:
        logs.append(line)
        if len(logs) > 5000:
            logs = logs[-5000:]

    return JSONResponse({"status": "ok"})


@app.get("/logs")
async def logs_get():
    """Trả log dạng JSON"""
    return JSONResponse({"logs": logs})


@app.get("/logs/text")
async def logs_text():
    """Trả log dạng text giống Serial Monitor"""
    return PlainTextResponse("\n".join(logs))


# ================================
# Trang upload ảnh web
# ================================
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile):
    try:
        content = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")

        with open(image_path, "wb") as f:
            f.write(content)

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
# API ESP32 gửi ảnh nhận diện
# ================================
@app.post("/recognize")
async def recognize_face(request: Request):
    try:
        image_bytes = await request.body()
        if not image_bytes:
            return PlainTextResponse("no")

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return PlainTextResponse("no")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"{timestamp}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        cv2.imwrite(image_path, frame)

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

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "name": matched_name,
            "confidence": round(confidence, 2),
            "image": f"/uploads/{image_name}"
        }

        with open(os.path.join(LOG_FOLDER, "recognition_log.jsonl"), "a", encoding="utf-8") as log:
            log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return PlainTextResponse(result)

    except Exception:
        return PlainTextResponse("no")


# ================================
# Gallery ảnh
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


@app.get("/")
async def root():
    return {
        "status": "online",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        "upload_page": "/upload",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
