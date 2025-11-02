import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request, UploadFile, Form, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ================================
# Cấu hình thư mục
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_FOLDER = os.path.join(BASE_DIR, "face_data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

# Tự động tạo thư mục nếu chưa có
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Mount thư mục static cho ảnh upload
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# ================================
# Tải khuôn mặt đã lưu
# ================================
known_face_encodings = []
known_face_names = []

print("=" * 60)
print(f"[INFO] Loading known faces from: {FACE_FOLDER}")
print("=" * 60)

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
            print(f"   ✗ Error loading {file}: {str(e)}")

print("=" * 60)
print(f"[INFO] Total faces loaded: {len(known_face_names)}")
if len(known_face_names) > 0:
    print(f"[INFO] Names: {', '.join(known_face_names)}")
else:
     print("⚠ WARNING: No faces loaded! Add images to 'face_data' folder.")
print("=" * 60 + "\n")

# ================================
# UPLOAD PANEL
UPLOAD_PASSWORD = "123456"

os.makedirs(FACE_FOLDER, exist_ok=True)

# Load danh sách UID hiện có
def load_uids():
    return [os.path.splitext(f)[0] for f in os.listdir(FACE_FOLDER) if f.lower().endswith(".jpg")]

# Hàm xóa UID
def delete_uid(uid: str):
    path = os.path.join(FACE_FOLDER, f"{uid}.jpg")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

@app.get("/upload_panel", response_class=HTMLResponse)
async def upload_panel_get():
    uids = load_uids()
    uid_list_html = "<ul>"
    for uid in uids:
        uid_list_html += f"""
        <li>{uid} 
            <form method="POST" style="display:inline-block;">
                <input type="hidden" name="delete_uid" value="{uid}">
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Xóa</button>
            </form>
        </li>"""
    uid_list_html += "</ul>" if uids else "<p>Chưa có UID nào.</p>"

    html = f"""
    <h2>Upload Face Data</h2>
    <form method="POST" enctype="multipart/form-data">
        <label>Password:</label><br>
        <input type="password" name="password" required><br><br>

        <label>UID (Tên người):</label><br>
        <input type="text" name="uid" required><br><br>

        <label>Chọn ảnh JPG:</label><br>
        <input type="file" name="file" required><br><br>

        <button type="submit">Upload</button>
    </form>
    <hr>
    <h3>Danh sách UID hiện có</h3>
    {uid_list_html}
    """
    return HTMLResponse(html)

@app.post("/upload_panel", response_class=HTMLResponse)
async def upload_panel_post(
    password: str = Form(...),
    uid: str = Form(None),
    file: UploadFile = File(None),
    delete_uid_field: str = Form(None)
):
    # Xác thực password
    if password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=403, detail="❌ Sai mật khẩu")

    # Xử lý xóa UID
    if delete_uid_field:
        success = delete_uid(delete_uid_field)
        return HTMLResponse(f"{'✅ Đã xóa UID: ' + delete_uid_field if success else '❌ Không tìm thấy UID'}<br><a href='/upload_panel'>⬅ Quay lại</a>")

    # Xử lý upload
    if not uid or not file:
        raise HTTPException(status_code=400, detail="❌ Thiếu UID hoặc file")

    if not file.filename.lower().endswith(".jpg"):
        raise HTTPException(status_code=400, detail="❌ Chỉ hỗ trợ file .jpg")

    save_path = os.path.join(FACE_FOLDER, f"{uid}.jpg")
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Reload danh sách UID
    return HTMLResponse(f"✅ Upload thành công: {uid}<br><a href='/upload_panel'>⬅ Quay lại</a>")
# GALLERY
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
        <br><a href="/upload_panel">⬅ Quay lại upload</a>
      </body>
    </html>
    """)

# ================================
# NHẬN DIỆN KHUÔN MẶT
# ================================
@app.post("/recognize")
async def recognize_face(request: Request):
    final_result = "no"
    face_details_for_log = []

    try:
        # 1. Nhận dữ liệu ảnh
        image_bytes = await request.body()
        if len(image_bytes) == 0:
            return PlainTextResponse(content="no", status_code=400)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return PlainTextResponse(content="no", status_code=400)

        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_raw.jpg")
        cv2.imwrite(image_path, frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        if len(face_locations) == 0 or len(known_face_encodings) == 0:
            return PlainTextResponse(content="no")

        THRESHOLD = 0.5
        for i, face_encoding in enumerate(face_encodings):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = float(face_distances[best_match_index])

            name = "Unknown"
            confidence = 0.0
            if best_distance < THRESHOLD:
                name = known_face_names[best_match_index]
                confidence = (1 - best_distance) * 100
                final_result = "yes"

            face_details_for_log.append({
                "name": name,
                "confidence": round(float(confidence), 2),
                "distance": round(float(best_distance), 4),
                "threshold": float(THRESHOLD),
                "match": bool(best_distance < THRESHOLD)
            })

        # Ghi log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "result_sent": final_result,
            "face_count": len(face_locations),
            "faces_detail": face_details_for_log,
            "image_path": image_path,
        }
        detail_log_path = os.path.join(LOG_FOLDER, "recognition_log.jsonl")
        with open(detail_log_path, "a", encoding="utf-8") as log:
            log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return PlainTextResponse(content=final_result)

    except Exception as e:
        return PlainTextResponse(content="no", status_code=500)

# ================================
# KIỂM TRA TRẠNG THÁI SERVER
# ================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "2.2",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        "upload_panel": "/upload_panel",
        "gallery": "/gallery",
        "endpoint_docs": "/docs"
    }

# ================================
# CHẠY SERVER
# ================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
