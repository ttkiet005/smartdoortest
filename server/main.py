import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request, UploadFile, Form, File, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_FOLDER = os.path.join(BASE_DIR, "face_data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/face_data", StaticFiles(directory=FACE_FOLDER), name="face_data")
# Danh sách nhận diện
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for file in os.listdir(FACE_FOLDER):
        if file.lower().endswith(".jpg"):
            path = os.path.join(FACE_FOLDER, file)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(os.path.splitext(file)[0])
            except:
                pass

load_known_faces()

UPLOAD_PASSWORD = "123456"

def load_uids():
    return [os.path.splitext(f)[0] for f in os.listdir(FACE_FOLDER) if f.lower().endswith(".jpg")]

def delete_uid_file(uid: str):
    path = os.path.join(FACE_FOLDER, f"{uid}.jpg")
    if os.path.exists(path):
        os.remove(path)
        try:
            idx = known_face_names.index(uid)
            known_face_names.pop(idx)
            known_face_encodings.pop(idx)
        except ValueError:
            pass
        return True
    return False

# ====================
# Upload panel
# ====================
# ====================
# Upload panel (UI đẹp)
# ====================
@app.get("/upload_panel", response_class=HTMLResponse)
async def upload_panel_get():
    uids = load_uids()

    uid_rows = ""
    if uids:
        for uid in uids:
            img_path = f"/face_data/{uid}.jpg"   # đường hiển thị ảnh
            uid_rows += f"""
            <tr>
                <td>{uid}</td>

                <td style="text-align:center;">
                    <img src='{img_path}' width='80' height='80'
                        style="object-fit:cover;border-radius:8px;border:1px solid #ccc;">
                </td>

                <td>
                    <form method="POST" action="/upload_panel/delete" class="delete-form">
                        <input type="hidden" name="delete_uid" value="{uid}">
                        <input type="password" name="password" placeholder="Password" class="pw-input" required>
                        <button type="submit" class="delete-btn">Xóa</button>
                    </form>
                </td>
            </tr>
            """
    else:
        uid_rows = "<tr><td colspan='3' style='text-align:center;'>Chưa có UID nào.</td></tr>"

    html = f"""
    <html>
    <head>
        <title>Upload Face Data</title>
        <style>
            body {{
                font-family: Arial;
                background: #f7f7f7;
                padding: 30px;
            }}
            .container {{
                max-width: 850px;
                margin: auto;
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.15);
            }}

            h2 {{
                color: #333;
                margin-bottom: 10px;
            }}

            input[type="text"], input[type="password"], input[type="file"] {{
                width: 100%;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #ccc;
                margin-top: 5px;
                margin-bottom: 15px;
                font-size: 15px;
            }}

            button {{
                padding: 10px 18px;
                background: #0078ff;
                border: none;
                color: white;
                font-size: 15px;
                border-radius: 8px;
                cursor: pointer;
            }}

            button:hover {{
                background: #005fcc;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}

            th, td {{
                padding: 12px;
                border-bottom: 1px solid #e5e5e5;
                text-align: left;
            }}

            th {{
                background: #f0f0f0;
            }}

            .delete-btn {{
                background: #ff4444;
            }}
            .delete-btn:hover {{
                background: #cc0000;
            }}

            .delete-form {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}

            .pw-input {{
                width: 150px;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h2>📤 Upload Face Data</h2>
            
            <form method="POST" action="/upload_panel/upload" enctype="multipart/form-data">
                <label>Password:</label>
                <input type="password" name="password" required>

                <label>UID (Tên người):</label>
                <input type="text" name="uid" required>

                <label>Chọn ảnh JPG:</label>
                <input type="file" name="file" required>

                <button type="submit">Upload</button>
            </form>

            <hr style="margin: 30px 0;">

            <h3>📋 Danh sách UID hiện có</h3>

            <table>
                <tr>
                    <th>UID</th>
                    <th>Ảnh</th>
                    <th>Hành động</th>
                </tr>
                {uid_rows}
            </table>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

# POST upload
# ====================
@app.post("/upload_panel/upload", response_class=HTMLResponse)
async def upload_face(
    password: str = Form(...),
    uid: str = Form(...),
    file: UploadFile = File(...)
):
    if password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=403, detail="❌ Sai mật khẩu")

    if not file.filename.lower().endswith(".jpg"):
        raise HTTPException(status_code=400, detail="❌ Chỉ hỗ trợ file .jpg")

    save_path = os.path.join(FACE_FOLDER, f"{uid}.jpg")
    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        image = face_recognition.load_image_file(save_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(uid)
    except:
        pass

    return HTMLResponse(f"✅ Upload thành công: {uid}<br><a href='/upload_panel'>⬅ Quay lại</a>")

# ====================
# POST delete
# ====================
@app.post("/upload_panel/delete", response_class=HTMLResponse)
async def delete_face(password: str = Form(...), delete_uid: str = Form(...)):
    if password != UPLOAD_PASSWORD:
        raise HTTPException(status_code=403, detail="❌ Sai mật khẩu")

    success = delete_uid_file(delete_uid)
    return HTMLResponse(f"{'✅ Đã xóa UID: ' + delete_uid if success else '❌ Không tìm thấy UID'}<br><a href='/upload_panel'>⬅ Quay lại</a>")

# ====================
# Gallery
# ====================
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
#  WIFI CONFIG – LƯU VÀ LẤY CHO ESP32
# ================================
WIFI_CONFIG_FILE = os.path.join(BASE_DIR, "wifi.json")
WIFI_PANEL_PASSWORD = "adminwifi"   # đổi tùy ý

# Tạo file mặc định nếu chưa có
if not os.path.exists(WIFI_CONFIG_FILE):
    with open(WIFI_CONFIG_FILE, "w", encoding="utf8") as f:
        json.dump({"ssid": "", "password": ""}, f, ensure_ascii=False)


def load_wifi():
    with open(WIFI_CONFIG_FILE, "r", encoding="utf8") as f:
        return json.load(f)


def save_wifi(ssid, password):
    with open(WIFI_CONFIG_FILE, "w", encoding="utf8") as f:
        json.dump({"ssid": ssid, "password": password}, f, ensure_ascii=False)


# ================================
# 1) API để ESP32 lấy WiFi
# ================================
@app.get("/wifi_config")
async def get_wifi_config():
    """
    ESP32 gọi API này để lấy SSID + PASSWORD mới nhất
    """
    return load_wifi()


# ================================
# 2) WEB PANEL đổi WiFi (có mật khẩu)
# ================================
@app.get("/wifi_panel", response_class=HTMLResponse)
async def wifi_panel():
    wifi = load_wifi()
    return f"""
    <h2>WiFi Configuration</h2>

    <form method="POST" action="/wifi_panel">
        <label>Admin Password:</label><br>
        <input type="password" name="admin_pw" required><br><br>

        <label>WiFi SSID:</label><br>
        <input type="text" name="ssid" value="{wifi['ssid']}" required><br><br>

        <label>WiFi Password:</label><br>
        <input type="text" name="password" value="{wifi['password']}" required><br><br>

        <button type="submit">Update WiFi</button>
    </form>

    <hr>
    <p><b>Current Saved WiFi:</b><br>
    SSID: {wifi['ssid']}<br>
    Password: {wifi['password']}</p>
    """


@app.post("/wifi_panel", response_class=HTMLResponse)
async def update_wifi(
    admin_pw: str = Form(...),
    ssid: str = Form(...),
    password: str = Form(...)
):
    if admin_pw != WIFI_PANEL_PASSWORD:
        return HTMLResponse("❌ Sai mật khẩu Admin<br><a href='/wifi_panel'>Quay lại</a>")

    save_wifi(ssid, password)

    return HTMLResponse(f"""
        ✅ WiFi đã cập nhật thành công!<br>
        SSID: {ssid}<br>
        Password: {password}<br><br>
        <a href="/wifi_panel">Quay lại</a>
    """)

# ====================
# Nhận diện khuôn mặt
# ====================
@app.post("/recognize")
async def recognize_face(request: Request):
    final_result = "no"
    face_details_for_log = []
    try:
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
    except:
        return PlainTextResponse(content="no", status_code=500)

# ====================
# Root
# ====================
@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "2.5",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        "upload_panel": "/upload_panel",
        "gallery": "/gallery",
        "endpoint_docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
