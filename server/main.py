import os
from flask import Flask, request, jsonify, render_template_string

import face_recognition
import numpy as np
import cv2

app = Flask(__name__)

# ================================
# CẤU HÌNH
# ================================
FACE_DATA_DIR = "face_data"
UPLOAD_FOLDER = "uploads"
UPLOAD_PASSWORD = "123456"   # ✅ Mật khẩu truy cập trang upload ảnh

os.makedirs(FACE_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================================
# HÀM LOAD ẢNH KHUÔN MẶT TỪ FOLDER face_data
# ==========================================================
def load_face_data():
    known_encodings = {}
    for filename in os.listdir(FACE_DATA_DIR):
        if filename.lower().endswith(".jpg"):
            uid = filename.split(".")[0]
            img_path = os.path.join(FACE_DATA_DIR, filename)

            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings[uid] = encodings[0]
                print(f"[LOAD] Loaded: {uid}")
            else:
                print(f"[WARN] Không tìm thấy khuôn mặt trong ảnh {filename}")

    return known_encodings


known_faces = load_face_data()


# ==========================================================
# ROUTE: TRANG UPLOAD PANEL (dùng Google Chrome hoặc Blynk)
# ==========================================================
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload Face Data</title>
</head>
<body>
    <h2>Upload Face Data</h2>
    <form method="POST" enctype="multipart/form-data">
        <label>Mật khẩu:</label><br>
        <input type="password" name="password"><br><br>

        <label>UID (mã thẻ RFID):</label><br>
        <input type="text" name="uid"><br><br>

        <label>Chọn ảnh JPG:</label><br>
        <input type="file" name="file"><br><br>

        <button type="submit">Upload</button>
    </form>
</body>
</html>
"""


@app.route("/upload_panel", methods=["GET", "POST"])
def upload_panel():
    if request.method == "GET":
        return render_template_string(html_form)

    # Check password
    if request.form.get("password") != UPLOAD_PASSWORD:
        return "❌ Sai mật khẩu", 403

    uid = request.form.get("uid")
    file = request.files.get("file")

    if not uid or not file:
        return "❌ Thiếu UID hoặc file", 400

    if not file.filename.lower().endswith(".jpg"):
        return "❌ Chỉ hỗ trợ file .jpg", 400

    save_path = os.path.join(FACE_DATA_DIR, f"{uid}.jpg")
    file.save(save_path)

    global known_faces
    known_faces = load_face_data()  # ✅ Tự động reload dữ liệu

    return f"✅ Thành công!<br>Ảnh đã lưu tại: {save_path}"


# ==========================================================
# ROUTE: ESP32-CAM GỬI ẢNH LÊN SERVER ĐỂ NHẬN DIỆN
# ==========================================================
@app.route("/recognize", methods=["POST"])
def recognize():
    uid = request.args.get("uid")  # UID do ESP32 (MCU) gửi sang ESP32-CAM
    if not uid:
        return jsonify({"result": "error", "msg": "Missing UID"}), 400

    file = request.data
    if not file:
        return jsonify({"result": "error", "msg": "Missing image"}), 400

    img_path = os.path.join(UPLOAD_FOLDER, "latest.jpg")
    with open(img_path, "wb") as f:
        f.write(file)

    # Load ảnh chụp
    img = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, face_locations)

    if len(encodings) == 0:
        return "no_face"

    # Lấy dữ liệu khuôn mặt theo UID
    if uid not in known_faces:
        return "no_uid"

    match = face_recognition.compare_faces([known_faces[uid]], encodings[0])[0]

    return "yes" if match else "no"


# ==========================================================
# ROUTE KIỂM TRA SERVER
# ==========================================================
@app.route("/")
def home():
    return "✅ SmartDoor Server is running"


# ==========================================================
# START SERVER
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
