import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="SmartDoor Face Recognition API")

# ================================
# Cấu hình thư mục
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_FOLDER = os.path.join(BASE_DIR, "face_data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Cho phép truy cập ảnh qua URL
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

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
# API: Nhận diện khuôn mặt
# ================================
@app.post("/recognize")
async def recognize_face(request: Request):
    MAX_IMAGE_SIZE = 3 * 1024 * 1024  # 3MB

    try:
        image_bytes = await request.body()
        if len(image_bytes) == 0:
            return PlainTextResponse("no", status_code=400)
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return PlainTextResponse("no", status_code=413)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New request: {len(image_bytes)} bytes")

        # Giải mã ảnh
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("   ✗ Invalid image format")
            return PlainTextResponse("no", status_code=400)

        # Lưu ảnh upload
        image_name = f"{timestamp}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        cv2.imwrite(image_path, frame)
        print(f"   Saved uploaded image: {image_path}")

        # Nhận diện khuôn mặt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        print(f"   Found {len(face_locations)} face(s)")

        result = {
            "status": "no",
            "timestamp": datetime.now().isoformat(),
            "name": None,
            "confidence": 0,
            "image_url": f"/uploads/{image_name}",
        }

        if len(face_encodings) > 0 and len(known_face_encodings) > 0:
            for face_encoding in face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]

                if best_distance < 0.5:
                    name = known_face_names[best_match_index]
                    confidence = (1 - best_distance) * 100
                    result.update({
                        "status": "yes",
                        "name": name,
                        "confidence": round(confidence, 2),
                    })
                    print(f"   ✓ Match found: {name} ({confidence:.2f}%)")
                    break

        # Ghi log JSONL
        log_entry = {
            "timestamp": result["timestamp"],
            "status": result["status"],
            "name": result["name"],
            "confidence": result["confidence"],
            "image": result["image_url"],
        }
        with open(os.path.join(LOG_FOLDER, "recognition_log.jsonl"), "a", encoding="utf-8") as log:
            log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Trả kết quả JSON
        return JSONResponse(result)

    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ================================
# API: Reload khuôn mặt thủ công
# ================================
@app.get("/reload_faces")
async def reload_faces():
    load_known_faces()
    return {"status": "reloaded", "faces_count": len(known_face_names)}


# ================================
# API: Kiểm tra trạng thái
# ================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        "upload_folder": "/uploads",
        "docs": "/docs",
    }


# ================================
# Chạy server
# ================================
if __name__ == "__main__":
    print("\n🚀 Starting Face Recognition Server (v3.0)")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
