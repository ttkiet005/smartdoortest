import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

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
    final_result = "no"
    face_details_for_log = []
    MAX_IMAGE_SIZE = 3 * 1024 * 1024  # 3MB

    try:
        image_bytes = await request.body()

        if len(image_bytes) == 0:
            return PlainTextResponse("no", status_code=400)
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return PlainTextResponse("no", status_code=413)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New request: {len(image_bytes)} bytes")

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("   ✗ Invalid image format")
            return PlainTextResponse("no", status_code=400)

        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_raw.jpg")
        cv2.imwrite(image_path, frame)
        print(f"   Saved raw image: {image_path}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        print(f"   Found {len(face_locations)} face(s)")

        if len(face_locations) > 0 and len(known_face_encodings) > 0:
            for i, face_encoding in enumerate(face_encodings):
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]

                name = "Unknown"
                confidence = 0

                if best_distance < 0.5:
                    name = known_face_names[best_match_index]
                    confidence = (1 - best_distance) * 100
                    final_result = "yes"

                print(f"     Face {i+1}: {name} (Dist: {best_distance:.4f})")

                face_details_for_log.append({
                    "name": name,
                    "confidence": round(confidence, 2),
                    "distance": round(best_distance, 4)
                })

        # Ghi log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "result_sent": final_result,
            "face_count": len(face_locations),
            "faces_detail": face_details_for_log,
            "image_path": image_path,
        }
        with open(os.path.join(LOG_FOLDER, "recognition_log.jsonl"), "a", encoding="utf-8") as log:
            log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"   → Result: {final_result}")
        print("=" * 50)
        return PlainTextResponse(final_result)

    except Exception as e:
        print(f"[ERROR] {e}")
        return PlainTextResponse("no", status_code=500)

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
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return PlainTextResponse("OK")

# ================================
# Chạy server
# ================================
if __name__ == "__main__":
    print("\n🚀 Starting Face Recognition Server (v2.1)")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
