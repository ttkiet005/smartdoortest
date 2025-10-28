import uvicorn
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

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

# ================================
# Tải khuôn mặt đã lưu
# ================================
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
            print(f"   ✗ Error loading {file}: {str(e)}")

print("=" * 50)
print(f"[INFO] Total faces loaded: {len(known_face_names)}")
if len(known_face_names) > 0:
    print(f"[INFO] Names: {', '.join(known_face_names)}")
else:
     print("⚠ WARNING: No faces loaded! Add images to 'face_data' folder.")
print("=" * 50)


# ================================
# API nhận diện
# ================================
@app.post("/recognize")
async def recognize_face(request: Request):
    """
    Nhận ảnh thô (raw bytes) từ ESP32-CAM, nhận diện
    và trả về "yes" hoặc "no".
    """
    final_result = "no"  # Mặc định là không hợp lệ
    face_details_for_log = []
    
    try:
        # 1. Nhận dữ liệu ảnh (raw bytes)
        image_bytes = await request.body()
        
        if len(image_bytes) == 0:
            return PlainTextResponse(content="no", status_code=400)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New request, {len(image_bytes)} bytes")

        # 2. Chuyển đổi bytes thành ảnh OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("   ✗ Invalid image format")
            return PlainTextResponse(content="no", status_code=400)
        
        # 3. Lưu ảnh gốc
        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_raw.jpg")
        cv2.imwrite(image_path, frame)
        print(f"   Saved raw to: {image_path}")

        # 4. Tiền xử lý (tùy chọn) và chuyển sang RGB
        # frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10) # Bật nếu ảnh quá tối
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 5. Tìm và nhận diện khuôn mặt
        print("   Detecting faces...")
        face_locations = face_recognition.face_locations(rgb, model="hog") # "hog" nhanh hơn "cnn"
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        
        print(f"   Found {len(face_locations)} face(s)")

        if len(face_locations) > 0 and len(known_face_encodings) > 0:
            for i, face_encoding in enumerate(face_encodings):
                # So sánh
                face_distances = face_recognition.face_distance(
                    known_face_encodings, 
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                name = "Unknown"
                confidence = 0
                
                # Ngưỡng nhận diện (0.5 là khá chặt)
                if best_distance < 0.5:
                    name = known_face_names[best_match_index]
                    confidence = (1 - best_distance) * 100
                    final_result = "yes"  # CHỈ CẦN 1 KHUÔN MẶT HỢP LỆ
                
                print(f"     Face {i+1}: {name} (Dist: {best_distance:.4f})")

                # Lưu chi tiết để ghi log
                face_details_for_log.append({
                    "name": name,
                    "confidence": round(confidence, 2),
                    "distance": round(best_distance, 4)
                })
        
        # 6. Ghi log (bất kể kết quả)
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
        
        # 7. Trả về kết quả đơn giản cho ESP32
        print(f"   Sending response: {final_result}")
        print("=" * 50)
        return PlainTextResponse(content=final_result)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return PlainTextResponse(content="no", status_code=500)

# ================================
# API kiểm tra trạng thái
# ================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "known_faces_count": len(known_face_names),
        "known_names": known_face_names,
        "endpoint_docs": "/docs"
    }

# ================================
# Chạy server
# ================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Face Recognition Server (v2)")
    print(f"   Watching folder: {FACE_FOLDER}")
    print(f"   API Endpoint: http://0.0.0.0:5000/recognize")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)