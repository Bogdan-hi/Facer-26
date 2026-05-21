import os
import sys
import cv2
import math
import shutil
import tempfile
import numpy as np
import subprocess
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# CONFIG

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = str(BASE_DIR / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MASK_MODEL_PATH = str(BASE_DIR / "models" / "mask_detection.pt")
SUNGLASSES_MODEL_PATH = str(BASE_DIR / "models" / "sunglasses_detection.pt")
SEG_MODEL = str(BASE_DIR / "models" / "selfie_segmenter.tflite")
FACE_MODEL_ORIG = str(BASE_DIR / "models" / "blaze_face_short_range.tflite")

# FASTAPI
app = FastAPI()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# LOAD MODELS
mask_model = YOLO(MASK_MODEL_PATH)
sunglasses_model = YOLO(SUNGLASSES_MODEL_PATH)

# MEDIAPIPE
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# HELPERS
def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_face_angle(landmarks, w, h):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    return abs(angle)

def check_occlusion(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness > 40

# FACE DETECTOR (модель загружается в память)

# Читаем модель в bytes
with open(FACE_MODEL_ORIG, 'rb') as f:
    face_model_buffer = f.read()

face_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_buffer=face_model_buffer),
    running_mode=vision.RunningMode.IMAGE
)
face_detector = vision.FaceDetector.create_from_options(face_options)


# VALIDATION


def validate_face(image):

    h, w, _ = image.shape

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # 1. STRICT FACE COUNT


    detections = face_detector.detect(mp_image)

    valid_faces = []

    if detections.detections:

        for det in detections.detections:

            score = det.categories[0].score

            if score > 0.5:
                valid_faces.append(det)

    if len(valid_faces) == 0:
        return False, "Лицо не найдено"

    if len(valid_faces) != 1:
        return False, "На фото должен быть только один человек"


    # FACE MESH


    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False, "Не удалось определить лицо"

    face_landmarks = results.multi_face_landmarks[0]


    # 2. HEAD ANGLE


    angle = calculate_face_angle(
        face_landmarks.landmark,
        w,
        h
    )

    if angle > 10:
        return False, "Держите голову ровно"


    # 3. BRIGHTNESS


    brightness = calculate_brightness(image)

    if brightness < 70:
        return False, "Недостаточное освещение"

    if brightness > 220:
        return False, "Слишком яркое освещение"


    # FACE ROI


    xs = []
    ys = []

    for lm in face_landmarks.landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(w, max(xs))
    y2 = min(h, max(ys))

    face_crop = image[y1:y2, x1:x2]


    # 4. BLUR / OCCLUSION


    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score < 45:
        return False, "Лицо размыто или перекрыто"


    # 5. MASK DETECTION


    mask_results = mask_model.predict(
        face_crop,
        conf=0.55,
        verbose=False
    )

    for r in mask_results:

        if r.boxes is None:
            continue

        for cls_id in r.boxes.cls.tolist():

            cls_id = int(cls_id)

            class_name = mask_model.names[cls_id]

            if class_name == "face_mask":
                return False, "Снимите маску"


    # 6. SUNGLASSES DETECTION

    sunglasses_results = sunglasses_model.predict(
        face_crop,
        conf=0.1,      # чуть ниже, чтобы точно ловить очки
        verbose=False
    )

    for r in sunglasses_results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])          # класс найденного объекта
            conf = float(box.conf[0])         # уверенность

            # Проверяем только нужный класс (0) и достаточную уверенность
            if cls_id == 0 and conf > 0.1:
                return False, "Снимите солнцезащитные очки"
    return True, "Фото успешно загружено"

# ROUTES

@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    ext = os.path.splitext(file.filename)[1]

    filename = f"{uuid.uuid4()}{ext}"

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = cv2.imread(file_path)

    if image is None:
        return JSONResponse({
            "success": False,
            "message": "Ошибка чтения изображения"
        })

    success, message = validate_face(image)

    return JSONResponse({
        "success": success,
        "message": message
    })

@app.post("/capture")
async def capture_photo():
    result_file = os.path.join(UPLOAD_DIR, "capture_result.txt")
    # Удалим старый файл, если есть
    if os.path.exists(result_file):
        os.remove(result_file)

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "camera_capture.py")],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )

    # Проверяем файл-результат
    if not os.path.exists(result_file):
        return JSONResponse({
            "success": False,
            "message": "Фото не было сделано (нет результата)"
        })

    with open(result_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Удаляем файл, чтобы не захламляться
    os.remove(result_file)

    if content == "CANCEL":
        return JSONResponse({
            "success": False,
            "message": "Съёмка отменена"
        })
    elif content.startswith("ERROR"):
        return JSONResponse({
            "success": False,
            "message": content
        })
    elif os.path.exists(content):
        # Это путь к фото
        image = cv2.imread(content)
        if image is None:
            return JSONResponse({
                "success": False,
                "message": "Ошибка чтения фотографии"
            })
        success, message = validate_face(image)
        filename = os.path.basename(content)
        return JSONResponse({
            "success": success,
            "message": message,
            "image": f"/uploads/{filename}"
        })
    else:
        return JSONResponse({
            "success": False,
            "message": "Файл фотографии не найден по пути: " + content
        })

# START

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)