import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from pathlib import Path
import os
import sys
import uuid
import tempfile
import shutil


# АБСОЛЮТНЫЕ ПУТИ

BASE_DIR = Path(__file__).resolve().parent
SEG_MODEL = str(BASE_DIR / "models" / "selfie_segmenter.tflite")
FACE_MODEL = str(BASE_DIR / "models" / "blaze_face_short_range.tflite")

# НАСТРОЙКИ

SAVE_DIR = str(BASE_DIR / "uploads")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "captured_face.jpg")
# Временный ASCII-путь для OpenCV
TEMP_SAVE_PATH = os.path.join(tempfile.gettempdir(), "captured_face.jpg")

BG_COLOR = (0, 200, 255)

RESULT_FILE = os.path.join(SAVE_DIR, "capture_result.txt")

# ЗАГРУЗКА МОДЕЛЕЙ ЧЕРЕЗ БУФЕР (без использования пути)

with open(SEG_MODEL, 'rb') as f:
    seg_buffer = f.read()
with open(FACE_MODEL, 'rb') as f:
    face_buffer = f.read()

seg_options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_buffer=seg_buffer),
    running_mode=vision.RunningMode.VIDEO,
    output_category_mask=True
)
segmentor = vision.ImageSegmenter.create_from_options(seg_options)

face_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_buffer=face_buffer),
    running_mode=vision.RunningMode.VIDEO
)
face_detector = vision.FaceDetector.create_from_options(face_options)


# ПАРАМЕТРЫ ВАЛИДАЦИИ

POSITION_TOLERANCE = 0.3
SIZE_TOLERANCE = 0.4
SMOOTHING = 0.8
dark_alpha = 0.0
score_smooth = 0.0


# КАМЕРА

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # Сообщение об ошибке попадет в stdout и будет возвращено сервером
    print("Ошибка: не удалось открыть камеру")
    exit(1)

timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: потерян кадр с камеры")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


    # СЕГМЕНТАЦИЯ

    seg_result = segmentor.segment_for_video(mp_image, timestamp)
    mask = np.squeeze(seg_result.category_mask.numpy_view())

    if mask.max() > 1:
        mask = mask / 255.0
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (w, h))

    condition = mask > 0.1
    mask_f = mask.astype(np.float32)

    if np.mean(mask_f) > 0.5:
        mask_f = 1.0 - mask_f

    mask_f = cv2.GaussianBlur(mask_f, (21, 21), 0)
    mask_f = np.clip((mask_f - 0.1) * 1.3, 0, 1)
    mask_f = mask_f[..., None]


    # РАЗМЫТИЕ ФОНА

    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    output = frame * mask_f + blurred * (1 - mask_f)
    output = output.astype(np.uint8)


    # КВАДРАТ (ROI)

    center = (w // 2, h // 2)
    half_size = int(min(w, h) * 0.25)

    x1 = center[0] - half_size
    y1 = center[1] - half_size
    x2 = center[0] + half_size
    y2 = center[1] + half_size

    square_color = (
        0,
        int(255 * score_smooth),
        int(255 * (1 - score_smooth))
    )
    cv2.rectangle(output, (x1, y1), (x2, y2), square_color, 3)


    # ДЕТЕКЦИЯ ЛИЦА

    face_result = face_detector.detect_for_video(mp_image, timestamp)

    score = 0.0
    face_inside = False

    if face_result.detections:
        det = face_result.detections[0]
        bbox = det.bounding_box
        x, y, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        fx = x + bw / 2
        fy = y + bh / 2

        dx = (fx - center[0]) / half_size
        dy = (fy - center[1]) / half_size
        position_error = max(abs(dx), abs(dy))

        face_size = (bw + bh) / 2
        target_size = half_size * 2 * 0.9
        size_error = abs(face_size - target_size) / target_size

        pos_score = max(0, 1 - position_error / POSITION_TOLERANCE)
        size_score = max(0, 1 - size_error / SIZE_TOLERANCE)
        score = pos_score * size_score

        score_smooth = SMOOTHING * score_smooth + (1 - SMOOTHING) * score
        face_inside = score_smooth > 0.75

        color = (0, int(255 * score_smooth), int(255 * (1 - score_smooth)))
        cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)


    # ПЛАВНОЕ ЗАТЕМНЕНИЕ

    target_dark = 0.6 if not face_inside else 0.0
    dark_alpha = 0.9 * dark_alpha + 0.1 * target_dark

    if dark_alpha > 0.01:
        dark_layer = np.zeros_like(output, dtype=np.uint8)
        output = cv2.addWeighted(output, 1 - dark_alpha, dark_layer, dark_alpha, 0)


    # UI

    if face_inside:
        text = "Perfect - press SPACE to photo"
        color = (0, 255, 0)
    else:
        text = "Align the face with the area"
        color = (0, 0, 255)

    cv2.putText(output, text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Face Capture", output)

    key = cv2.waitKey(1) & 0xFF


    # СЪЁМКА

    if key == 32 and face_inside:  # SPACE
        # Сначала сохраняем во временную папку (ASCII)
        success = cv2.imwrite(TEMP_SAVE_PATH, frame)
        if success:
            try:
                # Копируем в папку uploads (может содержать кириллицу)
                shutil.copy2(TEMP_SAVE_PATH, SAVE_PATH)
                os.remove(TEMP_SAVE_PATH)  # удаляем временный файл

                with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                    f.write(SAVE_PATH)
                print(SAVE_PATH)
            except Exception as e:
                with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"ERROR: не удалось скопировать фото: {e}")
                print(f"ERROR: не удалось скопировать фото: {e}")
        else:
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                f.write("ERROR: не удалось сохранить фото (OpenCV)")
            print("ERROR: не удалось сохранить фото (OpenCV)")
        break

    timestamp += 1

cap.release()
cv2.destroyAllWindows()