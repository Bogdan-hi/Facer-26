import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# НАСТРОЙКИ

SEG_MODEL = "selfie_segmenter.tflite"
FACE_MODEL = "blaze_face_short_range.tflite"  # модель детекции лица

BG_COLOR = (0, 200, 255)
SAVE_PATH = "captured_face.jpg"

# СЕГМЕНТАЦИЯ

seg_options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=SEG_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    output_category_mask=True
)
segmentor = vision.ImageSegmenter.create_from_options(seg_options)

# ДЕТЕКЦИЯ ЛИЦА

face_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.VIDEO
)
face_detector = vision.FaceDetector.create_from_options(face_options)

# ПАРАМЕТРЫ ВАЛИДАЦИИ

POSITION_TOLERANCE = 0.3   # допустимое смещение
SIZE_TOLERANCE = 0.4       # допустимое отклонение размера

SMOOTHING = 0.8             # сглаживание (0.7–0.9)
dark_alpha = 0.0            # текущее затемнение (0..1)
score_smooth = 0.0

# КАМЕРА

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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

    # бинарная маска человека
    condition = mask > 0.1

    # РАЗМЫТИЕ ФОНА

    mask_f = mask.astype(np.float32)

    # ФИКС
    # проверяем, кто "больше" — фон или человек
    # если среднее значение большое → это фон → инвертируем
    if np.mean(mask_f) > 0.5:
        mask_f = 1.0 - mask_f

    # сглаживаем маску (края)
    mask_f = cv2.GaussianBlur(mask_f, (21, 21), 0)

    # делаем диапазон более контрастным
    mask_f = np.clip((mask_f - 0.1) * 1.3, 0, 1)

    mask_f = mask_f[..., None]

    # блюрим фон
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)

    # смешивание: человек резкий, фон размытый
    output = frame * mask_f + blurred * (1 - mask_f)
    output = output.astype(np.uint8)

    # КВАДРАТ (ROI)

    center = (w // 2, h // 2)
    half_size = int(min(w, h) * 0.25)  # половина стороны квадрата

    x1 = center[0] - half_size
    y1 = center[1] - half_size
    x2 = center[0] + half_size
    y2 = center[1] + half_size

    square_color = (
        0,
        int(255 * score_smooth),
        int(255 * (1 - score_smooth))
    )

    # рисуем квадрат
    cv2.rectangle(output, (x1, y1), (x2, y2), square_color, 3)

    # ДЕТЕКЦИЯ ЛИЦА

    face_result = face_detector.detect_for_video(mp_image, timestamp)

    score = 0.0
    face_inside = False

    if face_result.detections:
        det = face_result.detections[0]  # берём одно лицо
        bbox = det.bounding_box
        x, y, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        # центр лица
        fx = x + bw / 2
        fy = y + bh / 2

        # 1. ПОЗИЦИЯ (нормированная)

        dx = (fx - center[0]) / half_size
        dy = (fy - center[1]) / half_size

        # для квадрата логичнее использовать max, а не евклидово расстояние
        position_error = max(abs(dx), abs(dy))

        face_size = (bw + bh) / 2
        target_size = half_size * 2 * 0.9  # немного меньше квадрата

        size_error = abs(face_size - target_size) / target_size

        # 3. ОБЩИЙ SCORE

        pos_score = max(0, 1 - position_error / POSITION_TOLERANCE)
        size_score = max(0, 1 - size_error / SIZE_TOLERANCE)

        score = pos_score * size_score

        # сглаживание
        score_smooth = SMOOTHING * score_smooth + (1 - SMOOTHING) * score

        face_inside = score_smooth > 0.75

        # цвет bbox (индикация)
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
        text = "Perfect - press SPACE"
        color = (0, 255, 0)
    else:
        text = "Align face with the oval"
        color = (0, 0, 255)
        
    cv2.imshow("Face Capture", output)
    cv2.putText(output, text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    key = cv2.waitKey(1) & 0xFF

    # СЪЕМКА
    
    if key == 32 and face_inside:  # SPACE
        cv2.imwrite(SAVE_PATH, frame)
        print(f"Фото сохранено: {SAVE_PATH}")
        break

    if key == 27:  # ESC
        break

    timestamp += 1

cap.release()
cv2.destroyAllWindows()