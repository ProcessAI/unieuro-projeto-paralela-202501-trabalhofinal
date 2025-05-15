import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carrega o classificador de rosto e o modelo de máscara
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('mask_detector.model')  # substitua pelo caminho do seu modelo

# Função para detectar máscaras
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    results = []

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.astype("float") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        (mask, no_mask) = model.predict(face_img)[0]
        results.append(((x, y, w, h), 'No Mask' if no_mask > mask else 'Mask'))

    return results

# Processa o vídeo
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_num = 0

timestamps_no_mask = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = detect_mask(frame)
    if any(label == 'No Mask' for _, label in results):
        timestamp = frame_num / frame_rate
        timestamps_no_mask.append(timestamp)

    frame_num += 1

cap.release()

# Exibe os momentos com pessoas sem máscara
for t in timestamps_no_mask:
    print(f"Pessoas sem máscara detectadas em: {t:.2f} segundos")
