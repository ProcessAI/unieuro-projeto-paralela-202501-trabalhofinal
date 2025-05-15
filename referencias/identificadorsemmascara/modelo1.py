import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces, confidences = cv.detect_face(frame)

    for face in faces:
        x, y, x2, y2 = face
        cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
