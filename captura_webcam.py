import cv2

camera = cv2.VideoAccelerationType(0)

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        print("Frame vazio da câmera")
        continue
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break