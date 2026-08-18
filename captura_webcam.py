import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

camera = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Frame vazio da câmera")
            continue

        # transforma BGR em RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        face_mesh_output = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            for face_landmarks in face_mesh_output.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        except:
             pass

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break