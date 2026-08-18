import cv2
import mediapipe as mp

# Configurações do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def main():
    # Abre a webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    # Cria o FaceMesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as facemesh:

        while camera.isOpened():
            ret, frame = camera.read()

            if not ret:
                print("Frame vazio da câmera")
                break

            # Opcional: espelhar a imagem
            # frame = cv2.flip(frame, 1)

            # Converte BGR para RGB, que é o formato esperado pelo MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processa o frame com o FaceMesh
            face_mesh_output = facemesh.process(frame_rgb)

            # Se detectar rosto, desenha os pontos
            if face_mesh_output.multi_face_landmarks:
                for face_landmarks in face_mesh_output.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(170, 100, 80), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 200, 0), thickness=1, circle_radius=1),
                    )

            # Mostra a imagem
            cv2.imshow("Camera", frame)

            # Sai com a tecla ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Libera a câmera e fecha janelas
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()