import cv2
import mediapipe as mp

# --- Configurações Iniciais do MediaPipe ---
# Importa as utilidades de desenho (para desenhar os pontos e linhas no rosto)
mp_drawing = mp.solutions.drawing_utils
# Importa a solução específica de FaceMesh (malha facial com 468 pontos)
mp_face_mesh = mp.solutions.face_mesh

# Índices dos landmarks (pontos) específicos para os olhos no modelo do MediaPipe
# Pontos do contorno do olho esquerdo
p_left_eye = [385, 380, 387, 373, 362, 263]
# Pontos do contorno do olho direito
p_right_eye = [160, 144, 158, 153, 33, 133]
# Lista combinada com todos os pontos dos dois olhos
p_eyes = p_left_eye + p_right_eye


def main():
    # Abre a webcam padrão do computador (índice 0)
    camera = cv2.VideoCapture(0)

    # Verifica se a câmera foi aberta com sucesso
    if not camera.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    # Inicializa o FaceMesh dentro de um bloco 'with' para garantir que os recursos sejam liberados no final
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,              # Detecta no máximo 1 rosto por vez
        refine_landmarks=True,        # Refina os landmarks dos olhos e lábios para maior precisão
        min_detection_confidence=0.5, # Confiança mínima para considerar um rosto detectado (50%)
        min_tracking_confidence=0.5   # Confiança mínima para rastrear os pontos no próximo frame (50%)
    ) as facemesh:

        # Loop principal de captura de vídeo
        while camera.isOpened():
            # Lê um frame (imagem) da câmera
            ret, frame = camera.read()

            # Se o frame não for lido corretamente (ex: câmera desconectada), sai do loop
            if not ret:
                print("Frame vazio da câmera")
                break

            # Opcional: espelhar a imagem horizontalmente (como um espelho)
            # frame = cv2.flip(frame, 1)

            # Converte a imagem de BGR (padrão do OpenCV) para RGB (padrão exigido pelo MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Obtém as dimensões do frame (altura, largura e canais de cor)
            # Nota: 'length' aqui representa a altura (height) e 'width' a largura
            length, width, _ = frame.shape

            # Processa a imagem RGB com o modelo de FaceMesh para extrair os landmarks
            face_mesh_output = facemesh.process(frame_rgb)

            # Verifica se algum rosto foi detectado no frame
            if face_mesh_output.multi_face_landmarks:
                # Itera sobre cada rosto detectado (neste caso, no máximo 1)
                for face_landmarks in face_mesh_output.multi_face_landmarks:
                    
                    # Desenha a malha facial (contornos) na imagem original
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        # Desenha as conexões dos contornos do rosto (boca, nariz, olhos, rosto)
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        # Estilo dos pontos (cor marrom, espessura 1, raio 1)
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(170, 100, 80), thickness=1, circle_radius=1),
                        # Estilo das linhas de conexão (cor verde, espessura 1, raio 1)
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 200, 0), thickness=1, circle_radius=1),
                    )

                    # Extrai a lista de coordenadas normalizadas (de 0.0 a 1.0) do rosto detectado
                    face = face_landmarks.landmark
                    # print(face)  # Descomente para ver todas as coordenadas no terminal (cuidado, polui muito o console)

                    # Itera sobre todos os 468 pontos do rosto
                    for id_coord, coord_xyz in enumerate(face):
                        # Verifica se o índice do ponto atual está na nossa lista de olhos (p_eyes)
                        if id_coord in p_eyes:
                            # Converte as coordenadas normalizadas (x, y) para coordenadas de pixels da tela (cv)
                            coord_cv = mp_drawing._normalized_to_pixel_coordinates(
                                coord_xyz.x,
                                coord_xyz.y,
                                width,  # largura da imagem
                                length  # altura da imagem
                            )

                            # Desenha um círculo preenchido nos pontos dos olhos
                            # (CORREÇÃO: Adicionei a vírgula antes do -1. O -1 indica que o círculo deve ser preenchido)
                            if coord_cv is not None:
                                cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

            # Exibe o frame processado em uma janela chamada "Camera"
            cv2.imshow("Camera", frame)

            # Aguarda 1 milissegundo e verifica se a tecla ESC (código 27) foi pressionada para sair
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Libera o recurso da câmera
    camera.release()
    # Fecha todas as janelas abertas pelo OpenCV
    cv2.destroyAllWindows()


# Garante que a função main só seja executada se o script for rodado diretamente (não importado)
if __name__ == "__main__":
    main()