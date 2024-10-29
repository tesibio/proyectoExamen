import cv2 
import cv2.aruco as aruco
import numpy as np

#Inicializacion de la cámara y diccionario ArUco
cap = cv2.VideoCapture(0)

#cargar el diccionario aruco y parametros
parameters = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
#aruco_dict = cv2.aruco.Dictionary(aruco.DICT_6X6_250)
#aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#parameters = aruco.DetectorParameters_create()

#Cargar la imagen que deseo proyectar
overlay1 = cv2.imread("imagen001.png")
overlay2 = cv2.imread("imagen002.png")
overlay3 = cv2.imread("imagen003.png")

#Variable de escala
ESCALA = 2

# Crear y configurar una única ventana correctamente
cv2.namedWindow("Proyección AR", cv2.WINDOW_NORMAL)  # Crear ventana única
cv2.resizeWindow("Proyección AR", 1080, 960)  # Establecer tamaño fijo

#Estados iniciales
#E_0_0 = False
#E_0_1 = False
#A_4_0 = False

#frame_count = 0
#UPDATE_INTERVAL = 30

while True:
    ret,frame = cap.read()
    if not ret:
        print("No se pudo capturar el video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

    if ids is not None and 2 in ids:
        for i, corner in enumerate(corners):
            if ids[i][0] == 2:
                # Escalar la imagen 3 al tamaño deseado
                new_width = int(overlay3.shape[1] * ESCALA)
                new_height = int(overlay3.shape[0] * ESCALA)
                overlay = cv2.resize(overlay3, (new_width, new_height))

                 # Calcular las esquinas ajustadas
                pts_dst = np.array(corner[0], dtype="float32")
                offset_x = int(pts_dst[0][0] - new_width / 2)
                offset_y = int(pts_dst[0][1] - new_height / 2)
                pts_dst = np.array([
                    [offset_x, offset_y],
                    [offset_x + new_width, offset_y],
                    [offset_x + new_width, offset_y + new_height],
                    [offset_x, offset_y + new_height]
                ], dtype="float32")

                # Calcular la homografía
                pts_src = np.array([[0, 0], [new_width, 0], 
                                    [new_width, new_height], [0, new_height]], dtype="float32")
                matrix, _ = cv2.findHomography(pts_src, pts_dst)
                warped = cv2.warpPerspective(overlay, matrix, (frame.shape[1], frame.shape[0]))

                # Crear una máscara y combinar
                mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts_dst.astype(int), (255, 255, 255))
                frame = cv2.addWeighted(cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)[:, :, 0]),
                                        1, warped, 1, 0)
   
    if ids is not None:
        for i, corner in enumerate (corners):
            marker_id = ids[i][0]
            if marker_id in [0, 1]:
                overlay = overlay1 if marker_id == 0 else overlay2

                pts_dst = np.array(corner[0], dtype = "float32")
                h, w, _ = overlay.shape
                pts_src = np.array([[0,0],[w,0],[w,h],[0,h]],dtype="float32")

                matrix, _ = cv2.findHomography(pts_src, pts_dst)
                warped = cv2.warpPerspective(overlay, matrix, frame.shape[1], frame.shape[0])

                mask = np.zeros_like(frame, dtype = np.uint8)
                cv2.fillConvexPoly(mask, pts_dst.astype(int), (255,255,255))

                frame = cv2.addWeighted(cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)[:, :, 0]),1, warped, 1, 0)
    
    if ids is not None and set([0, 1]).issubset(ids.flatten()):
        A_4_0 = True
    else:
        A_4_0 = False

     # Mostrar estado de la salida A_4_0
    color = (0, 255, 0) if A_4_0 else (0, 0, 255)
    cv2.putText(frame, f'Salida A 4.0: {"Encendida" if A_4_0 else "Apagada"}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Dibujar los marcadores detectados
    aruco.drawDetectedMarkers(frame, corners, ids)

    # Mostrar el frame en la única ventana
    cv2.imshow("Proyección AR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


