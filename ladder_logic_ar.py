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
ESCALA = 0.5

#Estados iniciales
E_0_0 = False
E_0_1 = False
A_4_0 = False

while True:
    ret,frame = cap.read()
    if not ret:
        print("No se pudo capturar el video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectar marcadores
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0] #Obtener el ID del marcador
            pts_dst = np.array(corner[0], dtype = "float32")

            #Seleccionar la imagen correspondiente segun el ID 
            if marker_id == 0:
                overlay = overlay1
            elif marker_id == 1:
                overlay = overlay2
            elif marker_id == 2:
                new_width = int(frame.shape[1] * ESCALA)
                new_height = int(frame.shape[0] * ESCALA)
                overlay = cv2.resize(overlay3, (new_width, new_height)) 
                # Ajustar las nuevas esquinas de la imagen grande
                pts_src = np.array([[0, 0], [new_width, 0], 
                                    [new_width, new_height], [0, new_height]], dtype="float32")

                # Mover la imagen más grande hacia la esquina superior izquierda del marcador
                offset_x = int(pts_dst[0][0] - new_width / 8)
                offset_y = int(pts_dst[0][1] - new_height / 8)
                pts_dst = np.array([
                    [offset_x, offset_y],
                    [offset_x + new_width, offset_y],
                    [offset_x + new_width, offset_y + new_height],
                    [offset_x, offset_y + new_height]
                ], dtype="float32")
                #overlay = cv2.resize(overlay3, (frame.shape[1], frame.shape[0]))
            else:
                continue

        #for corner in corners:
        #    #obtener las esquinas del marcador
        #    pts_dst = np.array(corner[0], dtype = "float32")

            #Dimensiones de la imagen a proyectar
            h, w, _ = overlay.shape
            h2, w2, _ = overlay2.shape
            pts_src = np.array([[0,0], [w,0], [w,h], [0,h]], dtype = "float32")
            pts_src2 = np.array([[0,0], [w,0], [w,h], [0,h]], dtype = "float32")

            #Calcular la homografía
            matrix, _ = cv2.findHomography(pts_src, pts_dst)
            warped = cv2.warpPerspective(overlay, matrix,(frame.shape[1],frame.shape[0]))
            
            #Crear una mascara para combinar las imagenes
            mask = np.zeros_like(frame, dtype = np.uint8)
            cv2.fillConvexPoly(mask, pts_dst.astype(int),(255,255,255))

            #Combinar la imagen proyectada con el frame original
            frame = cv2.addWeighted(cv2.bitwise_and(frame,frame, mask = cv2.bitwise_not(mask)[:,:,0]),1,warped,1,0)


        for marker_id in ids.flatten():
            if marker_id == 0: #E 0.0
                E_0_0 = True
            elif marker_id == 1: #E 0.1
                E_0_1 = True

        #Lógica de activación: ambos marcadores deben de estar presentes
        A_4_0 = E_0_0 and E_0_1

    #Mostrar estado de la salida
    color = (0, 255, 0) if A_4_0 else (0, 0, 255)
    cv2.putText(frame,
                f'Salida A 4.0: {"Encendida" if A_4_0 else "Apagada"}',
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color ,2
               )
    
    #Mostrar los marcadores detectados
    aruco.drawDetectedMarkers(frame, corners, ids)

 #Mostrar la imagen con los FPS
    cv2.imshow("Deteccion de manos",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAll


