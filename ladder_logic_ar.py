import cv2 
import cv2.aruco as aruco

#Inicializacion de la cámara y diccionario ArUco
cap = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

#Estados iniciales
E_0_0 = False
E_0_1 = False
A_4_0 = False

while True:
    ret,frame = cap.read()
    gray = cv2.ctvColor(frame, cv2.COLOR_BGR2GRAY)

    #detectar marcadores
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

    if ids is not None:
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

    #mostrar la imagen en pantalla
    cv2.imshow('Diagrama en escalera AR', frame)

    if cv2.waitkey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAll



