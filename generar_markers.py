import cv2
import cv2.aruco as aruco

# Crear diccionario ArUco del tipo 6x6 con 250 marcadores
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

for marker_id in range(5):
    # Generar un marcador con ID 0
    #marker_id = 0
    marker_size = 200  # Tamaño del marcador en píxeles
    marker = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    filename = f'marker_{marker_id}.png'
    # Guardar el marcador como imagen PNG
    cv2.imwrite(filename, marker)
    print(f"Marcador {marker_id} generado y guardado como marker_{marker_id}.png")


