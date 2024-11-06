import cv2
import numpy as np

# Cargar el clasificador en cascada de OpenCV para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar las imágenes de gafas, bigote y gorra en formato RGBA (con canal alfa)
gafas_img = cv2.imread('gafas.png', cv2.IMREAD_UNCHANGED)  # Cargar con canal alfa
bigote_img = cv2.imread('bigote.png', cv2.IMREAD_UNCHANGED)  # Cargar con canal alfa
gorra_img = cv2.imread('gorra.png', cv2.IMREAD_UNCHANGED)  # Cargar con canal alfa

# Verifica que las imágenes se han cargado correctamente
if gafas_img is None or bigote_img is None or gorra_img is None:
    print("Error: Las imágenes no se pudieron cargar. Asegúrate de que las rutas sean correctas y las imágenes estén en formato PNG.")
    exit()

# Función para obtener el canal alfa (si está presente)
def get_alpha_channel(image):
    if image.shape[2] == 4:
        return image[:, :, 3] / 255.0
    else:
        return np.ones((image.shape[0], image.shape[1]), dtype=float)

# Abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

while True:
    # Leer la imagen desde la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara")
        break

    # Convertir la imagen a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor de la cara (opcional)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Colocar las gafas
        gafas_width = int(w * 0.8)
        gafas_height = int(gafas_width / gafas_img.shape[1] * gafas_img.shape[0])
        gafas_resized = cv2.resize(gafas_img, (gafas_width, gafas_height))
        
        gafas_x = x + (w - gafas_width) // 2
        gafas_y = y + int(h * 0.2)
        
        alpha_gafas = get_alpha_channel(gafas_resized)
        for c in range(0, 3):
            frame[gafas_y:gafas_y + gafas_height, gafas_x:gafas_x + gafas_width, c] = \
                (1. - alpha_gafas) * frame[gafas_y:gafas_y + gafas_height, gafas_x:gafas_x + gafas_width, c] + \
                alpha_gafas * gafas_resized[:, :, c]

        # Colocar el bigote
        bigote_width = int(w * 0.7)
        bigote_height = int(bigote_width / bigote_img.shape[1] * bigote_img.shape[0])
        bigote_resized = cv2.resize(bigote_img, (bigote_width, bigote_height))

        bigote_x = x + (w - bigote_width) // 2
        bigote_y = y + int(h * 0.65)
        if bigote_y + bigote_height > frame.shape[0]:
            bigote_y = frame.shape[0] - bigote_height

        alpha_bigote = get_alpha_channel(bigote_resized)
        for c in range(0, 3):
            frame[bigote_y:bigote_y + bigote_height, bigote_x:bigote_x + bigote_width, c] = \
                (1. - alpha_bigote) * frame[bigote_y:bigote_y + bigote_height, bigote_x:bigote_x + bigote_width, c] + \
                alpha_bigote * bigote_resized[:, :, c]

        # Colocar la gorra
        gorra_width = int(w * 1.8)  # Ajustar el tamaño de la gorra según el tamaño del rostro
        gorra_height = int(gorra_width / gorra_img.shape[1] * gorra_img.shape[0])

        gorra_resized = cv2.resize(gorra_img, (gorra_width, gorra_height))

        gorra_x = x + (w - gorra_width) // 2  # Colocar la gorra centrada sobre la cabeza
        gorra_y = y - int(gorra_height * 0.7)  # Ajustar la gorra por encima de la cara

        # Mover la gorra más a la derecha
        gorra_x += 23  # Ajusta el valor (por ejemplo, 20 píxeles más a la derecha)

        # Asegurarse de que la gorra no se salga de los límites de la imagen
        if gorra_y < 0:
            gorra_y = 0

        # Verificar las dimensiones de la gorra y la región donde se coloca
        gorra_resized = gorra_resized[:frame[gorra_y:gorra_y + gorra_height, gorra_x:gorra_x + gorra_width].shape[0],
                                      :frame[gorra_y:gorra_y + gorra_height, gorra_x:gorra_x + gorra_width].shape[1]]

        # Obtener el canal alfa de la gorra
        alpha_gorra = get_alpha_channel(gorra_resized)
        for c in range(0, 3):  # Recorrer los canales RGB
            frame[gorra_y:gorra_y + gorra_height, gorra_x:gorra_x + gorra_width, c] = \
                (1. - alpha_gorra) * frame[gorra_y:gorra_y + gorra_height, gorra_x:gorra_x + gorra_width, c] + \
                alpha_gorra * gorra_resized[:, :, c]

    # Mostrar la imagen final con los filtros
    cv2.imshow("Filtro de Realidad Aumentada", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
