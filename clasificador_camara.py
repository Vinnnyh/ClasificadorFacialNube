import importlib.util
import json
import cv2
import torch
import numpy as np
import time
#import RPi.GPIO as GPIO
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Configurar el GPIO
#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)  # Suprimir advertencias
#servo_pin = 17
#GPIO.setup(servo_pin, GPIO.OUT)
#servo = GPIO.PWM(servo_pin, 50)
#servo.start(0)

# Función para cargar el modelo YOLOv5
def load_yolov5_model(weights, device):
    model = torch.load(weights, map_location=device)['model'].float()
    model.eval()
    return model

# Función para procesar las detecciones de YOLOv5
def detect_faces_yolo(model, frame, device):
    img_size = 640
    img = letterbox(frame, img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img.copy())  # Hacer una copia para evitar problemas de "stride"

    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

    faces = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                xyxy = [int(coord) for coord in xyxy]  # Convertir coordenadas a enteros
                faces.append(xyxy)  # Agregar a la lista de caras
    return faces

# Cargar las etiquetas de clase
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# Cargar el modelo VGG16 reentrenado
model_vgg16 = load_model('model_vgg16_finetuned.h5')

# Cargar el modelo YOLOv5
yolov5_face_model_path = 'yolov5s-face.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolov5_face_model = load_yolov5_model(yolov5_face_model_path, device)

# Inicializar la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Variables de estado
current_state = None
hidden_detected_time = 0
max_hidden_time = 2
grace_period = 1
grace_period_counter = 0
frame_skip = 1
frame_count = 0
frame_analysis_interval = 10
analysis_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el marco de la cámara.")
        break

    frame_count += 1  # Incrementar el conteo de cuadros


    faces = detect_faces_yolo(yolov5_face_model, frame, device)
    if faces:  # Asegúrate de que esta línea esté separada
        for (x1, y1, x2, y2) in faces:
            # Ajuste de región de la cara
            y1 = max(0, y1 - 30)
            y2 = min(frame.shape[0], y2 + 30)
            x1 = max(0, x1 - 30)
            x2 = min(frame.shape[1], x2 + 30)

            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue


        # Preprocesar la imagen de la cara
        face_region = cv2.resize(face_region, (224, 224))
        face_image = keras_image.img_to_array(face_region)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = preprocess_input(face_image)

        predictions = model_vgg16.predict(face_image)
        predicted_class = np.argmax(predictions[0])
        class_name = class_labels.get(predicted_class, 'Desconocido')
        print(f'Clase detectada: {class_name}')
        if class_name == "Destapado":
            #servo.ChangeDutyCycle(2.5)  # Abierto
            pass
        elif class_name == "Oculto":
            hidden_detected_time += 1
            #servo.ChangeDutyCycle(12.5)  # Cerrado
            grace_period_counter = 0
            if hidden_detected_time >= max_hidden_time:
                print("¡Alerta! Persona oculta detectada.")
                #servo.ChangeDutyCycle(12.5)  # Posición de alerta
                time.sleep(1)
                #servo.ChangeDutyCycle(7.5)  # Regresar a posición neutral
                while True:
                    if input("Presiona 'q' para terminar el modo alarma: ") == "q":
                        hidden_detected_time = 0
                        break
        else:
            grace_period_counter += 1
            if grace_period_counter >= grace_period:
                hidden_detected_time = 0
                grace_period_counter = 0

        analysis_counter += 1
    else:
        # Reiniciar hidden_detected_time si no se detecta nada
        grace_period_counter += 1
        if grace_period_counter >= grace_period:
            hidden_detected_time = 0
            grace_period_counter = 0

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
#servo.stop()  # Detener el PWM del servomotor
#GPIO.cleanup()  # Limpiar los pines GPIO
