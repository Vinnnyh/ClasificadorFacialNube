from flask import Flask
from flask import request
from flask import Response
from http import HTTPStatus
import numpy as np
import json
import cv2
import torch
import numpy as np
#import RPi.GPIO as GPIO
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

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

      
app = Flask(__name__)

@app.post("/")
def post():
    frame = request.data
    frame = np.frombuffer(frame,np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
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
        print(class_name)
    return Response(class_name,HTTPStatus(200), mimetype="text/plain")
