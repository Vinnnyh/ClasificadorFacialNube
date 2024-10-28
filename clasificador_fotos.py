import json
from PIL import Image
import random
import cv2
import os
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ruta al archivo del modelo
model_path = 'model_vgg16_finetuned.h5'

# Cargar el modelo VGG16 reentrenado
try:
    model = load_model(model_path)
except OSError as e:
    print(f"Error de OSError al cargar el modelo: {e}")
    exit()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Cargar las etiquetas de clase
try:
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)
    # Convertir las claves del diccionario a enteros
    class_labels = {int(k): v for k, v in class_labels.items()}
except Exception as e:
    print(f"Error al cargar las etiquetas de clase: {e}")
    exit()

# Crear el detector MTCNN
detector = MTCNN()

# Ruta de la carpeta con imágenes
image_folder = 'D:/proyecto rostros/fotos/caras/todas/'
# Número de imágenes a procesar
num_images_to_process = 10

# Obtener lista de archivos de imagen
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Seleccionar aleatoriamente las imágenes
if len(image_files) < num_images_to_process:
    print(f"No hay suficientes imágenes en la carpeta. Solo hay {len(image_files)} imágenes disponibles.")
    num_images_to_process = len(image_files)
image_files = random.sample(image_files, num_images_to_process)

print(image_files)
for image_file in image_files:
    # Cargar la imagen
    image_path = os.path.join(image_folder, image_file)
    print(image_path)
    image = Image.open(image_path)
    image_array = np.array(image)
    
    if image is None:
        print(f"Error al cargar la imagen {image_file}")
        continue

    # Inicializar la variable text
    text = "No se detectó ninguna cara"
    
    width,height,_=image_array.shape
    
    # Detectar las caras en la imagen
    faces = detector.detect_faces(image_array) 
    
    for face in faces: 
        x1, y1=0,0
        x2, y2 = width,height
         
        face_region = image_array[y1:y2, x1:x2]
        face_region = cv2.resize(face_region, (224, 224))

        face_image = keras_image.img_to_array(face_region)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = preprocess_input(face_image)

        predictions = model.predict(face_image)
        predicted_class = np.argmax(predictions[0])
        predicted_prob = np.max(predictions[0])

        class_name = class_labels.get(predicted_class, 'Desconocido')
        

        # Escribir el texto en la imagen
        text = f'{class_name}: {predicted_prob:.2f}'
        
        # Dibujar el rectángulo alrededor del rostro
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Escribir el texto en la imagen
        cv2.putText(image_array, text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
         
     
    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Crear una nueva figura para cada imagen
    plt.figure()
    plt.imshow(image_array)
    plt.title(f'Procesado: {image_file}')
    plt.axis('off')
    plt.show()
    
    # Opcional: espera para visualizar cada imagen antes de pasar a la siguiente
    input("Presiona Enter para continuar con la siguiente imagen...")

    # Cerrar la figura actual para evitar la acumulación
    plt.close()

print("Procesamiento completo.")
