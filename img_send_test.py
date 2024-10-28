import requests
import cv2
from http import HTTPStatus
import time
#import RPi.GPIO as GPIO

# Configurar el GPIO
#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)  # Suprimir advertencias
#servo_pin = 17
#GPIO.setup(servo_pin, GPIO.OUT)
#servo = GPIO.PWM(servo_pin, 50)
#servo.start(0)

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
analysis_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el marco de la cámara.")
        break

    # Codifica la imagen en formato PNG directamente en memoria
    _, img_encoded = cv2.imencode(".png", frame)
    
    # Convierte la imagen codificada en un buffer de bytes para enviarla
    response = requests.post("http://127.0.0.1:5000", data=img_encoded.tobytes(), headers={'Content-Type': 'image/png'})
    
    if response.status_code == HTTPStatus.OK:
        print(response.text)
        if response.text == "Destapado":
                #servo.ChangeDutyCycle(2.5)  # Abierto
                pass
        elif response.text == "Oculto":
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

cap.release()

#servo.stop()  # Detener el PWM del servomotor
#GPIO.cleanup()  # Limpiar los pines GPIO