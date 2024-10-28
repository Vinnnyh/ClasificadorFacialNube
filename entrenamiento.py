from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

#pip install --upgrade pip
#pip install --upgrade scipy

# Configurar los generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Directorios de datos
train_dir = r'proyecto_profesor\Data\train'
validation_dir = r'proyecto_profesor\Data\validation'

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Usa 'categorical' para one-hot encoding
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # Usa 'categorical' para one-hot encoding
)

# Cargar el modelo VGG16 preentrenado sin la parte superior (head)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Añadir nuevas capas para clasificación
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Número de clases = 2

# Definir el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar las capas base
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Entrenar el modelo con callbacks
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

# Descongelar algunas capas para el ajuste fino
for layer in base_model.layers[15:]:
    layer.trainable = True

# Recompilar el modelo
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continuar el entrenamiento
history_fine_tuning = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test Accuracy: {test_acc:.2f}')

# Guardar el modelo entrenado
model.save('model_vgg16_finetuned.h5')

# Guardar etiquetas de clase
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)


import matplotlib.pyplot as plt

# Graficar la pérdida y precisión
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida')
plt.plot(history.history['val_loss'], label='Val Pérdida')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
