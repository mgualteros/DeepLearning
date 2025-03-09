# train.py
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from model import build_model  # Asegúrate de que el archivo model.py esté en el mismo directorio

# Cargar los datos de CIFAR-10
(x_train, y_train), (x_val, y_val) = cifar10.load_data()

# Normalizar los datos
x_train, x_val = x_train / 255.0, x_val / 255.0

# Construir el modelo
model = build_model()

# Ruta para guardar el modelo
save_dir = '/home/mike/EntornoGPU/Deep Learning/Ejercicio_alexnet_cifar10/src/'

# Definir el callback para guardar el modelo con el mejor rendimiento
checkpoint_callback = ModelCheckpoint(save_dir + 'alexnet_cifar10_best.h5', 
                                      monitor='val_loss', 
                                      save_best_only=True, 
                                      mode='min', 
                                      verbose=1)

# Entrenar el modelo
history = model.fit(x_train, y_train, 
                    epochs=5, 
                    batch_size=64, 
                    validation_data=(x_val, y_val), 
                    callbacks=[checkpoint_callback])

# Guardar el modelo final al terminar el entrenamiento
model.save(save_dir + 'alexnet_cifar10_final.h5')
