# model.py
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),  # Asegurando que la entrada tiene el tamaño correcto
        layers.Conv2D(96, (3, 3), strides=(1, 1), activation='relu'),  # Usamos kernel más pequeño
        layers.MaxPooling2D((2, 2), strides=(2, 2)),  # Ajustamos el pooling
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
