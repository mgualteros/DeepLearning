# evaluate.py
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

# Cargar los datos de CIFAR-10
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalizar los datos
x_test = x_test / 255.0

# Cargar el modelo previamente guardado
model = load_model('alexnet_cifar10_best.h5')

# Evaluar el modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

# Mostrar los resultados
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
