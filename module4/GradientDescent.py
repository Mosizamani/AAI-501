import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# 1. Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize input (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Build a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),        # Flatten 28x28 image to 784
    Dense(128, activation='relu'),        # Hidden layer with 128 units
    Dense(10, activation='softmax')       # Output layer for 10 digits
])

# 5. Compile the model using Stochastic Gradient Descent
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
