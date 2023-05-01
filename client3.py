import flwr as fl
import sys
import numpy as np
from keras.layers import (
    Input,
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
)
from keras.models import Model

# Define the input shape
input_shape = (240, 240, 3)

# Define the input placeholder as a tensor with shape input_shape.
X_input = Input(input_shape)  # shape=(?, 240, 240, 3)

# Zero-Padding: pads the border of X_input with zeroes
X = ZeroPadding2D((2, 2))(X_input)  # shape=(?, 244, 244, 3)

# CONV -> BN -> RELU Block applied to X
X = Conv2D(32, (7, 7), strides=(1, 1), name="conv0")(X)
X = BatchNormalization(axis=3, name="bn0")(X)
X = Activation("relu")(X)  # shape=(?, 238, 238, 32)

# MAXPOOL
X = MaxPooling2D((4, 4), name="max_pool0")(X)  # shape=(?, 59, 59, 32)

# MAXPOOL
X = MaxPooling2D((4, 4), name="max_pool1")(X)  # shape=(?, 14, 14, 32)

# FLATTEN X
X = Flatten()(X)  # shape=(?, 6272)

# FULLYCONNECTED
X = Dense(1, activation="sigmoid", name="fc")(X)  # shape=(?, 1)

# Create model
model = Model(inputs=X_input, outputs=X, name="BrainDetectionModel")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Generate random data
X_train = np.random.rand(100, 240, 240, 3)
y_train = np.random.randint(2, size=100)
X_test = np.random.rand(10, 240, 240, 3)
y_test = np.random.randint(2, size=10)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(X_train, y_train, epochs=2, batch_size=10)

        # print("\nFit history : ", r.history, "\n")
        weights = model.get_weights()

        # Set the noisy weights back into the model
        # model.set_weights(noisy_weights)
        # noise_factor = 0.1  # adjust this to control the amount of noise
        # noise = np.random.randn(*weights.shape) * noise_factor
        # noisy_weights = weights + noise
        return weights, len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x=X_test, y=y_test)
        print("\n\n\nEval accuracy : ", accuracy, "\n\n\n")
        return loss, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024,
)
