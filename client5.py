import flwr as fl
import sys
from helper import load_data, split_data, build_model

# Data loading
data_path = "data_5/"

data_yes = data_path + "yes"
data_no = data_path + "no"

IMG_WIDTH, IMG_HEIGHT = (240, 240)
X, y = load_data([data_yes, data_no], (IMG_WIDTH, IMG_HEIGHT))

# Data split
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

# define image shape
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# Build model
model = build_model(IMG_SHAPE)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(
            x=X_train,
            y=y_train,
            batch_size=32,
            epochs=5,
            validation_data=(X_val, y_val),
        )
        # print("\nFit history : ", r.history, "\n")
        return model.get_weights(), len(X_train), {}

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
