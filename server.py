import flwr as fl
import sys
import numpy as np
import tensorflow as tf
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
from helper import load_data, split_data, build_model
import sys


def weighted_average(metrics):
    """An evaluation function for server-side evaluation"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)

    print("\n\n Accuracy: ", accuracy, "\n\n")

    return {"accuracy": accuracy}


# Evaluation function for server side
def get_evaluate_fn(model, x_val, y_val):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_val, y_val)
        # print("Server side accuracy : ", accuracy, " at round : ", server_round)
        return loss, {"accuracy": accuracy}

    return evaluate


# Load and compile model for server-side parameter evaluation
# Data loading
# data_path = "augmented data/"

# data_yes = data_path + "yes"
# data_no = data_path + "no"

# IMG_WIDTH, IMG_HEIGHT = (240, 240)
# X, y = load_data([data_yes, data_no], (IMG_WIDTH, IMG_HEIGHT))

# # Data split
# X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.2)

# print(X_val.shape)
# Define image shape
# IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# Build model
# model = build_model(IMG_SHAPE)

# Compile the model
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fed byzantine averaging
fed_byz = fl.server.strategy.FedByzantineAvg(
    # eval_fn=get_evaluate_fn(model, X_val, y_val),
    evaluate_metrics_aggregation_fn=weighted_average
)
# Fed averaging
fed_avg = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
    # evaluate_fn=get_evaluate_fn(model, X_val, y_val),
)

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address="localhost:" + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=3),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=fed_byz,
)
