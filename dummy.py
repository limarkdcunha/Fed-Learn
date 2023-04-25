import numpy as np
import random

# Number of clients
NUM_CLIENTS = 10

# Number of malicious clients
NUM_MALICIOUS = 3

# Data dimension
DIM = 5

# Learning rate
LR = 0.1

# Number of rounds
NUM_ROUNDS = 100

# Generate random data for each client
data = [np.random.rand(DIM) for i in range(NUM_CLIENTS)]
print(data)
print("\n")


# Define a malicious client
def malicious_client():
    return np.ones(DIM) * random.randint(-10, 10)


# Define a regular client
def regular_client():
    return data[random.randint(0, NUM_CLIENTS - 1)]


# Define the Byzantine averaging function
def byzantine_average(client_data):
    if len(client_data) == NUM_MALICIOUS:
        return malicious_client()
    else:
        return sum(client_data) / (NUM_CLIENTS - NUM_MALICIOUS)


# Run the federated learning algorithm
global_model = np.zeros(DIM)

for round in range(NUM_ROUNDS):
    client_data = []
    for i in range(NUM_CLIENTS):
        if i < NUM_MALICIOUS:
            client_data.append(malicious_client())
        else:
            client_data.append(regular_client())

    local_model = byzantine_average(client_data)
    print("local model : ", local_model)
    global_model = global_model - LR * (global_model - local_model)

print("Global model:", global_model)
