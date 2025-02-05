import pennylane as qml
import torch
import torch.nn as nn


def state_prepare(input, qbits):
    qml.AngleEmbedding(input, wires=range(qbits))


def conv(b1, b2, params):
    qml.RZ(-torch.pi / 2, wires=b2)
    qml.CNOT([b2, b1])
    qml.RZ(params[0], wires=b1)
    qml.RY(params[1], wires=b2)
    qml.CNOT([b1, b2])
    qml.RY(params[2], wires=b2)
    qml.CNOT([b2, b1])
    qml.RZ(torch.pi / 2, wires=b1)


def pool(b1, b2, params):
    qml.RZ(-torch.pi / 2, wires=b2)
    qml.CNOT([b2, b1])
    qml.RZ(params[0], wires=b1)
    qml.RY(params[1], wires=b2)
    qml.CNOT([b1, b2])
    qml.RY(params[2], wires=b2)


dev4 = qml.device("default.qubit", wires=4)


@qml.qnode(dev4, interface="torch", diff_method="backprop")
def qcnn4(weights, inputs):
    state_prepare(inputs, 4)
    conv(0, 1, weights[0:3])
    conv(2, 3, weights[3:6])
    pool(0, 1, weights[6:9])
    pool(2, 3, weights[9:12])
    conv(1, 3, weights[12:15])
    pool(1, 3, weights[15:18])
    return qml.probs(wires=3)


dev8 = qml.device("default.qubit", wires=8)


@qml.qnode(dev8, interface="torch", diff_method="backprop")
def qcnn8(weights, inputs):
    state_prepare(inputs, 8)
    conv(0, 1, weights[0:3])
    pool(0, 1, weights[3:6])
    conv(2, 3, weights[6:9])
    pool(2, 3, weights[9:12])
    conv(4, 5, weights[12:15])
    pool(4, 5, weights[15:18])
    conv(6, 7, weights[18:21])
    pool(6, 7, weights[21:24])
    conv(1, 3, weights[24:27])
    pool(1, 3, weights[27:30])
    conv(5, 7, weights[30:33])
    pool(5, 7, weights[33:36])
    conv(3, 7, weights[36:39])
    pool(3, 7, weights[39:42])
    return qml.probs(wires=7)


# Define a 12-qubit device
dev12 = qml.device("default.qubit", wires=12)


# Define the quantum convolutional neural network
@qml.qnode(dev12, interface="torch", diff_method="backprop")
def qcnn12(weights, inputs):
    # Prepare the quantum state based on the input
    state_prepare(inputs, 12)

    # First layer: Convolution and pooling for each adjacent pair of qubits
    conv(0, 1, weights[0:3])
    pool(0, 1, weights[3:6])
    conv(2, 3, weights[6:9])
    pool(2, 3, weights[9:12])
    conv(4, 5, weights[12:15])
    pool(4, 5, weights[15:18])
    conv(6, 7, weights[18:21])
    pool(6, 7, weights[21:24])
    conv(8, 9, weights[24:27])
    pool(8, 9, weights[27:30])
    conv(10, 11, weights[30:33])
    pool(10, 11, weights[33:36])

    # Second layer: Convolution and pooling between non-adjacent pairs of qubits
    conv(1, 3, weights[36:39])
    pool(1, 3, weights[39:42])
    conv(5, 7, weights[42:45])
    pool(5, 7, weights[45:48])
    conv(9, 11, weights[48:51])
    pool(9, 11, weights[51:54])

    # Third layer: Convolution and pooling across larger separations
    conv(3, 7, weights[54:57])
    pool(3, 7, weights[57:60])
    conv(7, 11, weights[60:63])
    pool(7, 11, weights[63:66])
    conv(3, 11, weights[66:69])
    pool(3, 11, weights[69:72])

    # Return the probabilities of the last qubit
    return qml.probs(wires=11)


def draw(input, params, cir, name):
    qml.drawer.use_style("pennylane")
    fig, ax = qml.draw_mpl(cir)(params, input)
    fig.savefig("result/" + name + ".png")


class QCNN(nn.Module):
    def __init__(self, qbits):
        super().__init__()
        self.qbits = qbits

        if qbits == 4:
            self.qcnn_layer = qml.qnn.TorchLayer(qcnn4, weight_shapes={"weights": 18})
        elif qbits == 8:
            self.qcnn_layer = qml.qnn.TorchLayer(qcnn8, weight_shapes={"weights": 42})
        elif qbits == 12:
            self.qcnn_layer = qml.qnn.TorchLayer(qcnn12, weight_shapes={"weights": 72})

    def forward(self, x):

        max = torch.max(x)
        min = torch.min(x)
        x = (x - min) / (max - min)
        x = 2 * x * torch.pi
        return self.qcnn_layer(x)
