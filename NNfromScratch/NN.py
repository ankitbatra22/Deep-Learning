import sys
import matplotlib
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

"""# Neuron 1:
inputs[0]*weights1[0] +
inputs[1]*weights1[1] +
inputs[2]*weights1[2] +
inputs[3]*weights1[3] + bias1,

# Neuron 2:
inputs[0]*weights2[0] +
inputs[1]*weights2[1] 
inputs[2]*weights2[2] +
inputs[3]*weights2[3] + bias2,

# Neuron 3:
inputs[0]*weights3[0] +
inputs[1]*weights3[1] +
inputs[2]*weights3[2] +
inputs[3]*weights3[3] + bias3]"""

# ONE LAYER 
"""inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
  neuron_output = 0
  for n_input, weight in zip(inputs, neuron_weights):
    neuron_output += n_input*weight
  neuron_output += neuron_bias
  layer_outputs.append(neuron_output)


print(layer_outputs)

#print(np.dot(a,b))"""


#Dense Layer
nnfs.init()

# Dense layer
class Layer_Dense:
# Layer initialization
  def __init__(self, n_inputs, n_neurons):
# Initialize weights and biases
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

# Forward pass
  def forward(self, inputs):
# Calculate output values from inputs, weights and biases
    self.output = np.dot(inputs, self.weights) + self.biases
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Let's see output of the first few samples:
print(dense1.output[:5])








