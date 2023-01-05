import numpy as np
import itertools
from typing import List, Tuple

class Layer:
    def __init__(self):
        pass
    
    def forward(self, input_: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    
    id_iter = itertools.count()

    def __init__(self, n_input_neurons: int, n_output_neurons: int):
        # id is used to identify the layer's weights and bias in the optimizer
        self.id = next(FullyConnectedLayer.id_iter)
        self.weights = np.random.randn(n_input_neurons, n_output_neurons) * 0.01
        self.bias = np.random.randn(1, n_output_neurons) * 0.01

        #! Code below was not used. Was intended to be used in the optimizer
        """
        self.parameter_by_gradient_id = {
            f"dW{self.id}": self.weights,
            f"db{self.id}": self.bias
        }
        """
    
    def forward(self, input_: np.ndarray) -> np.ndarray:
        """
        input_ is a 2D array with shape (batch_size, n_input_neurons)
        output is a 2D array with shape (batch_size, n_output_neurons)
        """
        self.input_ = input_
        self.output = np.dot(self.input_, self.weights) + self.bias
        return self.output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # The math explanation: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights_gradient = np.dot(self.input_.T, output_gradient)
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        self.W_and_b_grad = (self.weights_gradient, self.bias_gradient)
        return input_gradient
    
    def get_W_and_b_ids(self) -> Tuple[str, str]:
        weights_id = f"dW{self.id}"
        bias_id = f"db{self.id}"
        return weights_id, bias_id
    
    #! Code below was not used. Was intended to be used in the optimizer
    """
    def update_parameter_by_gradient_id(self, gradient_id: str, gradient: np.ndarray):
        self.parameter_by_gradient_id[gradient_id] = gradient
    """


class Conv2d(Layer):
    id_iter = itertools.count()

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.id = next(Conv2d.id_iter)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # out_channels is the number of filters and in_channels, kernel_size, kernel_size are the shape of the filter
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        if self.bias:
            self.bias = np.random.randn(out_channels) * 0.01
        else:
            self.bias = None
    
    def get_padded_input(self, input_: np.ndarray) -> np.ndarray:
        batch_size, in_channels, height, width = input_.shape
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        padded_input = np.zeros((batch_size, in_channels, padded_height, padded_width))
        padded_input[:, :, self.padding:self.padding+height, self.padding:self.padding+width] = input_
        return padded_input
    
    # ! May be make convolution a separate function and call it in forward and backward
    def forward(self, input_: np.ndarray) -> np.ndarray:
        """
        input_ is a 4D array with shape (batch_size, in_channels, height, width)
        """
        self.input_ = input_
        batch_size, _, height, width = input_.shape
        padded_input = self.get_padded_input(input_)
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        out_height = (padded_height - self.kernel_size) // self.stride + 1
        out_width = (padded_width - self.kernel_size) // self.stride + 1
        output = np.empty((batch_size, self.out_channels, out_height, out_width))
        # oci stands for output channel index
        for oci in range(self.out_channels):
            for h in range(out_height):
                for w in range(out_width):
                    output[:, oci, h, w] = np.sum(
                        self.weights[oci] * padded_input[:, :, h*self.stride:h*self.stride+self.kernel_size, w*self.stride:w*self.stride+self.kernel_size],
                        axis=(1, 2, 3))
            if self.bias is not None:
                output[:, oci] += self.bias[oci]
        return output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        output_gradient is a 4D array with shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, out_channels, out_height, out_width = output_gradient.shape
        _, in_channels, height, width = self.input_.shape
        padded_input = self.get_padded_input(self.input_)
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        input_gradient = np.zeros((batch_size, in_channels, padded_height, padded_width))
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.bias.shape)
        # bi stands for batch index
        for bi in range(batch_size):
            for oci in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        input_gradient[bi, :, h*self.stride:h*self.stride+self.kernel_size, w*self.stride:w*self.stride+self.kernel_size] += self.weights[oci] * output_gradient[bi, oci, h, w]
                        self.weights_gradient[oci] += padded_input[bi, :, h*self.stride:h*self.stride+self.kernel_size, w*self.stride:w*self.stride+self.kernel_size] * output_gradient[bi, oci, h, w]
                        if self.bias is not None:
                            self.bias_gradient[oci] += output_gradient[bi, oci, h, w]
        return input_gradient[:, :, self.padding:self.padding+height, self.padding:self.padding+width]

        

    



class ActivationLayer(Layer):
    """
    This base class is not crucial but it presevrves OOP ideology and 
    it is useful for type hinting like List[ActivationLayer].
    """
    def __init__(self):
        pass

class ReLULayer(ActivationLayer):
    def forward(self, input_: np.ndarray) -> np.ndarray:
        self.input_ = input_
        return np.maximum(0, input_)
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (self.input_ > 0)

class SigmoidLayer(ActivationLayer):
    def forward(self, input_: np.ndarray) -> np.ndarray:
        # clip is used to avoid overflow
        self.output = 1 / (1 + np.exp(-np.clip(input_, 1e-8, 1e2)))
        # self.output = 1 / (1 + np.exp(-input_))
        return self.output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.output * (1 - self.output)

class LinearActivation(ActivationLayer):
    def forward(self, input_: np.ndarray) -> np.ndarray:
        return input_
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient


def softmax(x: np.ndarray) -> np.ndarray:
    # The output of softmax will be same if we substract some constant c
    # from the input. It's same as multiplying initial expression by
    # e^(-c)/e^(-c). The substraction helps to avoid overflow.
    e_subtracted = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return e_subtracted / np.sum(e_subtracted, axis=1, keepdims=True)

class SoftMaxLayer(Layer):
    def forward(self, input_: np.ndarray) -> np.ndarray:
        self.output = softmax(input_)
        return self.output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.output * (1 - self.output)
    
               

class CrossEntropyLoss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        batch_size = pred.shape[0]
        # summing over the batch and then dividing by the batch size
        # return -np.sum(np.dot(self.target, np.log(self.cliped_pred()).T)) / batch_size
        return -np.sum(self.target * np.log(self.cliped_pred())) / batch_size
    
    def backward(self) -> np.ndarray:
        batch_size = self.pred.shape[0]
        return - self.target / self.cliped_pred() / batch_size

    def cliped_pred(self) -> np.ndarray:
        # Clip is used to avoid negative values as input to if we don't
        # use softmax. !Maybe it's better to have an upper bound of 1 so that
        # the possible values of the input to log are in the range [1e-8, 1]
        # even if we don't use softmax
        return np.clip(self.pred, 1e-8, None)

class CrossEntropyLossWithSoftMax:
    def forward(self, activation: np.ndarray, target: np.ndarray) -> float:
        self.pred = np.clip(softmax(activation), 1e-8, None)
        self.target = target
        batch_size = activation.shape[0]
        return -np.sum(self.target * np.log(self.pred)) / batch_size
    
    def backward(self) -> np.ndarray:
        batch_size = self.pred.shape[0]
        return (self.pred - self.target) / batch_size


class Optimizer:
    def __init__(self, trainable_layers: List[Layer], learning_rate: float):
        self.learning_rate = learning_rate
        self.trainable_layers = trainable_layers
    
    def step(self):
        raise NotImplementedError

class AdamOptimizer(Optimizer):
    """
    I store m and v values for each weight gradient and bias gradient
    in dictionaries m and v where the key is the id of the parameter matrix
    which has the form "dW{layer_id}" or "db{layer_id}". Thus, the parameter
    matrix ids serve as gradient matrix ids.
    
    This solution works but I'm not sure that this approach could be used in a graph based computation. 

    There's also an ugly option to have an instance of AdamOptimizer for each parameter.
    """
    def __init__(self, trainable_layers: List[Layer], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        #!? maybe we should pass paramters and parameters' ids to the optimizer instead of passing layers
        self.trainable_layers = trainable_layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        for layer in self.trainable_layers:
            # ids of weights and biases are same as the ids of corresponding gradients
            #! At the moment each subclass of Layer has its own id generator.
            #! We should either put it in the root class or cobine id number
            #! with the name of the subclass:
            #! W_id = f"{layer.__class__.__name__}_dW_{layer.id}"
            W_id, b_id = layer.get_W_and_b_ids()
            #! I don't see any pros of using zeros_like instead of zeros, but decided to use it anyway.
            self.m[W_id] = np.zeros_like(layer.weights)
            self.m[b_id] = np.zeros_like(layer.bias)
            self.v[W_id] = np.zeros_like(layer.weights)
            self.v[b_id] = np.zeros_like(layer.bias)
        
    
    def update(self, gradient: np.ndarray, cache_id: str) -> None:
        self.m[cache_id] = self.beta1 * self.m[cache_id] + (1 - self.beta1) * gradient
        self.v[cache_id] = self.beta2 * self.v[cache_id] + (1 - self.beta2) * gradient ** 2
        
    def step(self) -> None:
        for layer in self.trainable_layers:
            #! Since np arrays are passed by reference the weights and bias
            # layer properties are going to be properly updated.
            # The loop approach seems to be more general.
            # On the other hand, the commented code below might be more readable
            # and it updates the parameters explicitly so we immediately see what's going on.

            """
            for gradient, cache_id, parameter in zip(layer.W_and_b_grad, layer.get_W_and_b_ids(), (layer.weights, layer.bias)):                
                self.update(gradient, cache_id)
                parameter -= self.learning_rate * self.m[cache_id] / (np.sqrt(self.v[cache_id]) + self.epsilon)
            """
            
            W_id, b_id = layer.get_W_and_b_ids()
            self.update(layer.weights_gradient, W_id)
            layer.weights -= self.learning_rate * self.m[W_id] / (np.sqrt(self.v[W_id]) + self.epsilon)
            self.update(layer.bias_gradient, b_id)
            layer.bias -= self.learning_rate * self.m[b_id] / (np.sqrt(self.v[b_id]) + self.epsilon)
            
        self.t += 1


class GradientDescentOptimizer(Optimizer):
    def __init__(self, trainable_layers: List[Layer], learning_rate: float = 0.001):
        self.trainable_layers = trainable_layers
        self.learning_rate = learning_rate

    def step(self) -> None:
        for layer in self.trainable_layers:
            layer.weights -= self.learning_rate * layer.weights_gradient
            layer.bias -= self.learning_rate * layer.bias_gradient
    


class Sequential:
    """
    Feed forward neural network stack.
    Attributes:
        n_neurons: a list of integers that defines the number of neurons
            in each layer including the input and output layers.
        activations: a list of constructors of activation functions. They
            are applied to the output of each layer. The length of
            activations should be equal to the len(n_neurons) - 1.
    """
    def __init__(self, n_neurons: List[int], activations: List[Layer]):
        
        self.n_neurons = n_neurons
        self.activations = activations
        self.trainable_layers = []
        self.layers = []
        for i in range(len(n_neurons) - 1):
            fcl = FullyConnectedLayer(n_neurons[i], n_neurons[i+1])
            self.layers.append(fcl)
            self.trainable_layers.append(fcl)
            self.layers.append(activations[i]())
        
        
    
    def forward(self, input_: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input_ = layer.forward(input_)
        return input_
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers[:-1]):
            output_gradient = layer.backward(output_gradient)
        return output_gradient