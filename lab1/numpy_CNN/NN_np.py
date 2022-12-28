import numpy as np
import itertools
from typing import List, Tuple


# I need to implement a convolutional neural network with and adam optimizer
# using only numpy.  
# I will start by implementing a fully connected layer and build a plain neural network
# I think I need to implement a layer abstract class
# The layer class will have virtual methods for forward and backward pass
# Each succssor of the layer class will implement the forward and backward pass

# I will start with a fully connected layer without inheritance


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
        self.input_ = input_
        self.output = np.dot(self.input_, self.weights) + self.bias
        return self.output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # The math explanation: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        batch_size = output_gradient.shape[0]
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
        


class SoftMaxLayer(Layer):
    def forward(self, input_: np.ndarray) -> np.ndarray:
        self.output = np.exp(input_) / np.sum(np.exp(input_), axis=1, keepdims=True)
        return self.output
    
    #! backward needs to be checked
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.output * (1 - self.output)

        

class CrossEntropyLoss(Layer):
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        batch_size = pred.shape[0]
        # summing over the batch and then dividing by the batch size
        # The line below makes a division by zero error at some point
        return -np.sum(np.dot(self.target, np.log(self.pred).T)) / batch_size
    
    def backward(self) -> np.ndarray:
        return - self.target / self.pred


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

    There's also an ugly option to have an instance of AdamOptimizer for each.
    """
    #!? Maybe we should implement a class of a TrainableLayer that inherits from Layer. 
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
    This is a feed forward neural network stack.
    It is defined by two lists: one for the number of neurons in each layer
    and one for the activation function in each layer.
    """
    #! I think the activations should be a list of classes instead of a list of strings.
    # The classes should be inherited from an abstract Activation class.
    # The Activation class should be inherited from a Layer class.
    # The Layer class should have a forward and a backward method.
    # the af_name_to_af_class dictionary probably won't be needed.
    def __init__(self, n_neurons: List[int], activations: List[Layer]):
        """
        Attributes:
            n_neurons: a list of integers that defines the number of neurons
                in each layer including the input and output layers.
            activations: a list of constructors of activation functions. They
                are applied to the output of each layer. The length of
                activations should be equal to the len(n_neurons) - 1.

        """
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

