import sys
import  os
project_root = os.path.dirname(os.path.dirname(sys.path[0]))
if project_root not in sys.path:
    sys.path.append(project_root)



from typing import Callable, Tuple, List

import numpy as np
import torch

from numpy_nn.modules.np_nn import (
    FullyConnectedLayer,
    BatchNormalization2d,
    TrainableLayer,
    Module,
)


def test_module(my_module: Module, torch_module: torch.nn.Module,
                input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                atol: float = 1e-5, random_sampler: Callable = np.random.rand,
                skip_parameter_copying = False, print_tensors = False) -> None:
    """
    Compares the output and gradients of a numpy layer and a torch layer
    """    
    # copy weights from torch_module to my_module
    # if the numpy layer is trainable
    if not skip_parameter_copying and isinstance(my_module, TrainableLayer):
        if isinstance(my_module, FullyConnectedLayer):
            my_module.weights = torch_module.weight.detach().numpy().T
            my_module.bias = torch_module.bias.detach().numpy().reshape(-1, 1).T
        elif isinstance(my_module, BatchNormalization2d):
            n_channels = input_shape[1]
            my_module.gamma = torch_module.weight.detach().numpy().reshape(1, n_channels, 1, 1)
            my_module.beta = torch_module.bias.detach().numpy().reshape(1, n_channels, 1, 1)
            my_module.running_mean = torch_module.running_mean.detach().numpy().reshape(1, n_channels, 1, 1)
            my_module.running_var = torch_module.running_var.detach().numpy().reshape(1, n_channels, 1, 1)
        else:
            my_module.weights = torch_module.weight.detach().numpy()
            if my_module.bias is not None:
                my_module.bias = torch_module.bias.detach().numpy()


    input_np = random_sampler(*input_shape).astype(np.float32)
    input_torch = torch.from_numpy(input_np)
    input_torch.requires_grad = True

    output_np = my_module.forward(input_np)
    output_torch = torch_module(input_torch)


    if print_tensors:
        print("my and torch outputs:")
        print(output_np.flatten(), output_torch.detach().numpy().flatten())
    assert np.allclose(output_np, output_torch.detach().numpy(), atol=atol), "Outputs are not equal"
    print("Outputs are equal")

    output_grad_np = random_sampler(*output_shape)
    output_grad_torch = torch.from_numpy(output_grad_np)

    input_grad_np = my_module.backward(output_grad_np)
    output_torch.backward(output_grad_torch)
    input_grad_torch = input_torch.grad.detach().numpy()

    if print_tensors:
        print("my and torch input gradients:")
        print(input_grad_np.flatten(), input_grad_torch.flatten())
    assert np.allclose(input_grad_np, input_grad_torch, atol=atol), "Input gradients are not equal"
    print("Input gradients are equal")

    if not isinstance(my_module, TrainableLayer):
        return
    

    # compare weight and bias gradients
    if isinstance(my_module, FullyConnectedLayer):
        weight_grad_np = my_module.weights_gradient
        weight_grad_torch = torch_module.weight.grad.detach().numpy().T
        bias_grad_np = my_module.bias_gradient
        bias_grad_torch = torch_module.bias.grad.detach().numpy().reshape(-1, 1).T
    elif isinstance(my_module, BatchNormalization2d):
        n_channels = input_shape[1]
        weight_grad_np = my_module.gamma_gradient.flatten()
        weight_grad_torch = torch_module.weight.grad.detach().numpy()
        bias_grad_np = my_module.beta_gradient.flatten()
        bias_grad_torch = torch_module.bias.grad.detach().numpy()

        if print_tensors:
            print("my and torch running means:")
            print(my_module.running_mean.flatten(), torch_module.running_mean.detach().numpy().flatten())
        running_mean_close = np.allclose(
            my_module.running_mean.flatten(), torch_module.running_mean.detach().numpy().flatten(), atol=atol)
        assert running_mean_close, "Running mean is not equal"
        print("Running means are equal")

        if print_tensors:
            print("my and torch running vars:")
            print(my_module.running_var.flatten(), torch_module.running_var.detach().numpy().flatten())
        running_var_close = np.allclose(
            my_module.running_var.flatten(), torch_module.running_var.detach().numpy().flatten(), atol=atol)
        assert running_var_close, "Running var is not equal"
        print("Running vars are equal")
            
    else:
        weight_grad_np = my_module.weights_gradient
        weight_grad_torch = torch_module.weight.grad.detach().numpy()
        if my_module.bias is not None:
            bias_grad_np = my_module.bias_gradient
            bias_grad_torch = torch_module.bias.grad.detach().numpy()
    
    weight_grads_close = np.allclose(weight_grad_np, weight_grad_torch, atol=atol)

    if print_tensors:
        print("my and torch weight gradients:")
        print(weight_grad_np.flatten(), weight_grad_torch.flatten())
        
    assert weight_grads_close, "Weight gradients are not equal"
    print("Weight gradients are equal")

    if isinstance(my_module, BatchNormalization2d) or my_module.bias is not None:
        if print_tensors:
            print("my and torch bias gradients:")
            print(bias_grad_np.flatten(), bias_grad_torch.flatten())

        assert np.allclose(bias_grad_np, bias_grad_torch, atol=atol), "Bias gradients are not equal"
        print("Bias gradients are equal")




def test_stack_of_layers(
    my_stack_of_layers: List[Module],
    torch_module_constructor: List[torch.nn.Module],
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    atol: float = 1e-5,
    random_sampler: Callable = np.random.rand):

    """
    Compares the output and all gradients of a my layer stack and torch layer stack
    """

    input_np = random_sampler(*input_shape).astype(np.float32)
    input_torch = torch.from_numpy(input_np)
    input_torch.requires_grad = True

    output_np = input_np
    output_torch = input_torch
    for my_layer, torch_layer in zip(my_stack_of_layers, torch_module_constructor):
        output_np = my_layer.forward(output_np)
        output_torch = torch_layer(output_torch)

    assert np.allclose(output_np, output_torch.detach().numpy(), atol=atol), "Outputs are not equal"

    output_grad_np = random_sampler(*output_shape)
    output_grad_torch = torch.from_numpy(output_grad_np)

    input_grad_np = output_grad_np
    for my_layer, torch_layer in zip(my_stack_of_layers[::-1], torch_module_constructor[::-1]):
        input_grad_np = my_layer.backward(input_grad_np)
        output_torch.backward(output_grad_torch)
        input_grad_torch = input_torch.grad.detach().numpy()

    assert np.allclose(input_grad_np, input_grad_torch, atol=atol), "Input gradients are not equal"

    # compare weight and bias gradients
    for my_layer, torch_layer in zip(my_stack_of_layers, torch_module_constructor):
        if isinstance(my_layer, TrainableLayer):
            if isinstance(my_layer, FullyConnectedLayer):
                weight_grad_np = my_layer.weights_gradient
                weight_grad_torch = torch_layer.weight.grad.detach().numpy().T
                bias_grad_np = my_layer.bias_gradient
                bias_grad_torch = torch_layer.bias.grad.detach().numpy().reshape(-1, 1).T
            elif isinstance(my_layer, BatchNormalization2d):
                weight_grad_np = my_layer.gamma_gradient
                weight_grad_torch = torch_layer.weight.grad.detach().numpy()
                bias_grad_np = my_layer.beta_gradient
                bias_grad_torch = torch_layer.bias.grad.detach().numpy()
            else:
                weight_grad_np = my_layer.weights_gradient
                weight_grad_torch = torch_layer.weight.grad.detach().numpy()
                if my_layer.bias is not None:
                    bias_grad_np = my_layer.bias_gradient
                    bias_grad_torch = torch_layer.bias.grad.detach().numpy()

            assert np.allclose(weight_grad_np, weight_grad_torch, atol=atol), "Weight gradients are not equal"

            if my_layer.bias is not None:
                assert np.allclose(bias_grad_np, bias_grad_torch, atol=atol), "Bias gradients are not equal"