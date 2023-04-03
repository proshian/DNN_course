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
)

def test_layer(my_module_constructor: Callable,
               torch_module_constructor: Callable,
               my_module_params: dict | List,
               torch_module_params: dict | List,
               input_shape: Tuple[int, ...],
               output_shape: Tuple[int, ...],
               atol: float = 1e-5,
               random_sampler: Callable = np.random.rand) -> None:
    """
    Compares the output and gradients of a numpy layer and a torch layer
    """
    if isinstance(my_module_params, dict):
        my_module = my_module_constructor(**my_module_params)
    else:
        my_module = my_module_constructor(*my_module_params)

    if isinstance(torch_module_params, dict):
        torch_module = torch_module_constructor(**torch_module_params)
    else:
        torch_module = torch_module_constructor(*torch_module_params)
    
    
    # copy weights from the torch layer to the numpy layer if the numpy layer is trainable
    if isinstance(my_module, TrainableLayer):
        if isinstance(my_module, FullyConnectedLayer):
            my_module.weights = torch_module.weight.detach().numpy().T
            my_module.bias = torch_module.bias.detach().numpy().reshape(-1, 1).T
        elif isinstance(my_module, BatchNormalization2d):
            my_module.gamma = torch_module.weight.detach().numpy()
            my_module.beta = torch_module.bias.detach().numpy()
            my_module.running_mean = torch_module.running_mean.detach().numpy()
            my_module.running_var = torch_module.running_var.detach().numpy()
        else:
            my_module.weights = torch_module.weight.detach().numpy()
            if my_module.bias is not None:
                my_module.bias = torch_module.bias.detach().numpy()


    input_np = random_sampler(*input_shape).astype(np.float32)
    input_torch = torch.from_numpy(input_np)
    input_torch.requires_grad = True

    output_np = my_module.forward(input_np)
    output_torch = torch_module(input_torch)

    assert np.allclose(output_np, output_torch.detach().numpy(), atol=atol), "Outputs are not equal"
    print("Outputs are equal")

    output_grad_np = random_sampler(*output_shape)
    output_grad_torch = torch.from_numpy(output_grad_np)

    input_grad_np = my_module.backward(output_grad_np)
    output_torch.backward(output_grad_torch)
    input_grad_torch = input_torch.grad.detach().numpy()

    assert np.allclose(input_grad_np, input_grad_torch, atol=atol), "Input gradients are not equal"
    print("Input gradients are equal")


    # compare weight and bias gradients
    if isinstance(my_module, TrainableLayer):
        if isinstance(my_module, FullyConnectedLayer):
            weight_grad_np = my_module.weights_gradient
            weight_grad_torch = torch_module.weight.grad.detach().numpy().T
            bias_grad_np = my_module.bias_gradient
            bias_grad_torch = torch_module.bias.grad.detach().numpy().reshape(-1, 1).T
        elif isinstance(my_module, BatchNormalization2d):
            weight_grad_np = my_module.gamma_gradient
            weight_grad_torch = torch_module.weight.grad.detach().numpy()
            bias_grad_np = my_module.beta_gradient
            bias_grad_torch = torch_module.bias.grad.detach().numpy()
        else:
            weight_grad_np = my_module.weights_gradient
            weight_grad_torch = torch_module.weight.grad.detach().numpy()
            if my_module.bias is not None:
                bias_grad_np = my_module.bias_gradient
                bias_grad_torch = torch_module.bias.grad.detach().numpy()
            
        assert np.allclose(weight_grad_np, weight_grad_torch, atol=atol), "Weight gradients are not equal"
        print("Weight gradients are equal")

        if my_module.bias is not None:
            assert np.allclose(bias_grad_np, bias_grad_torch, atol=atol), "Bias gradients are not equal"
            print("Bias gradients are equal")