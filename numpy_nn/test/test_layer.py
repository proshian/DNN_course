import sys
import  os
project_root = os.path.dirname(os.path.dirname(sys.path[0]))
if project_root not in sys.path:
    sys.path.append(project_root)




import unittest
from datetime import datetime
import pickle
from typing import Callable, Tuple, Optional, Any

import numpy as np
import torch

from numpy_nn.modules.np_nn import (
    FullyConnectedLayer,
    BatchNormalization2d,
    TrainableLayer,
    Module,
)


class TestLayer(unittest.TestCase):
    def _get_default_test_data_save_path(self):
        failed_tests_dir_name = 'failed_tests_dumps'
        failed_tests_path = os.path.join('.',
                                         'numpy_nn',
                                         'test',
                                         failed_tests_dir_name)
        if not os.path.exists(failed_tests_path):
            # ! добавить сюда логирование
            os.makedirs(failed_tests_path)
        str_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        this_test_name = f'test_{str_date}.pickle'
        this_test_path = os.path.join(failed_tests_path, this_test_name)
        return this_test_path

    def assertNpCloseWithDumping(self,
                                 arr1,
                                 arr2,
                                 atol,
                                 msg: str,
                                 save_obj: Any = None,
                                 save_path: Optional[str] = None):
        expression_result = np.allclose(arr1, arr2, atol=atol)

        self.assertTrueWithDumping(expression_result, msg, save_obj, save_path)

        # if not expression_result:
        #     if not save_path:
        #         save_path = self._get_default_test_data_save_path()
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(save_obj, f)

        
        # self.assertTrue(
        #     np.allclose(arr1, arr2, atol=atol),
        #     msg
        # )
    
    def assertTrueWithDumping(self,
                              expression_result: bool,
                              msg: str,
                              save_obj: Any = None,
                              save_path: Optional[str] = None):
        if not expression_result:
            if not save_path:
                save_path = self._get_default_test_data_save_path()
            with open(save_path, 'wb') as f:
                pickle.dump(save_obj, f)
        
        self.assertTrue(
            expression_result,
            msg
        )


    def _copy_parameters(self, my_module: Module, torch_module: torch.nn.Module) -> None:
        if isinstance(my_module, FullyConnectedLayer):
            my_module.weights = torch_module.weight.detach().numpy().T
            my_module.bias = torch_module.bias.detach().numpy().reshape(-1, 1).T
        elif isinstance(my_module, BatchNormalization2d):
            n_channels = my_module.n_channels
            my_module.gamma = torch_module.weight.detach().numpy().reshape(1, n_channels, 1, 1)
            my_module.beta = torch_module.bias.detach().numpy().reshape(1, n_channels, 1, 1)
            my_module.running_mean = torch_module.running_mean.detach().numpy().reshape(1, n_channels, 1, 1)
            my_module.running_var = torch_module.running_var.detach().numpy().reshape(1, n_channels, 1, 1)
        else:
            my_module.weights = torch_module.weight.detach().numpy()
            if my_module.bias is not None:
                my_module.bias = torch_module.bias.detach().numpy()



    def _test_module(self,
                    my_module: Module,
                    torch_module: torch.nn.Module,
                    input_np: np.ndarray,
                    dJ_dout: np.ndarray,
                    atol: float = 1e-5,
                    skip_parameter_copying: bool = False,
                    print_tensors: bool = False,
                    print_results: bool = False) -> None:
        """
        Compares the output and (dJ/dW, dJ/d_input) of numpy and torch layer

        Args:
            my_module: neural network layer implemented in numpy.
            torch_module: neural network layer implemented in torch.
            input_shape: shape of the input tensor.
            output_shape: shape of the output tensor. It's used to generate
                a random tensor representing partial derivative of the loss
                function with respect to the output of the layer.
            atol: absolute tolerance for comparing
                numpy and torch tensors (used in np.allclose).
            random_sampler: function that generates random tensors of the given shape.
            skip_parameter_copying: if True, the weights and biases will be held intact.
                By default, weights and biases are copied from torch_module to my_module.
        """    
        # For dumping in assertNpCloseWithDumping
        mio = {
            'my_module': my_module,
            'torch_module': torch_module,
            'input_np': input_np,
            'dJ_dout': dJ_dout
        }

        # copy weights from torch_module to my_module
        # if the numpy layer is trainable
        if not skip_parameter_copying and isinstance(my_module, TrainableLayer):
            self._copy_parameters(my_module, torch_module)

        input_torch = torch.from_numpy(input_np)
        input_torch.requires_grad = True

        output_np = my_module.forward(input_np)
        output_torch = torch_module(input_torch)


        if print_tensors:
            print("my and torch outputs:")
            print(output_np.flatten(), output_torch.detach().numpy().flatten())
        
        self.assertNpCloseWithDumping(
            output_np,
            output_torch.detach().numpy(),
            atol,
            "Outputs are not equal",
            mio
        )

        if print_results:
            print("Outputs are equal")

        output_grad_np = dJ_dout
        output_grad_torch = torch.from_numpy(output_grad_np)

        input_grad_np = my_module.backward(output_grad_np)
        output_torch.backward(output_grad_torch)
        input_grad_torch = input_torch.grad.detach().numpy()

        if print_tensors:
            print("my and torch input gradients:")
            print(input_grad_np.flatten(), input_grad_torch.flatten())
        self.assertNpCloseWithDumping(
            input_grad_np,
            input_grad_torch,
            atol,
            "Gradients w.r.t input data are not equal",
            mio
        )
        if print_results:
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
            weight_grad_np = my_module.gamma_gradient.flatten()
            weight_grad_torch = torch_module.weight.grad.detach().numpy()
            bias_grad_np = my_module.beta_gradient.flatten()
            bias_grad_torch = torch_module.bias.grad.detach().numpy()

            if print_tensors:
                print("my and torch running means:")
                print(my_module.running_mean.flatten(), torch_module.running_mean.detach().numpy().flatten())
            self.assertNpCloseWithDumping(
                my_module.running_mean.flatten(),
                torch_module.running_mean.detach().numpy().flatten(),
                atol,
                "Running mean is not equal",
                mio
            )
            if print_results:
                print("Running means are equal")

            if print_tensors:
                print("my and torch running vars:")
                print(my_module.running_var.flatten(), torch_module.running_var.detach().numpy().flatten())
            self.assertNpCloseWithDumping(
                my_module.running_var.flatten(),
                torch_module.running_var.detach().numpy().flatten(),
                atol,
                "Running var is not equal",
                mio
            )
            if print_results:
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
            
        self.assertTrueWithDumping(
            weight_grads_close,
            "Gradients w.r.t. weights are not equal",
            mio)
        if print_results:
            print("Weight gradients are equal")

        if isinstance(my_module, BatchNormalization2d) or my_module.bias is not None:
            if print_tensors:
                print("my and torch bias gradients:")
                print(bias_grad_np.flatten(), bias_grad_torch.flatten())

            self.assertTrueWithDumping(
                np.allclose(bias_grad_np, bias_grad_torch, atol=atol),
                "Gradients w.r.t. biases are not equal",
                mio)
            if print_results:
                print("Bias gradients are equal")



    def _test_module_randomly(self,
                             my_module: Module,
                             torch_module: torch.nn.Module,
                             input_shape: Tuple[int, ...],
                             output_shape: Tuple[int, ...],
                             atol: float = 1e-5,
                             random_sampler: Callable = np.random.rand,
                             skip_parameter_copying: bool = False,
                             print_tensors: bool = False,
                             print_results: bool = False) -> None:
        
        input_np = random_sampler(*input_shape).astype(np.float32)
        dJ_dout = random_sampler(*output_shape)

        self._test_module(my_module, torch_module, input_np, dJ_dout, atol,
                         skip_parameter_copying,
                         print_tensors,
                         print_results)



    # def test_stack_of_layers(
    #     my_stack_of_layers: List[Module],
    #     torch_module_constructor: List[torch.nn.Module],
    #     input_shape: Tuple[int, ...],
    #     output_shape: Tuple[int, ...],
    #     atol: float = 1e-5,
    #     random_sampler: Callable = np.random.rand):

    #     """
    #     Compares the output and all gradients of a my layer stack and torch layer stack
    #     """

    #     input_np = random_sampler(*input_shape).astype(np.float32)
    #     input_torch = torch.from_numpy(input_np)
    #     input_torch.requires_grad = True

    #     output_np = input_np
    #     output_torch = input_torch
    #     for my_layer, torch_layer in zip(my_stack_of_layers, torch_module_constructor):
    #         output_np = my_layer.forward(output_np)
    #         output_torch = torch_layer(output_torch)

    #     assert np.allclose(output_np, output_torch.detach().numpy(), atol=atol), "Outputs are not equal"

    #     output_grad_np = random_sampler(*output_shape)
    #     output_grad_torch = torch.from_numpy(output_grad_np)

    #     input_grad_np = output_grad_np
    #     for my_layer, torch_layer in zip(my_stack_of_layers[::-1], torch_module_constructor[::-1]):
    #         input_grad_np = my_layer.backward(input_grad_np)
    #         output_torch.backward(output_grad_torch)
    #         input_grad_torch = input_torch.grad.detach().numpy()

    #     assert np.allclose(input_grad_np, input_grad_torch, atol=atol), "Input gradients are not equal"

    #     # compare weight and bias gradients
    #     for my_layer, torch_layer in zip(my_stack_of_layers, torch_module_constructor):
    #         if isinstance(my_layer, TrainableLayer):
    #             if isinstance(my_layer, FullyConnectedLayer):
    #                 weight_grad_np = my_layer.weights_gradient
    #                 weight_grad_torch = torch_layer.weight.grad.detach().numpy().T
    #                 bias_grad_np = my_layer.bias_gradient
    #                 bias_grad_torch = torch_layer.bias.grad.detach().numpy().reshape(-1, 1).T
    #             elif isinstance(my_layer, BatchNormalization2d):
    #                 weight_grad_np = my_layer.gamma_gradient
    #                 weight_grad_torch = torch_layer.weight.grad.detach().numpy()
    #                 bias_grad_np = my_layer.beta_gradient
    #                 bias_grad_torch = torch_layer.bias.grad.detach().numpy()
    #             else:
    #                 weight_grad_np = my_layer.weights_gradient
    #                 weight_grad_torch = torch_layer.weight.grad.detach().numpy()
    #                 if my_layer.bias is not None:
    #                     bias_grad_np = my_layer.bias_gradient
    #                     bias_grad_torch = torch_layer.bias.grad.detach().numpy()

    #             assert np.allclose(weight_grad_np, weight_grad_torch, atol=atol), "Weight gradients are not equal"

    #             if my_layer.bias is not None:
    #                 assert np.allclose(bias_grad_np, bias_grad_torch, atol=atol), "Bias gradients are not equal"