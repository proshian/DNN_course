# ResNet-101 numpy

ResNet-101 using numpy only

## Root directory structure
* [numpy_resnet_mnist.ipynb](./numpy_resnet_mnist.ipynb) - [numpy resnet implementation](./numpy_nn/models/resnet.py) training on MNIST using Adam optimizer
* [numpy_nn](./numpy_nn)
    * Numpy implementation of:
        * resnet101
        * all layers needed to create resnet101
        * optimizers (SGD, Adam)
        * CE Loss
    * Testing of all listed against pytorch implementations
* [pytorch_nn](./pytorch_nn) - pytorch resnet101 implementation
* Other directories:
    * [utils](./utils) - python modules that are helpful in jupyter notebooks. For example, [utils/plot.py](./utils/plot.py) contains functions for plotting epoch histories

[numpy_nn](./numpy_nn/) and [pytorch_nn](./pytorch_nn) directories should be treated as libraries.

## Theory
### ResNet-101
![There should be a picture of ResNet-101 architecture. It should be in images_for_readme directory](./images_for_readme/resnet101_architecture.svg)

### Adam
![Adam](./images_for_readme/Adam.png)

<!--
ResNet-101 включает в себя свертку conv1, макс пулинг и далее множество коллекций слоев convi. convi_x является bottleneck'ом. Bottlencek состоит из трех сверток: 1x1, 3x3, 1x1. Первая свертка понижает число выходных каналов, последняя повышает.
Помимо сверток Bottleneck имеет identity mapping (точная копия входа Bottleneck'а), который складывается с выходом последней свертки Bottleneck'а. В случае, когда число каналов identity mapping'a не совападает с числом каналов выхода последней свертки, перед складыванием с conv3 над identity mapping'ом производится свертка 1x1, приводящая его к необходимой размерности.

В conv1 размерность плоскости входного тензора уменьшается вдвое в связи с тем, что stride = 2. Перед conv2_1 производится даунсемплинг карты признаков (feature map) в 2 раза с помощью max pooling'а. Далее conv3_1, conv_4_1 и conv5_1 первая свертка bottleneck'а имеет stride = 2. Таким образом, ширина и высота сходного "изображения" сужаются в 32 раза перед тем как дойти до average pooling, который оставляет одно значение для каждого канала. Такой пулинг позволяет использовать входные данные произвольной размерности. Тем не менее, в связи с понижением размерности при проходе через сеть вход должен быть не менее 32 и, желательно, кратен 32 (иначе тензоры будут "обрезаться").
-->

## Resnet-101 numpy implementation (numpy_nn content) and trainig
### Basic Layers
Basic Layers of a neural network, optimizers and loss function are implemented in the directory [./numpy_nn/modules](./numpy_nn/modules).

Implemented classes include:
* FullyConnectedLayer
* Conv2d — convolution implementation using matrix multiplication. More details in [./numpy_nn/modules/README.md](./numpy_nn/modules/README.md).
* Conv2dWithLoops — convolution implementation using loops.
* MaxPool2d
* Flatten
* ReLULayer
* SigmoidLayer
* CrossEntropyLossWithSoftMax
* AdamOptimizer
* GradientDescentOptimizer
* Sequential
* GlobalAveragePooling2D
* BatchNormalization2d


The convolution implementation based on matrix multiplication has led to a **more than 34-fold acceleration in the training of ResNet101 on MNIST** relative to a naive implementation. The screenshot below illustrates that previously one epoch took over 114 hours, and now it's around 3 hours.

![performance comparison](./images_for_readme/performance.png)


Currently, optimizers take a list of neural network layers as input. This is necessary because, at the moment, to obtain the current partial derivatives of the loss function with respect to the parameters, they are requested from the layer since gradients are not changed in place. Each neural network module (child classes of the Module class, including basic layers, classes implementing parts of the neural network or entire neural networks) has a method called `get_trainable_layers`, which returns all trainable layers within the module. The output of this method is passed to the optimizer's constructor. The [optimizers_take_parameters](https://github.com/proshian/DNN_course_ITMO_2022/tree/optimizers_take_parameters) branch aims to rewrite toe code so that there are classes Parameters and TrainableParameters that always store current parametrs and gradient related to a corresponding nn.Module.


### ResNet-101
In the module [numpy_nn/models/resnet.py](./numpy_nn/models/resnet.py), the implementation of ResNet-101 in numpy can be found. It includes:
* The `Bottleneck` residual block class
* The `ResNet` class, which constructs the architecture by receiving a list of the numbers of residual bottlenecks
* The function `resnet101` that calls the `ResNet` constructor with a correct list of bottelneck numbers: [3, 4, 23, 3]

<!-- Also in the directory [numpy_nn/models](./numpy_nn/models/), there is an implementation of [resnet101 without batch normalization](./numpy_nn/models/resnet_without_batchnorm.py) -->


### Testing
The directory [numpy_nn/test](./numpy_nn/test/) is dedicated to testing classes that implement neural network modules in numpy.

For testing, pytorch implementations are used. When a trainable module is tested, both implementations are initialized with the same weights. Random tensors are generated as input data and the partial derivative of the loss function with respect to the module's output. Forward and backward passes are performed, and outputs, as well as partial derivatives of the loss function with respect to weights, biases, and input data, are compared.

The pseudo-code for a test looks like this:
```python
MY_MODULE.init_weights_with(TORCH_MODULE)

input_ = np.random.rand(INPUT_SHAPE)

input_torch = torch.from_numpy(input_).float()
input_torch.requires_grad = True

d_J_d_out = np.random.rand(OUTPUT_SHAPE)

out = MY_MODULE.forward(input_)
torch_out = TORCH_MODULE(input_torch)

d_J_d_in = MY_MODULE.backward(d_J_d_out)
torch_out.backward(torch.tensor(d_J_d_out), retain_graph=True)

print("out all close:", np.allclose(out, torch_out.detach().numpy()))
print("d_J_d_in all close:", np.allclose(d_J_d_in, input_torch.grad.detach().numpy()))
```

Testing is carried out using the unittest library. All classes that test a neural network module are subclasses of the TestLayer class defined in [test_layer.py](./numpy_nn/test/test_layer.py). The algorithm described above is implemented in the _test_module method of the TestLayer class. Most nn module tests are implemented as separate scripts in the [test](./numpy_nn/test) directory. However, some tests are temporarily performed in [module_tests.ipynb](./numpy_nn/test/module_tests.ipynb) jupyter notebook. Theese tests don't use unittest. They will be rewritten and moved to separate scripts.

If the tests pass successfully, the console displays an "OK" message. Otherwise, an error message is displayed, and a pickle file is saved in the [test/failed_tests_dumps](./numpy_nn/test/failed_tests_dumps) directory. This file contains a dictionary with keys 'my_module', 'torch_module', 'input_np', 'dJ_dout', allowing the reproduction of the failed test.

With each pull request to the main branch, the `compare_with_pytorch` workflow is triggered. It runs all tests and produces a coverage report.

All basic neural network modules, except batch normalization, have results (partial derivatives of the module's output with respect to all parameters and input data) matching PyTorch up to 6 decimal places. Batch normalization results match up to 4 decimal places.

The output and partial derivative of the loss function with respect to the input data of numpy ResNet101 match with its pytorch counterpart up to 6 decimal places.

### Training
Training is performed in [./numpy_resnet_mnist.ipynb](./numpy_resnet_mnist.ipynb). ResNet-101 numpy implementation is trained on MNIST dataset. This notebook also features code that defines and traines a small convolutional neural network to demonstrate how to use the implemented classes in a general case.

Training results of numpy resnet implementation are shown on the graph below.

![numpy resnet-101 results](./images_for_readme/numpy_resnet_results.png)


### Trained model
The train function in [./numpy_resnet_mnist.ipynb](./numpy_resnet_mnist.ipynb) generates a model_optimizer_history dictionary after each epoch. This dictionary contains the following keys:
* 'model': model (in state)
* 'optimizer': optimizer (in state)
* 'epoch_history': training history - a dictionary with keys corresponding to the training phase (in this case, 'train' and 'test'), and values - dictionaries reflecting the history of the phase. These nested dictionaries have metric names as keys and lists with metric values for each epoch as values
* 'model_info': a dictionary with information about the model and hyperparameters of training (for example, the batch size used)

The model_optimizer_history dictionary is saved to the file `./numpy_nn/models/trained_models/resnet101/model_optimizer_history_dict.pickle`. It allows you to continue training from the epoch at which it was interrupted or output a graph with the training history.

The `model_optimizer_history_dict.pickle` is not present in this github repository. It is versioned via DVC and stored in a remote storage. To get the version of the `model_optimizer_history_dict.pickle` file that is relevant to the version of the repository you are in, you need to go to the repository in the terminal and run the command `dvc pull`. After that, the `model_optimizer_history_dict.pickle` file will appear in the `./numpy_nn/models/trained_models/resnet101/` directory. Note that it will contain an up-to-date version of the file, meaning that if you run training, in `numpy_resnet_mnist.ipynb`, you will get a dictionary with the same structure, the same keys, the same values (the same neural network weights, and the same optimizer state).

There are two reasons why model and optimizer are saved in the same structure:
1. Encapsulation: there is no question of which optimizer file belongs to which model file
2. If saved separately, the links between the model parameters and the optimizer are broken. To be more precise, between the layers of the neural network and the optimizer, because in this implementation, the optimizer requests the current parameters and gradients from layers. This means that identical layers would be stored in the optimizer and in the model, but they would be different objects and optimizer won't update model parameters. The solution to the problem would be to execute `optimizer.trainable_layers = model.trainable_layer` after loading the optimizer and model.
3. Since the parameters need to be stored in both the model and the optimizer, saving to separate files would take up more memory

## resnet-101 pytorch implementation

[./pytorch_nn/models/resnet.py](./pytorch_nn/models/resnet.py) implements resnet-101 using pytorch. It contains `Bottleneck` and `ResNet` classes and `resnet101` function similar to their numpy counterparts described above.

## Conclusions
Implementing models in numpy is a captivating exercise that helps structure knowledge about neural networks and make sure that you fully understand how they work. 

Obviously, using numpy for real projects is not recommended. Highly optimized frameworks like pytorch, tensorflow, jax, etc are much more convenient and efficient (also note that numpy doesn't support GPU).

Switching to a Conv2d implementation based on matrix multiplication leads to a much faster backpropagation as shown at the end of [./numpy_nn/test/module_tests.ipynb](./numpy_nn/test/module_tests.ipynb). For example, with parameters n_input_channels = 4,n_output_channels = 2, width = 3, height = 5, kernel_size = 3, stride = 1, padding = 3, and batch size = 8, 1000 iterations of backpropagation on official pytorch implementation take 1.2 seconds, 4.2 seconds on a matrix convolution implementation, and 20.7 seconds on loops based numpy implementation.

<!-- Изначально моя имплементцаия resnet-101 не содержала батч-нормализацию. Ее использование ускорило обучение  -->

<!-- При обучении моей имплементации c Adabound функция потерь падает невероятно медленно и по сравненю с Adam выглядит как прямая линия. -->

<!-- Обучение официально имлементации resnet101 тоже было медленне с AdaBound. -->

<!-- В данном эксперименте не было выявлено заявленных преимуществ AdaBound. -->

## Sources
1. [Adam](https://arxiv.org/abs/1412.6980)
2. [ResNet](https://arxiv.org/pdf/1512.03385.pdf)


# TODO

* Create a batchnorm implementation that would pass the tests with accuracy 1e-6
* add loss test funciton
* Check if test_stack_of_layers works
* Return batchnorm to resnet when it's fixed

Minor todo tasks:
<!-- * Добавить нормализацию изображений Stanford Cars датасета
* Так как машины не квадратные, возможно, лучше приводить к размеру 64x96 -->

* Create methods for saving model parameters (or trained model layers) to a file and loading from a file. Thus we would avoid saving useless data that wastes space. Another reson is that pickling whole model may result in an outdated version if we make changes to the model class (add new methods for example).  
* Make a Conv2d variation, where forward does not store the transformed input, and backward applies the transformation to the original input. This would be a bit slower, but would save a lot of memory
* Maybe generalise batchnorm (to work for any dimensionality). For example, make backward as here: https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L969-L1215


## Other branches:
* `adabound-and-batchnorm` - returning batchnorm toresnet + experiments that compare Adam and Adabound
* `optimizers_take_parameters` - rewriting the code so that optimizers take neural network parameters as arguments instead of trainable layers