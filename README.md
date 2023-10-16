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

### Обучение
В [./numpy_resnet_mnist.ipynb](./numpy_resnet_mnist.ipynb) произведено обучение на датасете MNIST моей реализации resnet-101 на numpy. Также в этом файле предоставлен код для определения и обучения небольшой сверточной нейронной сети, чтобы продемонстрировать, как пользоваться реализованными класами.

Результаты обучения resnet на numpy на графикe ниже.

![numpy resnet-101 results](./images_for_readme/numpy_resnet_results.png)


### Обученная модель
В [./numpy_resnet_mnist.ipynb](./numpy_resnet_mnist.ipynb) после каждой эпохи обучения формируется словарь model_optimizer_history:
* 'model': модель в состоянии
* 'optimizer': оптимизатор в состоянии
* 'epoch_history': история обучения - словарь с ключами, соответствующими фазе обучения (в данном случае 'train' и 'test'), и значениями - словарями, отражающими историю фазы. Эти вложенные словари имеют названия метрик в качестве ключей и списки со значениями метрик на каждой эпохе в качестве значений.
* 'model_info': словарь с информацией о модели и о гиперпараметрах обучения (например, использованном батчсайзе) 

Словарь model_optimizer_history сохраняется в файл `./numpy_nn/models/trained_models/resnet101/model_optimizer_history_dict.pickle`. Благодаря этому словарю можно продолжить обучение с эпохи, на которой оно было прервано, вывести график с историей обучения. 

В github репозитории файл `model_optimizer_history_dict.pickle` отсутствует. Он версионируется с помощью DVC и хранится в удаленном хранилище. Чтобы получить версию файла `model_optimizer_history_dict.pickle`, актуальную для версии репозитория, в которой мы находимся,  необходимо в командной строке перейти в репозиторий и выполнить комманду `dvc pull`. После этого файл `model_optimizer_history_dict.pickle` появится в директории `./numpy_nn/models/trained_models/resnet101/`. При этом это будет актуальный словарь: если запустить обучение в `numpy_resnet_mnist.ipynb`, будет получен идентичный словарь с такими же значениями, такими же ключами, такой же структурой, такими же весами нейронной сети и таким же состоянием оптимизатора.

Нужно отметить, что есть две причины, по которым оптимизатор и модель сохраняются в одной структуре
1. Инкапсуляция: так не возникает вопроса какой файл оптимизатора относится к какому файлу модели
2. Если сохранять отдельно, нарушаются связи между параметрами модели и оптимизатором. Если быть точнее, между слоями нейронной сети и оптимизатором, потому что в данной имплементации оптимизатор запрашивает актуальные параметры у слоя (потому что мне сходу не пришло в голову, что я могу передавать параметры оптимизатору "по ссылке" если создам класс параметров). То есть и в оптимизаторе и в модели хранились бы идентичные слои, но являющиеся различными объектами. Решенем проблемы было бы выполнение  `optimizer.trainable_layers = model.trainable_layer` после загрузки оптимизатора и модели.
3. В связи с тем, что парметры нужно хранить и в модели и в оптимизаторе, раздельное сохранение затрачивало бы больше памяти


## Реализация resnet-101 на torch
<!-- Весь код находится в директории [./pytorch_nn](./pytorch_nn). -->

В [./pytorch_nn/models/resnet.py](./pytorch_nn/models/resnet.py) Находится моя имплементация resnet на pytorch. Классы аналогичны описанным выше для numpy.


## Выводы по работе
Очевидно, работать с моделями, используя фреймворки удобнее, так как они высокооптимизированы и поддерживают cuda.

Исползование реализации свертки в виде матричного умножения делает скорость обратного распространения значительно быстрее продемонстрировано в конце [./numpy_nn/test/module_tests.ipynb](./numpy_nn/test/module_tests.ipynb). Например, при параметрах n_input_channels = 4,n_output_channels = 2, width = 3, height = 5, kernel_size = 3, stride = 1, padding = 3 и batchsize = 8 1000 итераций обратного распространения на pytorch занимают 1.2 секунды, при матричной имлементации свертки - 4.2 секунды, а на циклах - 20.7 секунды.

<!-- Изначально моя имплементцаия resnet-101 не содержала батч-нормализацию. Ее использование ускорило обучение  -->

<!-- При обучении моей имплементации c Adabound функция потерь падает невероятно медленно и по сравненю с Adam выглядит как прямая линия. -->

<!-- Обучение официально имлементации resnet101 тоже было медленне с AdaBound. -->

<!-- В данном эксперименте не было выявлено заявленных преимуществ AdaBound. -->

## Использованные источники
1. [Adam](https://arxiv.org/abs/1412.6980)
2. [ResNet](https://arxiv.org/pdf/1512.03385.pdf)


# TODO

* Получить батчнормализацию, которая будет проходить тесты с точностью 1e-6

* add loss test funciton
* Check if test_stack_of_layers works

* Мб добавить в скрипты проверку, есть ли необходимые модули в sys.path, если нет, сделать добавление

* Когда батч-нормализация будет починена, удалить варианты resnet без батч-нормалищации 


Второстепенные todo задачи:

<!-- * Добавить нормализацию изображений Stanford Cars датасета
* Так как машины не квадратные, возможно, лучше приводить к размеру 64x96 -->
* Переписать [./numpy_CNN/NumpyNN/NN_np](./numpy_CNN/NumpyNN/NN_np.py), чтобы оптимизаторы принимали параметры, а не обучаемые слои. (Уже ведется работа в отдельном branch'е)
* Сделать методы сохранения параметров модели (или обучаемых слоев модели) в файл и загрузки из файла. Как минимум потому что обучаемые слои хранят входные данные => Если делать pickle модели целиком, записывется много бесполезной информации 
* Сделать вариант forward и backward Conv2d, где forward не сохраняет преобразованные input, а backward применяет преобразование к исходному input. Будет работать немного медленнее, но сильно сэкономит память
* Можно обобщить batchnorm (чтобы работал для любой размерности). Например, сделать backward как тут: https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L969-L1215


В ветке `adabound-and-batchnorm` ведется работа по добавлению в resnet батчевой нормализации, а также эксперименты по сравнению Adam и Adabound