# Реализация ResNet-101, Adam, AbaBound (pytorch vs numpy)

<!-- 
## Задание:
1. Скачайте датасет [CarDatasets](https://drive.google.com/drive/folders/1pkudEBabqbXMxRTgfGQs3e0VqfTjtqWU)
2. Реализуйте ResNet-101 с оптимизатором [Adabound](https://arxiv.org/abs/1902.09843v1) с использованием
Numpy и с **Torch**/Tensorflow/Jax
3. Оцените качество модели на тесте и сравните быстродействие
реализованных вариантов.
4. Запустить обучение на классическом Adam и сравнить сходимость
результатов с вариантом задания.
5. Сделайте отчёт в виде readme на GitHub, там же должен быть выложен
исходный код.
-->

## Теоретическая база
### ResNet-101
![Здесь должно быть изображение с архитектурой ResNet-101. Оно должжно быть в папке images_for_readme](./images_for_readme/ResNet-101_Architecture_half_size.png)

<!--
ResNet-101 включает в себя свертку conv1, макс пулинг и далее множество коллекций слоев convi. convi_x является bottleneck'ом. Bottlencek состоит из трех сверток: 1x1, 3x3, 1x1. Первая свертка понижает число выходных каналов, последняя повышает.
Помимо сверток Bottleneck имеет identity mapping (точная копия входа Bottleneck'а), который складывается с выходом последней свертки Bottleneck'а. В случае, когда число каналов identity mapping'a не совападает с числом каналов выхода последней свертки, перед складыванием с conv3 над identity mapping'ом производится свертка 1x1, приводящая его к необходимой размерности.

В conv1 размерность плоскости входного тензора уменьшается вдвое в связи с тем, что stride = 2. Перед conv2_1 производится даунсемплинг карты признаков (feature map) в 2 раза с помощью max pooling'а. Далее conv3_1, conv_4_1 и conv5_1 первая свертка bottleneck'а имеет stride = 2. Таким образом, ширина и высота сходного "изображения" сужаются в 32 раза перед тем как дойти до average pooling, который оставляет одно значение для каждого канала. Такой пулинг позволяет использовать входные данные произвольной размерности. Тем не менее, в связи с понижением размерности при проходе через сеть вход должен быть не менее 32 и, желательно, кратен 32 (иначе тензоры будут "обрезаться").
-->

## Реализация и обучение resnet-101 на numpy
### Описание разработанной системы (алгоритмы, принципы работы, архитектура)
Весь код, связанный с реализацией и обучением resnet-101 на numpy, в директории [./numpy_CNN](./numpy_CNN).

В файле [./numpy_CNN/numpy_resnet.py](./numpy_CNN/numpy_resnet.py) реализация resnet-101 на numpy. Точнее, там находятся:
* реалиация `Bottleneck` residual block'а
* класс `ResNet`, который собирает архитектуру получая на вход список количеств residual ботлнеков каждой конфигурации 
* Функция `resnet101` вызывающая конструктор класса `ResNet` с количествами ботлнеков: [3, 4, 23, 3]

Базовые модули сврточной нейронной сети, оптимизаторы и функция потерь реализованы в файле [./numpy_CNN/NumpyNN/NN_np](./numpy_CNN/NumpyNN/NN_np.py).

В моей реализации оптимизаторы получают на вход список слоёв. Каждый модуль нейронной сети (дочерние классы класса Layer, а также классы реализующие части нейронной сети или нейронную сеть целиком) имеют метод get_trainable_layers, возвращающий все обучаемые слои, входящие в состав модуля.

Реализованы классы:
* FullyConnectedLayer
* Conv2d — имплементация свёртки, где forward выполняется матричным умножением; backward градиент по весам считается матричным умноженим, градиент по входу временно считается в цикле. Подробнее в [./numpy_CNN/NumpyNN/README.md](./numpy_CNN/NumpyNN/README.md).
* Conv2dWithLoops — имплементация свертки на циклах.
* MaxPool2d
* Flatten
* ReLULayer
* SigmoidLayer
* CrossEntropyLossWithSoftMax
* AdamOptimizer
* GradientDescentOptimizer
* Sequential

В [./numpy_CNN/module_tests.ipynb](./numpy_CNN/module_tests.ipynb) производится проверка классов, реализованных на numpy путём сравнения результатов с аналогичными классами на pytorch. Для этого две реализации инициализируются одинаковыми весами (если речь об обучаемом модуле нейронной сети), в качетве входны данных и градиента по выходу генерируются тезоры случайных чисел. Сравниваются выходы, а также градиенты по весам, смещениям (bias) и входным данным. **В частности, было проверено, что resnet101, реализованный на numpy и на torch при одинаковых весах, входных данных и градиентах по выходным данным возвращают одинаковые выходные данные и одинаковые градиенты по входным данным**.

В [./numpy_CNN/numpy_CNN.ipynb](./numpy_CNN/numpy_CNN.ipynb) произведено обучение на датасете MNIST нейронной сети, состоящей из одого сверточного слоя, активации ReLU и одного полносвязного слоя. Активация последнего слоя - софтмакс, функция потерь - кросс-энтропия, оптимизатор - Адам. Результаты обучения на 3-ех эпохах представлены на графиках ниже. 

train:

![train loss](./images_for_readme/np_CNN_train_loss.png)
![train accuracy](./images_for_readme/np_CNN_train_accuracy.png)
![train f_score](./images_for_readme/np_CNN_train_fscore.png)

test:

![test loss](./images_for_readme/np_CNN_test_loss.png)
![test accuracy](./images_for_readme/np_CNN_test_accuracy.png)
![test f_score](./images_for_readme/np_CNN_test_fscore.png)

## Реализация resnet-101 на torch и сравнение обучения с использованием Adam и AdaBound
### Описание разработанной системы (алгоритмы, принципы работы, архитектура)
Весь код находится в директории [./pytorch_implementations](./pytorch_implementations).

В [.\pytorch_implementations\resnet-adam-vs-adabound.ipynb](.\pytorch_implementations\resnet-adam-vs-adabound.ipynb) планируется сравнение обучения на датасете [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) моей имплементации resnet101 с оптимизатором Adam и с AdaBound. На данный момент в этом jupyter блокноте произведено обучение с Adam, но не произведено обучение с AdaBound

В [.\pytorch_implementations\resnet.py](.\pytorch_implementations\resnet.py) Находится моя имплементация resnet на pytorch. Классы аналогичны описанным выше для numpy.

### Результаты работы и тестирования системы (скриншоты, изображения, графики, закономерности)

Ниже результаты обучения моей имплементации resnet101 на torch.

![train metrics](./images_for_readme/torch_resnet_train_metrics.png)
![val metrics](./images_for_readme/torch_resnet_val_metrics.png)

*график обучения более гладций, так как валидация производилась в 4 раза реже обучения*

### Выводы по работе

Очевидно, работать с моделями, используя фреймворки удобнее, так как они высокооптимизированы и поддерживают cuda.  

Как только обучу resnet101 с Adabound здесь будет вывод о том, какой оптимизатор в данной задаче был эффективнее.

## Использованные источники
1. [Adabound](https://arxiv.org/abs/1902.09843v1)
2. [Adam](https://arxiv.org/abs/1412.6980)
3. [ResNet](https://arxiv.org/pdf/1512.03385.pdf)

