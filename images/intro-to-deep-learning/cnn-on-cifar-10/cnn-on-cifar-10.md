基于 TensorFlow 使用卷积神经网络进行图片分类

## 任务描述

本次我们将会进行：
1. 使用 TensorFlow 定义 CNN 架构
2. 从零开始训练一个模型
3. 可视化训练的 卷积核

CIFAR-10 数据集 包含是个分类的 32x32 彩色图片 : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/cnn-on-cifar-10/1-cifar.png)

## 导入依赖初始化

```python
import sys
import download_utils # 这个文件是自定义的，下载数据的，可以自己下载

download_utils.link_all_keras_resources()
```


```python
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)

import keras_utils
from keras_utils import reset_tf_session
```

## 下载数据

```python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

这个过程比较慢，大家可以下载好了放在 ~/.keras/dataset/ 下面，注意文件名。

## 定义类别映射

```python
NUM_CLASSES = 10
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
```

## 展示图

```python
# show random images from train
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_train[random_index, :])
        ax.set_title(cifar10_classes[y_train[random_index, 0]])
plt.show()
```

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/cnn-on-cifar-10/2-show.png)

## 处理数据

首先归一化 

Xnorm = X/255 - 0.5

```python
#Dims are (50000, 32, 32, 3)
def normalize(x):
    return (x/255)-0.5
# normalize inputs
x_train2 = normalize(x_train)
x_test2 = normalize(x_test)

# 转化为 one-hot 编码
y_train2 = keras.utils.to_categorical(y_train)
y_test2 = keras.utils.to_categorical(y_test)
```

## 定义卷积神经网络的架构

我们结合 keras 定义

```python
# 导入必要的类
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
```

一般卷积网络通过不同的层构建：

- [Conv2D](https://keras.io/layers/convolutional/#conv2d)  卷积层
	- **filters**: 输出通道数
	- **kernel_size**: 指出卷积核的高和宽
	- **padding**: padding 的种类，padding="same" 会给边添加0, 输入输出等高等宽, padding='valid' 只有在 kernel 和i nput 完全 overlap 才会进行 convolution 操作。
	- **activation**: "relu", "tanh" 等.
	- **input_shape**: 输入的形状
- [MaxPooling2D](https://keras.io/layers/pooling/#maxpooling2d) - performs 2D max pooling.
- [Flatten](https://keras.io/layers/core/#flatten) - flattens the input, does not affect the batch size.
- [Dense](https://keras.io/layers/core/#dense) - 全连接层
- [Activation](https://keras.io/layers/core/#activation) - 激活函数层
- [LeakyReLU](https://keras.io/layers/advanced-activations/#leakyrelu) - relu activation.
- [Dropout](https://keras.io/layers/core/#dropout) - applies dropout.

我们的输入为 __(None, 32, 32, 3)__ ，输出的类别是 __(None, 10)__ 用来指定各个类别的概率，__None__ 代表了 batch 的数。

最终代码为：

```python
def make_model():
    """
    定义CNN架构
    """
    # 新建model
    model = Sequential()

	# 添加 卷积层，输入形状 (32,32,3)，卷积核高宽为 3，16个卷积核
    model.add(Conv2D(input_shape=(32,32,3),padding="same",kernel_size=3,filters=16))
    # 添加 leak relu 激活
    model.add(LeakyReLU(0.1))
    
    # 32 个核
    model.add(Conv2D(padding="same",kernel_size=3,filters=32))
    model.add(LeakyReLU(0.1))
    
    # 池化层
    model.add(MaxPooling2D(pool_size=2))
    # 池化层后加 dropout 层。
    model.add(Dropout(0.25))
    # 32 个核
    model.add(Conv2D(padding="same",kernel_size=3,filters=32))
    model.add(LeakyReLU(0.1))
    
    # 64 个核
    model.add(Conv2D(padding="same",kernel_size=3,filters=64))
    model.add(LeakyReLU(0.1))
    
    # 池化层
    model.add(MaxPooling2D(pool_size=2))
    # 池化层后加 dropout 层。
    model.add(Dropout(0.25))
    
    # 最后输出的时候将二维变一维
    model.add(Flatten())
    
    # 输出
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(LeakyReLU(0.1))
    
    # 最终用 softmax 分类
    model.add(Activation("softmax"))
    
    return model
```

打印一下我们的模型：

```python
# describe model
s = reset_tf_session()  # clear default graph
model = make_model()
model.summary()
```

输出：

```java
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 16)        448       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        4640      
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 32)        9248      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 16, 16, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               1048832   
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 10)                0         
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 1,084,234
Trainable params: 1,084,234
Non-trainable params: 0
_________________________________________________________________
```

## 训练模型

```python
INIT_LR = 5e-3  # 初始化 learning rate
BATCH_SIZE = 32 # 批次大小
EPOCHS = 10 # 每个训练周期

s = reset_tf_session()  # clear default graph

# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
model = make_model()  # define our model

# 准备我们的模型

model.compile(
    loss='categorical_crossentropy',  # 类型交叉熵
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))
```
    
设置检查点：
    
```python
# we will save model checkpoints to continue training in case of kernel death
model_filename = 'cifar.{0:03d}.hdf5'
last_finished_epoch = None


#### 下面的代码是用来从 checkoutpoint 训练的
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 7
# model = load_model(model_filename.format(last_finished_epoch))
```

开始训练

```python
# fit model
model.fit(
    x_train2, y_train2,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
               LrHistory(), 
               keras_utils.TqdmProgressCallback(),
               keras_utils.ModelSaveCallback(model_filename)],
    validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=0,
    initial_epoch=last_finished_epoch or 0
)

# 保存加载模型
model.save_weights("weights.h5")
model.load_weights("weights.h5")
```

训练过程比较慢，可能要一个多小时吧，这里大家需要等一段时间。

## 检查模型正确性

```python
# make test predictions
y_pred_test = model.predict_proba(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)
```

打印混淆矩阵:

```python
# confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), cifar10_classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), cifar10_classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))
```

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/cnn-on-cifar-10/3-maxtrix.png)

我们可以通过混淆矩阵看出，猫和狗比较难分辨，船和飞机也有一点难。

预测结果

```python
# inspect preditions
cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label = cifar10_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        true_label = cifar10_classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
               pred_label, pred_proba, true_label
        ))
plt.show()
```
	
![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/cnn-on-cifar-10/4-predict.png)

## 分析模型

我们现在有了模型，我们需要进行简单分析，最简单的就是看看 maximum stimuli

```python
s = reset_tf_session()  # clear default graph
K.set_learning_phase(0)  # disable dropout
model = make_model()
model.load_weights("weights.h5")  # that were saved after model.fit

# all weights we have
model.summary()
```

结果如下：

```java
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 16)        448       
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        4640      
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 32)        9248      
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 16, 16, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               1048832   
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570      
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 10)                0         
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 1,084,234
Trainable params: 1,084,234
Non-trainable params: 0
___________________________________________________________
```



定义函数：

```python

def find_maximum_stimuli(layer_name, is_conv, filter_index, model, iterations=20, step=1., verbose=True):
    
    def image_values_to_rgb(x):
        # normalize x: center on 0 (np.mean(x_train2)), ensure std is 0.25 (np.std(x_train2))
        # so that it looks like a normalized image input for our network
        x = ### YOUR CODE HERE

        # do reverse normalization to RGB values: x = (x_norm + 0.5) * 255
        x = ### YOUR CODE HERE
    
        # clip values to [0, 255] and convert to bytes
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # this is the placeholder for the input image
    input_img = model.input
    img_width, img_height = input_img.shape.as_list()[1:3]
    
    # find the layer output by name
    layer_output = list(filter(lambda x: x.name == layer_name, model.layers))[0].output

    # we build a loss function that maximizes the activation
    # of the filter_index filter of the layer considered
    if is_conv:
        # mean over feature map values for convolutional layer
        loss = K.mean(layer_output[:, :, :, filter_index])
    else:
        loss = K.mean(layer_output[:, filter_index])

    # we compute the gradient of the loss wrt input image
    grads = K.gradients(loss, input_img)[0]  # [0] because of the batch dimension!

    # normalization trick: we normalize the gradient
    grads = grads / (K.sqrt(K.sum(K.square(grads))) + 1e-10)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * (0.1 if is_conv else 0.001)

    # we run gradient ascent
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        if verbose:
            print('Current loss value:', loss_value)

    # decode the resulting input image
    img = image_values_to_rgb(input_img_data[0])
    
    return img, loss_value
```

```python
# sample maximum stimuli
def plot_filters_stimuli(layer_name, is_conv, model, iterations=20, step=1., verbose=False):
    cols = 8
    rows = 2
    filter_index = 0
    max_filter_index = list(filter(lambda x: x.name == layer_name, model.layers))[0].output.shape.as_list()[-1] - 1
    fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
    for i in range(cols):
        for j in range(rows):
            if filter_index <= max_filter_index:
                ax = fig.add_subplot(rows, cols, i * rows + j + 1)
                ax.grid('off')
                ax.axis('off')
                loss = -1e20
                while loss < 0 and filter_index <= max_filter_index:
                    stimuli, loss = find_maximum_stimuli(layer_name, is_conv, filter_index, model,
                                                         iterations, step, verbose=verbose)
                    filter_index += 1
                if loss > 0:
                    ax.imshow(stimuli)
                    ax.set_title("Filter #{}".format(filter_index))
    plt.show()
```


运行：

```python
# maximum stimuli for convolutional neurons
conv_activation_layers = []
for layer in model.layers:
    if isinstance(layer, LeakyReLU):
        prev_layer = layer.inbound_nodes[0].inbound_layers[0]
        if isinstance(prev_layer, Conv2D):
            conv_activation_layers.append(layer)

for layer in conv_activation_layers:
    print(layer.name)
    plot_filters_stimuli(layer_name=layer.name, is_conv=True, model=model)
leaky_re_lu_1
```

需要等待一段时间，然后可以看到如图提取的最大刺激：

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/cnn-on-cifar-10/5-mu.png)
