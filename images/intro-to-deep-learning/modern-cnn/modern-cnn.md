
# Training tips and tricks for deep CNNs

## sigmod activation

sigmod function 的导数在比较大或者比较小的时候会趋近于 0，也就是说这样会导致 vanishing gradients，梯度消失。
另外一个问题是它的输出不是 zero-centered，我们希望输入是 zero mean and standard variance，也就是归一化。
另外一个问题是指数的计算很复杂。

## ReLu function

也不是   zero-centered，如果一开始初值是0，那么将会一直不更新，也就是 dying RElu neuron。

## leaky Relu function

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/1-relu.png)


## weight init

常见初始化权重的方式我们可以在接下来进行介绍对比：

1. Zero: When all weights are set to 0
2. Random: When weights are set completely randomly
3. Random between -1 to +1: Random weights on the scale of -1 to +1
4. Xavier-Glorot Initialization [1]

首先我们基于 MNIST 的数据集构建类似 VGG-Net 卷积神经网络，运行数据得到如下的结果，这里的数据来自：https://medium.com/@amarbudhiraja/towards-weight-initialization-in-deep-neural-networks-908d3d9f1e02

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/2-compare.png)



![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/2-init.png)

如果都初始化为 0，最终训练出来的结果是一样的,这样你的神经网络就和一个线性模型一样。这是 symmetry problem，为了解决这个问题，我们通常需要从标准正太分布取值，然后乘以一个系数例如 0.03.

如果你随机初始化一个数据作为w，会有两个问题 vanishing gradients 和 exploding gradients，梯度爆炸或者梯度消失，所以需要归一化。

neuron 的输出是输入的线性组合加上一个激活函数，neuron 的输出会被下一层的 neuron 使用，所以我们希望输出的也是归一化的数据。所以我们希望我们的 weight 的 mean 是 0.

当你堆积很多层的时候，均值还是 0，但是 variance（方差）会发生变化。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/3-var.png)

如图为方差的计算公式，最终我们希望方差不会逐渐堆积，所以希望最后的括号内容是1 。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/4-var.png)

我们是从标准正太分布得到的w，方差是1，所以只需要乘以一个系数就满足方差是1，也就是 Xavier initialization。

其实还有很多初始化的方法，在 TensorFlow 中有一些函数，如：` W = tf.get_variable('W', [dims], initializer) where initializer = tf.contrib.layers.xavier_initializer()`

上面说了很多初始化 W，但是 b 却不用管，这是因为我们对 b 求偏导的时候，b 的偏导只依赖于 上一层的输入，不依赖下一层的导数，可以直接初始化为 0.

## batch normalization

上面介绍的是初始化的值，在训练过程中我们应该怎么办？

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/5-nor.png)


首先我们需要归一化 在 activation 之前的 neuron output ，也就是图中的 h。首先减去 mean 保证 zero mean 、除以方差保证 unit variance。
然后乘以 gamma 得到新的 variance，添加 beta 得到新的 mean。

这里面的 sigma 和 mu 哪里来的？基于当前的 batch 进行估计，而且我们可以在每一步 backpropagation 进行这个操作。

这里面的 gamma 和 beta 哪里来的？我也没太懂这句话，貌似是说这个 gamma 和 beta 都是需要通过训练得到的，其实就相当于添加了一层。使用 keras 训练时使用如下代码：

```python
from keras.layers import BatchNormalization
# 第一层
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 第二层
model.add(Dense(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
```

## dropout

另一种技术叫做 dropout，这个详情可以查看资料，实际上就是删掉部分节点，包括输入和隐藏层。

专业一点的说，每次训练的时候，每个节点都有 1-p 的概率被丢掉或者 p 的概率留下来。

在机器学习中，熟悉正则化的都知道可以通过添加 附加项防止过拟合。实际上深度学习通过 dropout 也可以防止过拟合。

Training Phase:
训练阶段：每个隐藏层 的 每次训练样本 的 每次迭代，都会有一定的概率 dropout（输出为0）。


Testing Phase:
测试阶段：使用所有的数，但是乘以 p (使得和训练时候一样).

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/6-dropout.png)

一些使用 dropout 的评论:
Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.（鲁棒性）
Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.（训练次数增量，但是时间少了）
With H hidden units, each of which can be dropped, we have 2^H possible models. In testing phase, the entire network is considered and each activation is reduced by a factor p.（模型种类增加）

下面是使用 Dropout 的 keras 代码：

```python
from keras import initializations
import copy
result = {}
y = {}
loss = []
acc = []
dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for dropout in dropouts:
    print "Dropout: ", (dropout)
    model = Sequential()                                               

    #-- layer 1
    model.add(Convolution2D(64, 3, 3,                                    
                            border_mode='valid',
                            input_shape=(3, img_rows, img_cols))) 
    model.add(Dropout(dropout))  
    model.add(Convolution2D(64, 3, 3))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))                                       
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ##--layer 2                        
    model.add(Convolution2D(128, 3, 3))
    model.add(Dropout(dropout)) 
    model.add(Activation('relu'))                                       
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ##--layer 3                         
    model.add(Convolution2D(256, 3, 3))
    model.add(Dropout(dropout)) 
    model.add(Activation('relu'))                                       
    model.add(MaxPooling2D(pool_size=(2, 2)))

    ##-- layer 4
    model.add(Flatten())                                                
    model.add(Dense(512))                                               
    model.add(Activation('relu'))                                       

    #-- layer 5
    model.add(Dense(512))                                                
    model.add(Activation('relu'))                                       

    #-- layer 6
    model.add(Dense(num_classes))                                       

    #-- loss
    model.add(Activation('softmax'))
        sgd = SGD(lr=learningRate, decay = lr_weight_decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    model_cce = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=20, verbose=1, shuffle=True, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    y[dropout] = model.predict(X_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    result[dropout] = copy.deepcopy(model_cce.history)   
    loss.append(score[0])
    acc.append(score[1])
    
print models
```

详情：https://github.com/budhiraja/DeepLearningExperiments/blob/master/Dropout%20Analysis%20for%20Deep%20Nets/Dropout%2BAnalysis.ipynb



## Augment

如果我们的数据太少了，我们怎么办？最好的办法是我们对数据进行 flip、rotation 等操作。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/7-augment.png)
