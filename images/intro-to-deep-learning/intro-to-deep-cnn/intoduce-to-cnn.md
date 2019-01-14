

# Motivation for convolutional layers

假设我们已经知道了 MLP(Multi Layer Perceptron，多层神经网络)的基础知识，这时候我们需要处理 vision（计算机视觉）的问题，我们应该怎么办？
这篇文章，我们会介绍一种专门处理图片输入的的神经网络层。

## What is image input?
我们先看一种简单的图片，gray-scale image.图片实际上就是一个像素矩阵，Dimensions of these metrics are called image resolution.
一个 300 * 300 的图片可以表示为 300 * 300 的像素矩阵，每个元素存储对应位置的亮度，值可以从0-155，0代表黑色，

彩色图片则是三维的张量（tensor），每个点代表红黄蓝的程度.当我们要做神经网络处理，首先要进行normalized（归一化）.

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//1-normailize.png)

这样我们的数据均值是0而且是归一化的。
我们知道了 MLP，我们可以用mlp来处理这个事情:
![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//2-mlp.png)

每个像素点作为一个输入，perceptron（感知）会将所有的像素值作为输入，乘以 weight W，加上 bias b，然后加上一个 activation function。这样可以么？No！

我们希望训练一个 cat detector, 
![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//3-cat.png)

上面的图中红色特征学到了猫，但是如果猫位置改变了，下面的图，我们需要这些绿色的特征，我们没有完全使用我们学到的特征。


## we have convolution

我们可以使用 convolutions（卷积），大家这里不要被卷积吓到了，或者被迷惑了，实际上这个卷积和真正的卷积公式并不是很相似。

卷积是 一个 kernel（或者filter）与图片上面某一部分的乘积。也被称为 `local receptive field of the same size`. 我们查看一个例子，

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//4-dot.png)


一张图片，我们有一个sliding window（滑动窗口）, 红色边框部分. 提取第一个 local receptive field.和 kernel 进行乘积计算，得到 5， Then we slide that window across the image, 在每个位置计算乘积. 

卷积实际上是经常用到的一个东西，在图片处理中，不同的卷积核有不同的效果。下图的三个卷积核分别有发现边、锐化、模糊的效果。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//5-con.png)

## property of convulotion

### Convolution is actually similar to correlation. 

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//6-con.png)

我们的输入图片有一个反斜杠，如图。我们的 kernel 也像一个反斜杠，我们最终的卷及结果会有两个地方不是0，如图中的红色红色边框，输出一个1，一个2，其余为0。

我们的另一个图片，slash并不是一个反的，二是正的斜杆， 用同一个kernel 处理, 输出是两个1，其余为0.

如果我们对 convolutional layer 使用 取最大值的激活函数，一个得到了1，一个得到了2。感觉我们将斜杆的方向进行了分类？


### Another interesting property of convolution is translation equivariance. 

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//7-con.png)


如图的两个输入，我们就是对斜杆移动了位置，结果是一样的，只是位置不一样。这个特效叫做 invariant to translation。

## How does convolutional layer in neural network works? 

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//8-padding.png)

### padding

为了使得 卷及操作前后的矩阵是一样大，我们会给图片周边补充一些0。图中的灰色部分。

### forward 

我们实际上就是对每个 perception field 就=进行一个卷积，然后加上 偏置 bias，再使用 activation function，例如 sigmod，然后移动 window 得到一个新的 neuron。
我们每次移动步长成为 stride，这里的 stride 是 1。注意每次都会共享参数W，b.然后得到新的 neurons ，新的层叫做 feature map，和输入层维度一样的矩阵。

在这个例子中，输入是 3x3，添加了 2x2 的padding，输出的 feature map 是 3*3。这样我们的参数就是 kernel 的 W1-W9，以及 bias b，十个参数。


# How does back propagation work for convolutional neural networks? 

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//9-back.png)


如图，我们 3x3的 inputs, 2x2的 convolution, 需要训练 2x2+1 个参数. 以最复杂的W4为了例，实际上有4个地方的用到了 W4，我们需要计算 DL/DW4, 使用 chain rule，需要计算 4 个的和。
我们直接加上他们的权重求和，加上共享同一个权重的所有变量的梯度. 


# Our first CNN architecture

## a color image input

前面说的输入是一个矩阵，原因是灰白图片，只有一层，而彩色图片有多维，这样我们的输入实际上是一个 三维张量。就像一个魔方一样，是立体的。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//11-color.png)

所以我们的 卷积核 kernel 也是三维的： Wk x Hk x Cin ,分别代表宽、高、input_channel.

## one kernel is not enough

我们使用上面说的方法输出只有一个图层，实际上我们需要输出多个图层，提取更多信息。如图：

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//12-kernel.png)

左边是一个卷积核，生成一个图层，我们需要 5 个这样的卷积核，才能生成 5 个这样的图层，也就是右边的 5 个 feature map，这里 的 5 称为 Cout。这样我们的参数个数就是：

```
(Wk * Hk * Cin +1) * Cout 
```

## one layer is not enough

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//12-recptive.png)

假设使用 3x3 的kernel，那么每个输出的单元能感受到的输入是 3x3 的，如果在加一层，实际上第三层能感受到第二层的 3x3 的范围，所以能感受到第一层 5x5 的范围。
如果有 n 层，能感受到 (2n+1)x(2n+1) 个来自第一层的field。如果我们希望最后的输出层能感受到来自输入层的每一个输出，那需要太多层了。

## bigger stride

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//13-stride.png)

如果使用更大的 stride，能减少很多次数。实际上这个想法让我们想到了 pooling layer。

## pooling layer

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//14-pooling.png)

池化层就是直接取每个区域最大值或者平均值，这样会丢失一些信息，但是很有用，大幅度减少了参数个数。

## back propagation

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//15-back.png)

最大值不是一个可微函数。但是可以模拟，如图，如果当前值不是最大值，那么偏导数为0，如果是最大值，偏导数为1.

## put together

所有的理论都有了，我们将他们放在一起。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//16-final.png)

如图，最下面是输入层，然后是六个卷积核得到的六个 feature map，是 convolution layer1，然后是 池化层。紧接着是 新的卷积层，和新的池化层

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-deep-cnn//17-final.png)

最后两层是 MLP，使用 softmax 分类。

