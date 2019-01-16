

# Motivation for recurrent layers

RNN 是循环神经网络的简称。实际上我们之前讲的 MLP 仅仅适用于批量的数据进行训练，而现实生活有很多序列化数据，注意这个序列化和数据存储的序列化是两个不同概念。
我们说的序列化数据指的是随着时间输入进行，例如一篇文章、一段音乐，以及用户访问网站数据也是一条一条传入的，这种时候循环神经网络就能进行处理。

首先我们思考为什么不适用 MLP 而要用 RNN。如图的数据 window 长 100，每个词进行 embed 后长度为 100，隐藏层有 100，那么要多少个参数？大概是 (100 * 100 + 1) * 100 个，太多了。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn//1-mlp.png)


MLP 每次只接受一个输入，但是同时传入上一次计算完成输出，如图。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/2-rnn.png)

如果使用 MLP，同样的模型只需要 (100 * 2 + 1) * 100 个。

这里只是一个简单的介绍，可能并没有领会，接下来通过一个简单的实例详细描述一下。

需要有一定记忆的东西比较多，首先想一下最简单的算术加减法，算术加减法需要我们需要每一位在做运算的时候考虑上一位的进位和退位，现在我们就来简单使用神经网络实现一个减法计算器。

加入两个数相减 651 - 324 ，

那首先传入的第一个数据就是 1 和 4，输出是7，传给下一位需要退位的是 -1， 
然后输入 5 和 2，加上一个时间点传入的退位数 -1，输出为2，以及传给下一位退位数 0，
然后传入 6 和 3，加上一个时间点传入的退位数 0，输出是 3。
我们使用 python 实现一个简单的减法计算器：http://www.k6k4.com/blog/show/aaahiiagt1547625649229


# Simple RNN and Backpropagation

我们大概知道了RNN的思想，那RNN需要训练参数怎么确定呢？

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/3-rnn.png)

其实上面的减法计算器例子中，我们已经看到了，每次前向传播的时候:
`layer_1 = sigmoid(np.dot(X, weight_0) + np.dot(layer_1_values[-1], weight_h))` 除了要加上 输入，还有上一次计算的 weight_h ，也要乘以一个系数加进来。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/4-bptt.png)

反向传播的时候需要考虑时间，其中计算 W 是最复杂的，因为W依赖了上一个时间的 W，还依赖了X，
`layer_1_delta = (future_layer_1_delta.dot(weight_h.T) + layer_2_delta.dot(weight_1.T)) * sigmoid_output_to_derivative(layer_1)` ，
计算误差的时候，每次我们都会累加前面的计算结果。也就是说不仅要沿着网络向后传播，还要沿着时间向后传播，
但是要记住的是，BPTT（backPropagation through time） 算法在随时间传递的时候，权值是共享的。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/5-rnn.png)


而 V 也需要沿着时间向后传播，图中的 U 则不需要。

