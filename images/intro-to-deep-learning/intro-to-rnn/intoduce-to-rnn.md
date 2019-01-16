

# Motivation for recurrent layers

RNN 是循环神经网络的简称。实际上我们之前讲的 MLP 仅仅适用于批量的数据进行训练，而现实生活有很多序列化数据，注意这个序列化和数据存储的序列化是两个不同概念。
我们说的序列化数据指的是随着时间输入进行，例如一篇文章、一段音乐，以及用户访问网站数据也是一条一条传入的，这种时候循环神经网络就能进行处理。

首先我们思考为什么不适用 MLP 而要用 RNN。如图的数据 window 长 100，每个词进行 embed 后长度为 100，隐藏层有 100，那么要多少个参数？大概是 (100 * 100 + 1) * 100 个，太多了。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn//1-mlp.png)


MLP 每次只接受一个输入，但是同时传入上一次计算完成输出，如图。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/2-rnn.png)

如果使用 MLP，同样的模型只需要 (100 * 2 + 1) * 100 个。

这里我们

# Simple RNN and Backpropagation

我们大概知道了RNN的思想，那RNN需要训练参数怎么确定呢？

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/3-rnn.png)

