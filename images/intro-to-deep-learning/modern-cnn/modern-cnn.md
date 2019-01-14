
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

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/2-init.png)

如果都初始化为 0，最终训练出来的结果是一样的。这是 symmetry problem，为了解决这个问题，我们通常需要从标准正太分布取值，然后乘以一个系数例如 0.03.

neuron 的输出是输入的线性组合加上一个激活函数，neuron 的输出会被下一层的 neuron 使用，所以我们希望输出的也是归一化的数据。所以我们希望我们的 weight 的 mean 是 0.

当你堆积很多层的时候，均值还是 0，但是 variance（方差）会发生变化。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/3-var.png)

如图为方差的计算公式，最终我们希望方差不会逐渐堆积，所以希望最后的括号内容是1 。

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/4-var.png)

我们是从标准正太分布得到的w，方差是1，所以只需要乘以一个系数就满足方差是1，也就是 Xavier initialization。

## batch normalization

上面介绍的是初始化的值，在训练过程中我们应该怎么办？

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/modern-cnn/5-nor.png)


首先我们需要归一化 在 activation 之前的 neuron output ，也就是图中的 h。首先减去 mean 保证 zero mean 、除以方差保证 unit variance。
然后乘以 gamma 得到新的 variance，添加 beta 得到新的 mean。

这里面的 sigma 和 mu 哪里来的？基于当前的 batch 进行估计，而且我们可以在每一步 backpropagation 进行这个操作。

这里面的 gamma 和 beta 哪里来的？我也没太懂这句话，貌似是说这个 gamma 和 beta 都是需要通过训练得到的。

## dropout

另一种技术叫做 dropout，这个详情可以查看资料，实际上就是删掉部分节点。

## argument

如果我们的数据太少了，我们怎么办？最好的办法是我们对数据进行 flip、rotation 等操作。
