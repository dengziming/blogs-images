

# python RNN

使用 python 实现一个简单的 RNN 减法计算器。减法计算的时候需要每次考虑当前的 被减数 、减数，还要考虑上一次的可能要的退位输入，这刚好就是一个时序网络模型。


## 定义一些工具类及激活函数

```python
import copy, numpy as np

np.random.seed(0)  # 随机数生成器的种子，可以每次得到一样的值


# 使用 sigmod 激活函数
def sigmoid(x):  # 激活函数
    output = 1 / (1 + np.exp(-x))
    return output


# 激活函数的导数
def sigmoid_output_to_derivative(output):  # 激活函数的导数
    return output * (1 - output)

```

为了简单，我们直接使用二进制进行计算，而且是 256 以内的，定义二进制互相转换的代码

```python

int2binary = {}  # 整数到其二进制表示的映射
binary_dim = 8  # 暂时制作256以内的减法

# 计算0-256的二进制表示
largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

```

## 定义模型的输入输出


首先我们的模型输入是减数和被减数的各自一位数，所以输入维度为 2，而每次计算的输出结果是两个一位相减后的结果，所以输出维度是1.

另外我们任务简单，只需要一个隐藏层就行，所以一个隐藏层即可，我们将层数设置为 16.

```python
# input variables
alpha = 0.9  # 学习速率
input_dim = 2  # 输入的维度是2
hidden_dim = 16
output_dim = 1  # 输出维度为1
```

我们使用输入是 2位，隐藏层输出是 16位，所以第一层的参数就是 2 * 16 的矩阵。
隐藏层输输出 16 为，最终输出一位，所以隐藏层参数是 16 * 1 位。
再加上我们每次都要将上一次时间点的输出传给下一个时间段，所以需要一个系数矩阵 16 * 16 位。

```python
# initialize neural network weights
weight_0 = (2 * np.random.random((input_dim, hidden_dim)) - 1) * 0.05  # 维度为2*16， 2是输入维度，16是隐藏层维度
weight_1 = (2 * np.random.random((hidden_dim, output_dim)) - 1) * 0.05
weight_h = (2 * np.random.random((hidden_dim, hidden_dim)) - 1) * 0.05

```

上面是正向传播的参数，同理需要反向传播：

```python
# 用于存放反向传播的权重更新值
weight_0_update = np.zeros_like(weight_0)
weight_1_update = np.zeros_like(weight_1)
weight_h_update = np.zeros_like(weight_h)

```

开始准备输入数据，输入数据可以直接随机生成，然后转化为 2进制。

```python

# 生成一个数字a
a_int = np.random.randint(largest_number)
# 生成一个数字b,b的最大值取的是largest_number/2,作为被减数，让它小一点。
b_int = np.random.randint(largest_number / 2)
# 如果生成的b大了，那么交换一下
if a_int < b_int:
    tt = b_int
    b_int = a_int
    a_int = tt

a = int2binary[a_int]  # binary encoding
b = int2binary[b_int]  # binary encoding
# true answer
c_int = a_int - b_int
c = int2binary[c_int]

# 存储神经网络的预测值
d = np.zeros_like(c)

# 存放总误差，每次训练完后总误差清零
overallError = 0  

layer_2_deltas = list()  # 存储每个时间点输出层的误差
layer_1_values = list()  # 存储每个时间点隐藏层的值

layer_1_values.append(np.ones(hidden_dim) * 0.1)  # 一开始没有隐藏层，所以初始化一下原始值为0.1
```

数据有了，我们需要连续传入每一位进行减法操作.

forward：

```python

for position in range(binary_dim):  # 循环遍历每一个二进制位

    # 得到输入，是两个数各取一位。
    X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])  # 从右到左，每次去两个输入数字的一个bit位
    # 得到输出
    y = np.array([[c[binary_dim - position - 1]]]).T  # 正确答案

    # 隐藏层 = 输入和上一时间点的  (Wx * X  ~+ Wh * H)
    layer_1 = sigmoid(np.dot(X, weight_0) + np.dot(layer_1_values[-1],
                                                   weight_h))  # （输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！

    # 输出层 = 隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
    layer_2 = sigmoid(np.dot(layer_1, weight_1))

    layer_2_error = y - layer_2  # 预测误差

    # 每一个时间点的误差导数 dL/dz = (y - p)*Dz/dw
    layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
    overallError += np.abs(layer_2_error[0])  # 总误差

    # 输出
    d[binary_dim - position - 1] = np.round(layer_2[0][0])  # 记录下每一个预测bit位

    # 记录下隐藏层的值，在下一个时间点传入。
    layer_1_values.append(copy.deepcopy(layer_1))

```

然后是向后传播进行调整：

```python
future_layer_1_delta = np.zeros(hidden_dim)

# 反向传播，从最后一个时间点到第一个时间点
for position in range(binary_dim):
    # 最后一次的两个输入
    X = np.array([[a[position], b[position]]])

    # 当前时间点的隐藏层
    layer_1 = layer_1_values[-position - 1]

    # 前一个时间点的隐藏层
    prev_layer_1 = layer_1_values[-position - 2]

    # 当前时间点输出层导数，前向传播时 dL/dz = (y - p)*Dz/dw 计算的。
    layer_2_delta = layer_2_deltas[-position - 1]

    # 等到完成了一次减法操作所有反向传播误差计算，才会更新权重矩阵，先暂时把更新矩阵存起来。
    weight_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)

    # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
    layer_1_delta = (future_layer_1_delta.dot(weight_h.T) +
                     layer_2_delta.dot(weight_1.T)) * sigmoid_output_to_derivative(layer_1)

    # 等到完成了一次减法操作所有反向传播误差计算， 才会更新权重矩阵，先暂时把更新矩阵存起来。
    weight_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
    weight_0_update += X.T.dot(layer_1_delta)

    future_layer_1_delta = layer_1_delta

# 完成所有反向传播之后，更新权重矩阵。并把矩阵变量清零
weight_0 += weight_0_update * alpha
weight_1 += weight_1_update * alpha
weight_h += weight_h_update * alpha
weight_0_update *= 0
weight_1_update *= 0
weight_h_update *= 0
```

每隔一段时间打印一下误差：

```python
# print out progress
if j % 800 == 0:
    # print(synapse_0,synapse_h,synapse_1)
    print("总误差:" + str(overallError))
    print("Pred:" + str(d))
    print("True:" + str(c))
    out = 0
    for index, x in enumerate(reversed(d)):
        out += x * pow(2, index)
    print(str(a_int) + " - " + str(b_int) + " = " + str(out))
    print("------------")
```

## 完整代码：

```python
import copy
import numpy as np

np.random.seed(0)  # 随机数生成器的种子，可以每次得到一样的值


# 使用 sigmod 激活函数
def sigmoid(x):  # 激活函数
    output = 1 / (1 + np.exp(-x))
    return output


# 激活函数的导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


int2binary = {}  # 整数到其二进制表示的映射
binary_dim = 8  # 暂时制作256以内的减法

# 计算0-256的二进制表示
largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.9  # 学习速率
input_dim = 2  # 输入的维度是2
hidden_dim = 16
output_dim = 1  # 输出维度为1

# initialize neural network weights
weight_0 = (2 * np.random.random((input_dim, hidden_dim)) - 1) * 0.05  # 维度为2*16， 2是输入维度，16是隐藏层维度
weight_1 = (2 * np.random.random((hidden_dim, output_dim)) - 1) * 0.05
weight_h = (2 * np.random.random((hidden_dim, hidden_dim)) - 1) * 0.05
# => [-0.05, 0.05)，

# 用于存放反向传播的权重更新值
weight_0_update = np.zeros_like(weight_0)
weight_1_update = np.zeros_like(weight_1)
weight_h_update = np.zeros_like(weight_h)

# training
for j in range(10000):

    # 生成一个数字a
    a_int = np.random.randint(largest_number)
    # 生成一个数字b,b的最大值取的是largest_number/2,作为被减数，让它小一点。
    b_int = np.random.randint(largest_number / 2)
    # 如果生成的b大了，那么交换一下
    if a_int < b_int:
        tt = b_int
        b_int = a_int
        a_int = tt

    a = int2binary[a_int]  # binary encoding
    b = int2binary[b_int]  # binary encoding
    # true answer
    c_int = a_int - b_int
    c = int2binary[c_int]

    # 存储神经网络的预测值
    d = np.zeros_like(c)
    overallError = 0  # 每次把总误差清零

    layer_2_deltas = list()  # 存储每个时间点输出层的误差
    layer_1_values = list()  # 存储每个时间点隐藏层的值

    layer_1_values.append(np.ones(hidden_dim) * 0.1)  # 一开始没有隐藏层，所以初始化一下原始值为0.1

    # moving along the positions in the binary encoding
    for position in range(binary_dim):  # 循环遍历每一个二进制位

        # 得到输入，是两个数各取一位。
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])  # 从右到左，每次去两个输入数字的一个bit位
        # 得到输出
        y = np.array([[c[binary_dim - position - 1]]]).T  # 正确答案

        # 隐藏层 = 输入和上一时间点的  (Wx * X  ~+ Wh * H)
        layer_1 = sigmoid(np.dot(X, weight_0) + np.dot(layer_1_values[-1],
                                                       weight_h))  # （输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！

        # 输出层 = 隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
        layer_2 = sigmoid(np.dot(layer_1, weight_1))

        layer_2_error = y - layer_2  # 预测误差

        # 每一个时间点的误差导数 dL/dz = (y - p)*Dz/dw
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])  # 总误差

        # 输出
        d[binary_dim - position - 1] = np.round(layer_2[0][0])  # 记录下每一个预测bit位

        # 记录下隐藏层的值，在下一个时间点传入。
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    # 反向传播，从最后一个时间点到第一个时间点
    for position in range(binary_dim):
        # 最后一次的两个输入
        X = np.array([[a[position], b[position]]])

        # 当前时间点的隐藏层
        layer_1 = layer_1_values[-position - 1]

        # 前一个时间点的隐藏层
        prev_layer_1 = layer_1_values[-position - 2]

        # 当前时间点输出层导数，前向传播时 dL/dz = (y - p)*Dz/dw 计算的。
        layer_2_delta = layer_2_deltas[-position - 1]

        # 等到完成了一次减法操作所有反向传播误差计算，才会更新权重矩阵，先暂时把更新矩阵存起来。
        weight_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)

        # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
        layer_1_delta = (future_layer_1_delta.dot(weight_h.T) +
                         layer_2_delta.dot(weight_1.T)) * sigmoid_output_to_derivative(layer_1)

        # 等到完成了一次减法操作所有反向传播误差计算， 才会更新权重矩阵，先暂时把更新矩阵存起来。
        weight_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        weight_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # 完成所有反向传播之后，更新权重矩阵。并把矩阵变量清零
    weight_0 += weight_0_update * alpha
    weight_1 += weight_1_update * alpha
    weight_h += weight_h_update * alpha
    weight_0_update *= 0
    weight_1_update *= 0
    weight_h_update *= 0

    # print out progress
    if j % 800 == 0:
        # print(synapse_0,synapse_h,synapse_1)
        print("总误差:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " - " + str(b_int) + " = " + str(out))
        print("------------")

print(weight_0)
print(weight_1)
print(weight_h)
```

运行结果如图：

```python

总误差:[3.99972855]
Pred:[0 0 0 0 0 0 0 0]
True:[0 0 1 1 0 0 1 1]
60 - 9 = 0
------------
总误差:[2.486562]
Pred:[0 0 0 0 0 0 0 0]
True:[0 0 0 1 0 0 0 1]
17 - 0 = 0
------------
总误差:[3.51869416]
Pred:[0 0 1 0 0 1 1 0]
True:[0 0 0 1 1 1 1 0]
89 - 59 = 38
------------
总误差:[0.18361106]
Pred:[0 0 0 1 1 0 0 0]
True:[0 0 0 1 1 0 0 0]
43 - 19 = 24
------------
总误差:[0.1709148]
Pred:[0 0 0 0 0 0 1 0]
True:[0 0 0 0 0 0 1 0]
73 - 71 = 2
------------
总误差:[0.13827615]
Pred:[0 0 1 1 1 1 0 0]
True:[0 0 1 1 1 1 0 0]
71 - 11 = 60
------------
总误差:[0.08982648]
Pred:[1 0 0 0 0 0 0 0]
True:[1 0 0 0 0 0 0 0]
230 - 102 = 128
------------
总误差:[0.17024705]
Pred:[0 1 1 1 0 0 0 1]
True:[0 1 1 1 0 0 0 1]
160 - 47 = 113
------------
总误差:[0.06442929]
Pred:[0 1 0 1 1 0 0 1]
True:[0 1 0 1 1 0 0 1]
92 - 3 = 89
------------
总误差:[0.04940924]
Pred:[0 0 0 1 1 0 1 1]
True:[0 0 0 1 1 0 1 1]
44 - 17 = 27
------------
总误差:[0.04009697]
Pred:[1 0 0 1 0 1 1 0]
True:[1 0 0 1 0 1 1 0]
167 - 17 = 150
------------
总误差:[0.06397785]
Pred:[1 0 0 1 1 0 0 0]
True:[1 0 0 1 1 0 0 0]
204 - 52 = 152
------------
总误差:[0.02595276]
Pred:[1 1 0 0 0 0 0 0]
True:[1 1 0 0 0 0 0 0]
209 - 17 = 192
------------
[[-1.98770636 -2.67813968  0.57785093 -8.06587785 -4.95048226 -5.49179799
  -2.16649364 -2.49054416  1.0711969   0.64682771  0.02610149 -0.8220327
  -0.54667881  1.87767507 -7.1230234   0.53598621]
 [-2.24300037  1.9567616  -2.3882899   6.38460053 -3.05617437  3.9908415
   0.76258289  0.55242759 -2.88987182 -2.36964744 -1.25218164 -0.62007534
  -0.8256679   4.5098746  -7.55297376 -2.03275335]]
[[ -0.47950593]
 [ -2.29261578]
 [  3.35784577]
 [  9.54490619]
 [ -8.19617962]
 [  3.4738001 ]
 [ -1.7685241 ]
 [  0.83798787]
 [  3.8244388 ]
 [  3.30050482]
 [  1.54616171]
 [ -0.07881348]
 [  0.51428454]
 [ -6.92353418]
 [-11.25141562]
 [  2.90943311]]
[[-4.26997603e-01 -1.71982552e-01  1.79820707e-01 -1.40155662e+00
  -4.01526804e-01 -7.55061492e-01 -3.01541270e-01 -2.40838522e-01
   2.52059783e-01  1.68904400e-01 -2.10499595e-01 -4.30196573e-01
  -3.90505957e-01 -9.27471842e-01  1.88258907e+00  4.80727051e-02]
 [-5.86487339e-01 -1.28426784e-01  2.05332024e-01  5.52230861e-01
   4.88604459e-01  4.51778157e-01 -3.89515554e-02 -4.63338910e-01
  -2.43026368e-01  7.04570341e-02 -3.38790135e-01 -3.18628823e-01
  -4.18833788e-01 -1.88583536e-01 -8.96821407e-01 -1.31401916e-02]
 [-5.98844018e-01  1.19393875e-01 -9.84298419e-02 -1.29064658e+00
  -3.70975416e-01 -5.61609194e-01 -3.04921917e-01 -2.53541627e-02
  -1.23689292e-01 -1.05975761e-01 -4.38366153e-01 -5.65858998e-01
  -5.97096075e-01 -1.20625502e+00  1.44154093e+00 -1.72179876e-01]
 [-6.81660218e-01 -9.03266942e-02 -5.59149617e-01  3.70066542e+00
   3.20883801e+00  2.37168170e+00  3.68195329e-01 -9.82742829e-01
  -1.44693296e+00 -7.44668959e-01 -6.84900855e-01 -1.02721495e-01
  -4.95289183e-01  1.09196350e+00 -3.05477915e+00 -6.26773959e-01]
 [-5.52135657e-01 -2.26179747e-01  4.47230059e-02  3.80674218e-01
   3.44383955e-01  3.96177427e-01 -2.58614068e-02 -6.32097403e-01
  -4.28513083e-01 -5.37495415e-02 -3.25525732e-01 -1.38779957e-01
  -3.45647380e-01 -2.47192348e-01 -1.17181868e+00 -7.88056423e-02]
 [-6.40626647e-01 -1.03356443e-01 -1.71302499e-01  2.48412967e+00
   2.40918728e+00  1.67241368e+00  2.61433179e-01 -9.56019232e-01
  -9.44185099e-01 -3.26596026e-01 -5.17591069e-01 -1.61922554e-01
  -4.03373997e-01  1.88434542e-01 -2.22584356e+00 -2.70398794e-01]
 [-4.72363848e-01 -8.93082852e-02  2.02692207e-01 -2.24498024e-01
   2.25301284e-01  1.89944411e-02 -1.44767447e-01 -3.72901520e-01
  -1.03727555e-01  1.73789770e-01 -3.31856715e-01 -2.68379721e-01
  -3.56503350e-01 -3.87676442e-01 -6.91435245e-03  5.47998067e-02]
 [-4.89159035e-01 -8.69907184e-02 -9.12141816e-02  1.95148616e-01
   4.70148912e-01  2.53492055e-01 -1.52655839e-01 -4.02154257e-01
  -2.38864898e-01 -1.85064448e-01 -4.04165968e-01 -3.51547622e-01
  -4.44463346e-01  1.08889652e-01 -4.58066969e-01 -2.08236742e-01]
 [-6.00534031e-01  2.11980815e-01 -9.64022314e-02 -1.45386106e+00
  -5.83281316e-01 -5.36410152e-01 -3.19642093e-01  4.54530897e-02
  -1.15316748e-01 -2.08061451e-01 -4.43774080e-01 -6.12102322e-01
  -5.73260831e-01 -1.14998287e+00  1.67260971e+00 -2.13511426e-01]
 [-6.16445387e-01  1.62697886e-01 -3.42289721e-02 -1.41826224e+00
  -4.90801778e-01 -6.09989127e-01 -3.80248970e-01  4.06877796e-04
  -1.44306772e-01 -1.44531944e-01 -4.46059937e-01 -6.45755716e-01
  -6.41750374e-01 -1.32263106e+00  1.63660193e+00 -1.45912425e-01]
 [-4.76784365e-01 -3.81194520e-02  1.66847736e-01 -1.22245490e+00
  -3.51052900e-01 -6.03160660e-01 -3.38025035e-01 -1.70013668e-01
   1.06263255e-01  7.07524427e-02 -2.94197615e-01 -4.51091849e-01
  -4.58373465e-01 -8.29332967e-01  1.53069040e+00  5.12344942e-02]
 [-5.20872653e-01 -6.41523927e-02  1.23479182e-01 -8.95885228e-01
  -2.84384058e-02 -3.68669732e-01 -2.42061881e-01 -3.48136109e-01
   4.46593691e-02  2.25858378e-02 -3.48232379e-01 -4.20597058e-01
  -4.60913094e-01 -7.03355114e-01  1.00962415e+00  1.58624628e-02]
 [-4.35245671e-01 -1.63561138e-01  9.93688765e-02 -1.06745298e+00
  -1.94493660e-01 -5.00590487e-01 -2.58178977e-01 -3.10477688e-01
   5.44932940e-02  6.75814218e-02 -3.25569019e-01 -3.94882927e-01
  -4.29187027e-01 -6.83963306e-01  1.25447167e+00 -3.60131651e-02]
 [-6.43720347e-01 -1.53557394e-01  5.74401818e-02 -8.37013586e-01
   2.63649894e-01 -2.32810808e-01 -1.87237280e-01 -7.65285285e-01
  -4.20197838e-01 -1.24000967e-01 -4.85415853e-01 -4.74395545e-01
  -6.91422420e-01 -1.90569349e+00  1.33851954e+00 -6.66827577e-02]
 [-4.27365909e-01  2.80345914e-02  3.48670505e-01 -2.18549164e+00
  -7.39832630e-01 -1.15377137e+00 -2.12492720e-01 -5.36632427e-02
   5.05145859e-01  4.60069172e-01 -1.52771441e-01 -4.37228020e-01
  -3.28103814e-01 -1.13518582e+00  2.62648029e+00  2.08451496e-01]
 [-5.42588482e-01  1.06514317e-01 -1.42753044e-02 -1.29660751e+00
  -3.19899846e-01 -5.68687247e-01 -2.79427362e-01 -5.44375309e-02
  -8.99399644e-02 -3.87196962e-02 -3.25795897e-01 -4.90593891e-01
  -4.76457746e-01 -1.08466426e+00  1.43109026e+00 -1.31051785e-01]]

```

可以看出误差越来越小，最后基本上能够计算正确。
