

# Generating names with RNNs

我们已经学了 RNN，现在上手练习一下 RNN。

## 下载数据简单处理

名字数据来自  https://raw.githubusercontent.com/hse-aml/intro-to-dl/master/week5/names


```python
start_token = " "

with open("names") as f:
    names = f.read()[:-1].split("\n")
    names = [start_token + name for name in names]

print('n samples:', len(names))

for x in names[::1000]:
    print(x)

MAX_LENGTH = max(map(len, names))
print("max length:", MAX_LENGTH)
```

简单展示：

```python
plt.title("sequence length distribution")
plt.hist(list(map(len, names)), bins=25)
plt.show()
```

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/6-names.png)


把名字分成字母为单位，并进行映射：

```python

from itertools import chain

tokens = set(chain(*map(lambda x: list(x), names)))
tokens = list(tokens)

n_tokens = len(tokens)

print('n_tokens = ', n_tokens)

assert 50 < n_tokens < 60

# 得到 token to id
token_to_id = {token: i for i, token in enumerate(tokens)}

assert len(tokens) == len(token_to_id), "dictionaries must have same size"

for i in range(n_tokens):
    assert token_to_id[tokens[i]] == i, "token identifier must be it's position in tokens list"

print("Seems alright!")
```

定义方法，将名字转化为编码矩阵：

```python
def to_matrix(names, max_len=None, pad=0, dtype='int32'):
    """Casts a list of names into rnn-digestable matrix"""

    max_len = max_len or max(map(len, names))
    names_ix = np.zeros([len(names), max_len], dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get, names[i]))
        names_ix[i, :len(name_ix)] = name_ix

    return names_ix.T

# Example: cast 4 random names to matrices, pad with zeros
print('\n'.join(names[::2000]))
print(to_matrix(names[::2000]).T)
```

## 定义模型

首先基于 keras 定义网络层
```
s = keras_utils.reset_tf_session()

# 中间层层数
rnn_num_units = 64

# 输入数据编码后的维度
embedding_size = 16

# 为 RNN 创建层
# 注意 TensorFlow 是懒加载执行的，这里定义了不会执行。
embed_x = Embedding(n_tokens, embedding_size)  # an embedding layer that converts character ids into embeddings

# 定义一个隐藏层 [x_t,h_t]->h_t+1
get_h_next = Dense(rnn_num_units, activation="relu")

# softmax 隐藏层 [h_t+1]->P(x_t+1|h_t+1)
get_probas = Dense(n_tokens, activation="softmax")

```

定义 forward 的方法：

```python
def rnn_one_step(x_t, h_t):
    """
    Recurrent neural network step that produces next state and output
    given prev input and previous state.
    We'll call this method repeatedly to produce the whole sequence.

    Follow inline isntructions to complete the function.
    """
    # 使用 embedding ，将矩阵编码
    x_t_emb = embed_x(tf.reshape(x_t, [-1, 1]))[:, 0]

    # 将前一次计算的 h 和输入的 x 连起来
    x_and_h = tf.concat([x_t_emb, h_t], axis=1)

    # 计算下一个 h
    h_next = get_h_next(x_and_h)

    # 计算输出
    output_probas = get_probas(h_next)

    return output_probas, h_next

```

使用 TensorFlow 定义输入：

```python
input_sequence = tf.placeholder('int32', (MAX_LENGTH, None))
batch_size = tf.shape(input_sequence)[1]

predicted_probas = []
h_prev = tf.zeros([batch_size, rnn_num_units])  # initial hidden state
```

循环一个名字输入，构建一个模型：

```python
for t in range(MAX_LENGTH):
    x_t = input_sequence[t]
    probas_next, h_next = rnn_one_step(x_t, h_prev)

    h_prev = h_next
    predicted_probas.append(probas_next)
```

定义输出和优化目标：

```python
predicted_probas = tf.stack(predicted_probas)

predictions_matrix = tf.reshape(predicted_probas[:-1], [-1, len(tokens)])
answers_matrix = tf.one_hot(tf.reshape(input_sequence[1:], [-1]), n_tokens)

# loss = <define loss as categorical crossentropy. Mind that predictions are probabilities and NOT logits!>
loss = tf.losses.softmax_cross_entropy(answers_matrix, predictions_matrix)

optimize = tf.train.AdamOptimizer(1e-4).minimize(loss)
```


## 训练模型

```python
from IPython.display import clear_output
from random import sample

s.run(tf.global_variables_initializer())

batch_size = 32
history = []

for i in range(1000):
    batch = to_matrix(sample(names, batch_size), max_len=MAX_LENGTH)
    loss_i, _ = s.run([loss, optimize], {input_sequence: batch})

    history.append(loss_i)

    if (i + 1) % 100 == 0:
        clear_output(True)
        plt.plot(history, label='loss')
        plt.legend()
        plt.show()

assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge"

x_t = tf.placeholder('int32', (None,))
h_t = tf.Variable(np.zeros([1, rnn_num_units], 'float32'))

next_probs, next_h = rnn_one_step(x_t, h_t)

```

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/7-loss.png)


## 生成名字

```python

def generate_sample(seed_phrase=' ', max_length=MAX_LENGTH):
    """
    The function generates text given a phrase of length at least SEQ_LENGTH.

    parameters:
        The phrase is set using the variable seed_phrase
        The optional input "N" is used to set the number of characters of text to predict.
    """
    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t, h_t.initial_value))

    # feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        s.run(tf.assign(h_t, next_h), {x_t: [ix]})

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        x_probs, _ = s.run([next_probs, tf.assign(h_t, next_h)], {x_t: [x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens, p=x_probs[0]))

    return ''.join([tokens[ix] for ix in x_sequence])


for _ in range(10):
    print(generate_sample())

for _ in range(50):
    print(generate_sample(' Trump'))
```

![avatar](https://raw.githubusercontent.com/dengziming/blogs-images/master/images/intro-to-deep-learning/intro-to-rnn/8-new-names.png)


