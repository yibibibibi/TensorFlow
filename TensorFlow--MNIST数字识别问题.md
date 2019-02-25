

```python
import tensorflow as tf
#导入MNIST的训练和测试数据
from tensorflow.examples.tutorials.mnist import input_data

```

### tensorflow中通过input_data.read_data_sets函数生成一个类会将MNIST数据划分为train、validation和test三个数据集。
在本例中input_data.read_data_sets可以提取四个数据集，如下表：

|数据集名称|内容|
|:----:|:--:|
|train-images-idx3-ubyte.gz|训练数据图片|
|train-labels-idx1-ubyte.gz|训练数据答案|
|t10k-images-idx3-ubyte.gz|测试数据图片|
|t10k-labels-idx1-ubyte.gz|测试书记答案|


```python
# 载入MNIST数据集，如果指定的地址/data下没有下载好的数据，TensorFlow会从网路上下载指定数据集。
mnist = input_data.read_data_sets("data/", one_hot = True)

#打印 training data size : 55000
print("Training data size: ", mnist.train.num_examples )
#打印 training data size: 5000
print("Validating data size:", mnist.train.num_examples )
# 打印 Testing data size：10000 
print("Testing data size:",mnist.train.num_examples ) 
# 打印Example training data :
# 处理后的每一张数据集是一个长度为784的一维数组。(28*28 = 784).每一个值代表像素值，在【0,1】之间。0表示白色背景。1表示黑色背景
print("Example training data :", len(mnist.train.images[0]))

#打印Example training data label:
print('Example training data label:', mnist.train.labels[0])
```

    Extracting data/train-images-idx3-ubyte.gz
    Extracting data/train-labels-idx1-ubyte.gz
    Extracting data/t10k-images-idx3-ubyte.gz
    Extracting data/t10k-labels-idx1-ubyte.gz
    Training data size:  55000
    Validating data size: 55000
    Testing data size: 55000
    Example training data : 784
    Example training data label: [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
    

### mnist还提供了一个mnist.train.next_batch函数，可以读取训练数据的一小部分作为一个训练的batch。


```python
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
#从train的集合选择batch_size个训练数据
print("X shape",(xs.shape))
print("Y shape",(ys.shape))
```

    X shape (100, 784)
    Y shape (100, 10)
    

# 利用TensorFlow程序解决MNIST手写体数字识别的问题。


```python
# MNIST数据集相关的常数
INPUT_NODE = 784  # 输入层的节点数：对于MNIST数据集，这个就等于图片的像素点。
OUTPUT_NODE = 10 # 输出层的节点数：这个等于类别的数目。因为在MNIST数据集中，区分0~9这十个数。所以输出层的节点数为10

#配置神经网络的参数
LAYER1_NODE = 500 #隐藏层的节点个数。本文只设置一个隐藏层
BATCH_SIZE = 100 # 一个训练的数据集个数。数字越小时，训练过程越接近随机梯度；数字越大时，训练越接近梯度下降。

LEARNING_RATE_BASE = 0.8 #基础学习率
LEARING_RATE_DECAY = 0.99 # 学习的衰减率
REGULARIZATION_RATE = 0.0001 #正则化在损失函数的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率
```


```python
# 给定一个辅助函数用来计算前向传播。
# 定义了ReLU激活函数用来实现去线性化
# 这个函数支持对变量实现滑动平均
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #如果没有实现滑动平均，直接计算当前参数的值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + \
                           avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
```


```python
# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE], name = 'y-input')
    
    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
    
    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE], stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
    
    # 计算当前参数的神经网络的前向传播算法。这里不计算参数的滑动平均，avg_class = None.
    y =  inference(x, None, weights1, biases1, weights2, biases2)
    
    # 定义存储训练轮数的变量。这个变量不计算滑动平均，这个变量为不可训练的变量(trainable  = FALSE).
    global_step = tf.Variable(0, trainable = False)
    
    #给定的滑动平均衰减率和训练轮数的变量， 初始化滑动平均类，
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在代表神经网络参数的变量上使用滑动平均。不包括训练的轮数。
    #tf.trainable_variables返回的就是图上集合：GraphKeys.TRAINABLE_VARIABLES 中的元素。这个集合就所有可以被迭代的元素。
    variables_averages_op = variable_average.apply(tf.trainable_variables())
    
    #计算了使用滑动平均的前向传播算法
    average_y = inference(\
                         x, variable_average, weights1, biases1, weights2, biases2)
    # 计算交叉熵刻画预测值和真实值的差距。
    # 使用sparse_softmax_cross_entropy_with_logits函数来计算交叉熵.当函数只有一个正确答案。这个函数适用。本例对于每一个图片只有一个正确答案。
    # 这个函数包括两个参数。第一个参数是不包含softmax层的前向传播结果。
    #                        还有一个参数是正确分类的结果。
    # tf.argmax()函数可以得到正确答案对应的编号。tf.argmax() 用来后去某一个坐标系的最大值。
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits = y, labels = tf.argmax(y_ , 1))
    
    # 计算当前batch中的所有样例的交叉熵平均值。
    
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络的权重项，不计算偏置项。
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失加上正则损失的和
    loss = cross_entropy_mean + regularization
    
    # 设置指数衰减学习率
    
    learning_rate  = tf.train.exponential_decay(LEARNING_RATE_BASE,   #基础的学习率，随着迭代的进行，
                                                                       #更新时使用的学习率在这个基础上递减
                                               global_step,          # 当前迭代的轮数
                                               mnist.train.num_examples / BATCH_SIZE,  #过完所有的训练数据需要迭代的次数 
                                               LEARING_RATE_DECAY)         #学习率的衰减速度
    # 使用tf.train.GradientDescentOptimizer 优化算法来优化损失函数。这里损失函数包含了交叉熵损失和L2正则化损失。 
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(loss, global_step  = global_step)

    # 在训练神经网络，没过一遍数据需要通过反向传播来更新神经网络的参数，又要更新参数的滑动平均值。
    # 为了一次完成多个操作。Tensorflow提供了：tf.control_dependencies 和 tf.group 两种机制。
    # 下面两行程序是等价的。
    # train_op = tf.group(train_step, variables_average_op
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name = 'train')

    # 检验使用了滑动平均模型的前向传播结果是否正确。
    # 判断两个张量的每一维数据是否相等。相等就返回True,否则返回False.
    corrent_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 将一个布尔型的数值转换为数值型，计算平均值，来表示正确率。
    accuracy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

    '''
    初始化会话并开始训练过程
    '''
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据。
        ## 交叉验证数据集
        validate_feed = {x: mnist.validation.images, y_:mnist.validation.labels}
        ## 准备测试数据
        test_feed = {x: mnist.test.images, y_:mnist.test.labels}

        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict = validate_feed)
                print("After %d training step(s), vaildation accuracy "\
                     "using average model is %g " % (i, validate_acc))
            # 产生这一轮使用的一个batch的训练数据集，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict = {x: xs, y_ : ys})

        #在训练结束之后，在测试数据集上检测神经网络的最终正确率
        test_acc = sess.run(accuracy, feed_dict = test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))

```


```python
# 主程序入口。
def main(argv=None):
    # 声明处理MNIST数据集的类，
    mnist = input_data.read_data_sets("data/",one_hot = True)
    train(mnist)
```


```python
if __name__ == '__main__':
    tf.app.run()
```

    Extracting data/train-images-idx3-ubyte.gz
    Extracting data/train-labels-idx1-ubyte.gz
    Extracting data/t10k-images-idx3-ubyte.gz
    Extracting data/t10k-labels-idx1-ubyte.gz
    After 0 training step(s), vaildation accuracy using average model is 0.1396 
    After 1000 training step(s), vaildation accuracy using average model is 0.978 
    After 2000 training step(s), vaildation accuracy using average model is 0.982 
    After 3000 training step(s), vaildation accuracy using average model is 0.9834 
    After 4000 training step(s), vaildation accuracy using average model is 0.9836 
    After 5000 training step(s), vaildation accuracy using average model is 0.984 
    After 6000 training step(s), vaildation accuracy using average model is 0.9852 
    After 7000 training step(s), vaildation accuracy using average model is 0.9844 
    After 8000 training step(s), vaildation accuracy using average model is 0.9858 
    After 9000 training step(s), vaildation accuracy using average model is 0.9848 
    After 10000 training step(s), vaildation accuracy using average model is 0.9858 
    After 11000 training step(s), vaildation accuracy using average model is 0.9848 
    After 12000 training step(s), vaildation accuracy using average model is 0.9844 
    After 13000 training step(s), vaildation accuracy using average model is 0.9864 
    After 14000 training step(s), vaildation accuracy using average model is 0.985 
    After 15000 training step(s), vaildation accuracy using average model is 0.9854 
    After 16000 training step(s), vaildation accuracy using average model is 0.9854 
    After 17000 training step(s), vaildation accuracy using average model is 0.986 
    After 18000 training step(s), vaildation accuracy using average model is 0.9858 
    After 19000 training step(s), vaildation accuracy using average model is 0.9856 
    After 20000 training step(s), vaildation accuracy using average model is 0.9856 
    After 21000 training step(s), vaildation accuracy using average model is 0.9862 
    After 22000 training step(s), vaildation accuracy using average model is 0.9862 
    After 23000 training step(s), vaildation accuracy using average model is 0.9854 
    After 24000 training step(s), vaildation accuracy using average model is 0.9862 
    After 25000 training step(s), vaildation accuracy using average model is 0.9864 
    After 26000 training step(s), vaildation accuracy using average model is 0.9862 
    After 27000 training step(s), vaildation accuracy using average model is 0.986 
    After 28000 training step(s), vaildation accuracy using average model is 0.9864 
    After 29000 training step(s), vaildation accuracy using average model is 0.9856 
    After 30000 training step(s), test accuracy using average model is 0.9837
    


    An exception has occurred, use %tb to see the full traceback.
    

    SystemExit
    


    C:\Users\14464\AppData\Local\Continuum\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2889: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)
    


```python

```
