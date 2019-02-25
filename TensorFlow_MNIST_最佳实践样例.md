

```python
import tensorflow as tf

# 定义神经网络结构的相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 通过tf.get_variable 函数来获取变量。

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))
    
    if regularizer != None:
        tf.add_to_collections('losses',regularizer(weights))
    return weights
```


```python
# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer = tf.constant_init(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    # 声明第二层神经网络的变量并完成向前传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer = tf.constant_init(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
```


```python
import os
from tensorflow.examples.tutorials.mnist import input_data

#配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVAERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./modelbest/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入输出的placeholder
    x = tf.placeholder(tf.float32,[None, INPUT_NODE], name = "x-input")
    y = tf.placeholder(tf.float32,[None, OUTPUT_NODE], name = "y-input")
    regulaizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用inference中定义的前向传播过程。
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable = False)
    
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DEACY, global_step)
    variables_averages_op = variable_averagew.apply(tf.trainables_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learing_rate = tf.train.exponential_decay(
                    LEARNING_RATE_BASE,
                    global_step,
                    mnist.train.num_examples / BATCH_SIZE,
                    LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step = global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        # 在训练过程中不在测试模型在验证数据集的表现，验证和测试过程将会有一个独立的程序来完成。
        for i in range(TRAINING_SETPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                          feed_dict = {x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on  training "
                     "batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
                
                
```


```python
def main(argv = None):
    mnist = input_data.read_data_sets("./data", one_hot = True)
    train(mnist)
tf.app.run()
```

    Extracting ./data\train-images-idx3-ubyte.gz
    Extracting ./data\train-labels-idx1-ubyte.gz
    Extracting ./data\t10k-images-idx3-ubyte.gz
    Extracting ./data\t10k-labels-idx1-ubyte.gz
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-4ec4163ca5b3> in <module>()
          2     mnist = input_data.read_data_sets("./data", one_hot = True)
          3     train(mnist)
    ----> 4 tf.app.run()
    

    C:\Users\14464\AppData\Local\Continuum\Anaconda3\lib\site-packages\tensorflow\python\platform\app.py in run(main, argv)
         46   # Call the main function, passing through any arguments
         47   # to the final program.
    ---> 48   _sys.exit(main(_sys.argv[:1] + flags_passthrough))
         49 
         50 
    

    <ipython-input-11-4ec4163ca5b3> in main(argv)
          1 def main(argv = None):
          2     mnist = input_data.read_data_sets("./data", one_hot = True)
    ----> 3     train(mnist)
          4 tf.app.run()
    

    <ipython-input-10-1f21d13e1337> in train(mnist)
         20     regulaizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
         21     # 直接使用inference中定义的前向传播过程。
    ---> 22     y = inference(x, regularizer)
         23     global_step = tf.Variable(0, trainable = False)
         24 
    

    NameError: name 'regularizer' is not defined



```python

```
