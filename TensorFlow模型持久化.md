
### 为了让训练好模型可以复用，需要将神经网络结果持久化。TensorFlow提供了一个简单的API来保存和恢复神经网络模型。这个API是tf.train.Saver.


```python
# 保存TensorFlow计算图的方法

import  tensorflow as tf

#声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = 'v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = 'v2')
result = v1 + v2

init_op = tf.global_variables_initializer()

#声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    #将模型保存在/path/to/model/model.ckpt文件中。
    saver.save(sess,"./mymodel/model.ckpt")
```


```python
# 加载这个已经保存的Tensorflow模型的方法

# 使用和保存模型代码一样的方式来声明变量
import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = 'v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = 'v2')
result = v1 + v2


saver = tf.train.Saver()

with tf.Session() as sess:
    #加载已经保存的模型，并通过已经保存的模型的变量的值来计算加法
    saver.restore(sess,"./mymodel/model.ckpt")
    print(sess.run(result))
```

    INFO:tensorflow:Restoring parameters from ./mymodel/model.ckpt
    [ 3.]
    


```python
# 直接加载已经持久化的图
# 加载持久化的图。
# model.ckpt.mate保存了TensorFlow的计算图结构
# checkpoint 报存了一个目录下所有模型的文件列表。
savers = tf.train.import_meta_graph("./mymodel/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "./mymodel/model.ckpt")
    # 通过张量名称获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
```

    INFO:tensorflow:Restoring parameters from ./mymodel/model.ckpt
    [ 3.]
    


```python
# 对于变量重命名的使用。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")

#直接使用tf.train.Saver()来加载模型，会报出变量找不到的错误。下面显示了报错信息
#tf.train.Saver()

# 使用一个字典来重命名变量就可以加载到原来的模型中。
# 
saver = tf.train.Saver({'v1':v1,'v2':v2})
```


```python
import tensorflow as tf
# 通过变量名重命名直接读取滑动平均值
v = tf.Variable(0, dtype = tf.float32, name = 'v')
# 在没有声明滑动平均模型只有一个变量v，所以以下语句只会输出“v:0”
for variables in tf.global_variables():
    print(variables.name)
```

    v:0
    


```python
#计算滑动平均值
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在申明滑动平均后，Tensorflow会自动生成一个影子变量
for variables in tf.global_variables():
    print(variables.name)
```

    v:0
    v/ExponentialMovingAverage:0
    


```python
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    # 保存时，TensorFlow会把v:0和v/ExponentialMovingAverage:0两个变量都保存下来
    saver.save(sess, "./emamodel/model.ckpt")
    print(sess.run([v,ema.average(v)]))
```

    [10.0, 0.099999905]
    


```python
# 通过变量重命名将原来的滑动平均值直接赋值给v.
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,'./emamodel/model.ckpt')
    print(sess.run(v))
```

    INFO:tensorflow:Restoring parameters from ./emamodel/model.ckpt
    0.0999999
    


```python
# tf.train.ExponentialMovingAverage  类提供了variables_to_restore 函数用来生成tf.train.Saver类所需要的变量重命名字典。
import tensorflow as tf
v = tf.Variable(0, dtype = tf.float32, name = 'v')
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用variables_to_restore 函数可以直接生成上面代码的字典。
# {"v/ExponentialMovingAverage":v}
print(ema.variables_to_restore())
# 输出 {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,'./emamodel/model.ckpt')
    print(sess.run(v))
```

    {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
    INFO:tensorflow:Restoring parameters from ./emamodel/model.ckpt
    0.0999999
    


```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape = [1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape = [1]), name = "v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算。
    graph_def = tf.get_default_graph().as_graph_def()
    
    # 将图中的变量及取值转换为常量，同时将图中不必要的节点删掉。
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    
    # 将导出的文件存入模型中。
    with tf.gfile.GFile("combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
```

    INFO:tensorflow:Froze 2 variables.
    Converted 2 variables to const ops.
    


```python
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename = "combined_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))
```

    [array([ 3.], dtype=float32)]
    


```python

```
