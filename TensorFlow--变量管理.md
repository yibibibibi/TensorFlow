
# TensorFlow提供了一种通过变量名称来创建或者获取一种变量的机制。
通过变量直接创建和获取变量tf.get_variable 和 tf.variable_scope 函数实现。


```python
import tensorflow as tf
```

### 1.利用tf.get_variable创建和获取变量。使用tf.get_variable来创建一个变量：



```python
# 下面这两个定义是等价的。
v = tf.get_variable("v1", shape = [1], initializer = tf.constant_initializer(1.0))
#v= tf.Variable(tf.constant(1.0, shape = [1]), name = 'v')
```

tensorflow中initializer函数。

|初始化函数|功能|主要参数|
|:--:|:----:|:--:|
|tf.constant_initializer|将变量初始化为给定常量|常量的取值|
|tf.random_normal_initializer|将变量初始化为正态分布的随机值|正态分布的均值和方差|
|tf.truncated_normal_initializer|将变量初始化为正态分布的随机值<br>，但是如果这个值偏离两个标准差，这个值将会被重新随机赋值|同上|
|tf.random_uniform_initializer|将变量初始化为平均分布的随机值|最大，最小值|
|tf.uniform_unit_scaling_initializer|将变量初始化为平均分布但不影响数量级的随机值|factor(产生随机值时乘以的系数)|
|tf.zeros_initializer|将变量设置为全0|变量维度|
|tf.ones_initializer|将变量设置为全1|变量维度|

### 2.通过tf.variable_scope函数来生成一个上下文管理器。


tf.variable_scope函数可以控制tf.get_variable函数的语义。对于参数reuse,有True和False两个参数可选。
- 当reuse为True时，tf.get_variable函数会获取已经创建好的变量。如果变量不存在就会报错。
- 当reuse为False时或者None,tf.get_variable函数会创建新的变量。如果变量存在就会报错。


```python
'''
# 在名字为foo的命名空间创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer = tf.constant_initializer(1.0))
    
# 因为在变量命名空间已经存在名字为“v”的变量，所以以下代码会报错。
# Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope?
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v",[1])
    
# 生成上下文管理器，将参数reuse设置为True.这样tf.get_variables函数会直接获取已经声明的变量
with tf.variable_scope("foo", reuse = True):
    v2 = tf.get_variable("v",[1])
'''
#reuse设置为True时，tf.variable_scope只能获取已经创建过得变量。下面bar还没创建变量v,所以代码会报错。
# Variable bar/v does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?a
# with tf.variable_scope("bar", reuse = True):
#     v = tf.get_variable("v",[1])
```




    '\n# 在名字为foo的命名空间创建名字为v的变量\nwith tf.variable_scope("foo"):\n    v = tf.get_variable("v", [1], initializer = tf.constant_initializer(1.0))\n    \n# 因为在变量命名空间已经存在名字为“v”的变量，所以以下代码会报错。\n# Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope?\n# with tf.variable_scope("foo"):\n#     v = tf.get_variable("v",[1])\n    \n# 生成上下文管理器，将参数reuse设置为True.这样tf.get_variables函数会直接获取已经声明的变量\nwith tf.variable_scope("foo", reuse = True):\n    v2 = tf.get_variable("v",[1])\n'




```python
# tf.variable_scope函数是可以嵌套的。
with tf.variable_scope("root"):
    # 可以通过tf.get_variable_scope().reuse 函数来获取当前上下文管理器中reuse参数。
    print(tf.get_variable_scope().reuse)   # 输出False
    with tf.variable_scope("foo", reuse = True):   # 新建一个嵌套的上下文管理器，并指定reuse为True
        print(tf.get_variable_scope().reuse)       
        with tf.variable_scope("bar"):             # 新建一个嵌套的上下文管理器，不指定reuse.这时reuse的取值会和外边的一层保持一致
            print(tf.get_variable_scope().reuse)
    print(tf.get_variable_scope().reuse)
```

    False
    True
    True
    False
    


```python
v1 = tf.get_variable("v", [1])
print(v1.name)  # 输出v:0。‘v’为变量的名称；‘:0’表示这个生成变量这个运算的运算结果
```

    v:0
    


```python
with tf.variable_scope("foo"):
    v2 = tf.get_variable("v",[1])
    print(v2.name)  # foo/v:0。在tf.variable_scope中创建的变量。名称前边会加上命名空间。通过“/”分割。
```

    foo/v:0
    


```python
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name) # 输出foo/bar/v:0。 命名空间可以嵌套，变量名称也必须加上所有的命名空间
    
    v4 = tf.get_variable("v1",[1])
    print(v4.name)  # foo/v1:0。当命名空间退出后，变量名称就不会再加入其前缀。
```

    foo/bar/v:0
    foo/v1:0
    


```python
# 创建一个名称为空的命名空间，并设置为reuse= True.
with tf.variable_scope("", reuse = True):
    v5 = tf.get_variable("foo/bar/v", [1]) # 可以用带命名空间的变量名来获取其命名空间下的变量。
    print(v3 == v5)
    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)
```

    True
    True
    


```python
#通过tf.variable_scope和tf.get_variable函数对计算前向传播函数进行了改进。
def inference(input_tensor, reuse = False):
    # 定义第一层神经网络变量和前向传播过程
    with tf.variable_scope('layer1', reuse = reuse):
        # 根据传进来reuse来判断是创建新变量还是使用已经创建好的变量。
        # 在第一次创建会使用新的变量，以后每次调用这个函数使用reuse = True。
        weights = tf.get_variable("weights",[INPUT_NODE,LAYER1_NODE],
                                 initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable("biases",[LAYER1_NODE],
                                initializer = tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        
    # 类似定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope('layer2', reuse = reuse):
        weights = tf.get_variable("weights",[LAYER1_NODE,OUTPUT_NODE],
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1))
        biases = tf.get_variable("biases",[OUTPUT_NODE],
                                  initializer = tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases
    return layer2
                                             
                                            
# x = tf.placeholder(tf.float32, [None,INPUT_NODE], name = 'x-input')
# y = inference(x)
                                             
                                    
        
```


```python

```
