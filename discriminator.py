import tensorflow as tf
import numpy as np
import yaml

with open("SeqGAN.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
#tf.nn.rnn_cell._linear的替代方法，因为在tensorflow1.0.1已经被移除
#highway层是从其他地方借过来的。
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k] 是用一个自学习？的矩阵去乘输入，得到的结果是batch x output size
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors. 输入的一个2维度的张量，是batch x n，n是什么呢，一维的一个值嘛？
    output_size: int, second dimension of W[i]. output size是一个整数，是W[i]矩阵的第二维度的值，
                                          因为 input[i] x W[i], batch * n X n * m, 这个m就是output szie 
    scope: VariableScope for the created subgraph; defaults to "Linear". 这个不太清楚。
  Returns:            返回值
    A 2D Tensor with shape [batch x output_size] equal to 就是相乘+b 得到的东西 batch x output_size
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices. W【i】是最近得到的matrics矩阵
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''
    ##定义常量？
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation. 在variable中定义变量，命名空间。
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

##告诉公路，有什么作用呢
def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y ———— g是非线性函数，t是 转换门， 1-t是c位门
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    
    数学表示很简单，但是为什么这样还是不知道呢。
    """
 
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_# 1.-t是个啥，看不懂。 
            input_ = output

    return output

###############从下面开始我看的还是云里雾里。 今晚逐行逐行的过

class Discriminator(object): ##判别器
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0): 
        ## 对D的初始化，也就是建立图吧， 参数有，句子长度，类别，词典大小，emb大小 卷积核大小？ 卷积核数量 l2正则项的值 
        # Placeholders for input, output and dropout  用了一个占位符坐形参，为了
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")#输入 batch x 句子长度
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  #标签 batch x 类别（0，1）
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") #dropout的超参数

        # Keeping track of l2 regularization loss (optional) L2正则项
        l2_loss = tf.constant(0.0)  

        with tf.variable_scope('discriminator'): # discriminator判别器的作用域
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"): #嵌入层的初始化？
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), #产生于low和high之间，产生的值是均匀分布的
                    name="W") #这个是初始化矩阵，vacab_size x emb size， vacab size是多大？ 有多少个呀
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) #得到input x对应的那一行向量。
                #input_x还没有给，也就是说，只是一个预运算，和函数里对形参操作一样。
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) #扩展维度？ #-1表示最后一维

            # Create a convolution + maxpool layer for each filter size 对每一个卷积核，都加上卷积，池化层。
            pooled_outputs = [] #这是一个列表， pooled层，输出。
            for filter_size, num_filter in zip(filter_sizes, num_filters):##zip
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filter] #卷积核矩阵 这个怎么是四维的呢，这个是怎么用的，奇怪。
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W") ## 就是初始化为正态分布，只不过这个可以截断
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")## b偏移值
                    '''
                    tf.nn.conv2d是TensorFlow里面实现卷积的函数，参考文档对它的介绍并不是很详细，实际上这是搭建卷积神经网络比较核心的一个方法，非常重要
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
除去name参数用以指定该操作的name，与方法有关的一共五个参数：
第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
那么TensorFlow的卷积具体是怎样实现的呢，用一些例子去解释它：
                    '''
                    conv = tf.nn.conv2d( ##2D的卷积层，不过这个是在干嘛呢？？
                        self.embedded_chars_expanded, #这个是卷积层的输入嘛
                        W,
                        strides=[1, 1, 1, 1], #为什么4个1，是因为有四个维度是吗
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") #conv出来之后，过一个relu
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool( #
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            ###上面这些，就是先用事先做好的输入，过一个卷积层，再过一个relu，在加一个max，最后把这些max出来的放到一个pooled好的列表当中
            ######################################################
            
            
            # Combine all the pooled features 这是在干嘛
            num_filters_total = sum(num_filters) #就是对里面东西求和
            self.h_pool = tf.concat(pooled_outputs, 3) #拼接张量，在第三个维度拼接
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) #然后把张量给reshape，压平。

            # Add highway
            with tf.name_scope("highway"): #这个是highway层，我是不是也可以写一个我自己的层，yue layer，并且是pool fat 过这个highway
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"): #加入doroput层  highway再加一个dropout
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions 最后的分数和预测
            with tf.name_scope("output"): 
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W") #相当于是全连接层吧，对的
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                
                #l2_loss一般用于优化目标函数中的正则项，防止参数太多复杂容易过拟合
                #(所谓的过拟合问题是指当一个模型很复杂时，它可以很好的“记忆”每一个训练数据中的随机噪声的部分而忘记了要去“学习”训练数据中通用的趋势)
                l2_loss += tf.nn.l2_loss(W) #这块是什么鬼
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores) #最后对 num classes个输出，做一个softmax
                self.predictions = tf.argmax(self.scores, 1, name="predictions") #最后再求一个argmax，作为最后的预测

            # CalculateMean cross-entropy loss 目标函数，目标函数有几种，分别有哪些区别
            with tf.name_scope("loss"): 
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)#计算 交叉熵 loss，tensorflow
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss#求loss均值，然后再加上一个正则项

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name] #所有再dis这个之下的变量门的参数，
        #用作接下来的更新
        
        # adam optimzier for discriminator adam的优化
        # may want to use SGD for stable learning of D
        d_optimizer = tf.train.AdamOptimizer(config['discriminator_lr'])
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2) #计算梯度下降
        self.train_op = d_optimizer.apply_gradients(grads_and_vars) #下降~！
