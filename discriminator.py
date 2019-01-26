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

class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")#输入 batch x 句子长度
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  #标签 batch x 类别（0，1）
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") #dropout的超参数

        # Keeping track of l2 regularization loss (optional) L2正则项
        l2_loss = tf.constant(0.0)  

        with tf.variable_scope('discriminator'): # discriminator判别器的作用域
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"): #嵌入层的初始化？
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W") #这个是初始化矩阵，vacab_size x emb size， vacab size是多大？ 有多少个呀
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) #得到input x对应的那一行向量。
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) #扩展维度？ #-1表示最后一维

            # Create a convolution + maxpool layer for each filter size 对每一个卷积核，都加上卷积，池化层。
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filter] #卷积核矩阵
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features 这是在干嘛
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"): #加入doroput层
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)   ########################这个之前我都看不懂。。。。哈哈
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)#计算loss，tensorflow
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        # adam optimzier for discriminator
        # may want to use SGD for stable learning of D
        d_optimizer = tf.train.AdamOptimizer(config['discriminator_lr'])
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
