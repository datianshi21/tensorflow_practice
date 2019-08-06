import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


lr= 0.001
#training_iters = 1000000
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10



xs = tf.placeholder(tf.float32,[None,n_inputs,n_steps])
ys = tf.placeholder(tf.float32,[None,10])


weights = {
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units, 64]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[64,]))
}


def RNN(X,weights,biases):
    # hidden layer for input to cell
    # X(128,28,28)
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in']) + biases['in']
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    #这里生成的state是tuple类型的，因为声明了state_is_tuple参数
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    #time_major指时间点是不是在主要的维度，因为我们的num_steps在次维，所以定义为了false
    outputs,final_states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)

    #final_states[1] 就是短时记忆h
    results = tf.matmul(final_states[1],weights['out']) + biases['out']

    return results
#xs: Shape: [batch_size, n_inputs, n_steps]
inputs = tf.reshape(xs, [batch_size, n_inputs * n_steps])

flag_tensor = tf.ones(shape=[batch_size, 1], dtype=tf.float32)
flag_tensor = tf.tile(flag_tensor, [1, num_outputs])
ones_tensor = tf.ones(shape=batch_size, num_outputs], dtype=tf.float32)

layer0 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=64, activation_fn=tf.nn.relu) #Shape: [batch_size, num_outputs]

#res = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=64, activation_fn=tf.nn.relu) #Shape: [batch_size, num_outputs]

#layer0 = tf.where(tf.equal(flag_tensor, ones_tensor), res, layer0)


#layer0 = RNN(xs,weights,biases)

prediction = tf.contrib.layers.fully_connected(inputs=layer0, num_outputs=n_classes, activation_fn=tf.nn.relu)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction,axis=1),tf.argmax(ys,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

acc_array = []
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #一个step是一行
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        if step % 100 == 0:
            print (batch_xs, batch_ys)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if step % 20 == 0:
            acc = sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys})
            print (acc)
            acc_array.append(acc)
        step = step + 1
plt.plot(acc_array)
plt.show()

