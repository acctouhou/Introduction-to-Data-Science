import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import random



aaa=0.000001
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



training_epochs = 1000000
batch_size = 128
n_input = 784
X = tf.placeholder("float", [None, n_input])
tt= tf.placeholder("float", [None,10])
n_hidden_1 = 392
n_hidden_2 = 196
n_hidden_3 = 98
n_hidden_4 = 10

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

weights = {
    'encoder_h1': tf.Variable(xavier_init([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(xavier_init([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(xavier_init([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(xavier_init([n_hidden_3, n_hidden_4])),
    'encoder_h4_1': tf.Variable(xavier_init([n_hidden_3, n_hidden_4])),
    'decoder_h1': tf.Variable(xavier_init([n_hidden_4, n_hidden_3])),
    'decoder_h2': tf.Variable(xavier_init([n_hidden_3, n_hidden_2])),
    'decoder_h3': tf.Variable(xavier_init([n_hidden_2, n_hidden_1])),
    'decoder_h4': tf.Variable(xavier_init([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1], mean=0, stddev=1)),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2], mean=0, stddev=1)),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3], mean=0, stddev=1)),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4], mean=0, stddev=1)),
    'encoder_b4_1': tf.Variable(tf.random_normal([n_hidden_4], mean=0, stddev=1)),
    
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3], mean=0, stddev=1)),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2], mean=0, stddev=1)),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1], mean=0, stddev=1)),
    'decoder_b4': tf.Variable(tf.random_normal([n_input], mean=0, stddev=1)),
}
def encoder(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    mu = tf.add(tf.matmul(layer_3, weights['encoder_h4']),biases['encoder_b4'])
    logvar=tf.add(tf.matmul(layer_3, weights['encoder_h4_1']),biases['encoder_b4_1'])
    
    return mu,logvar
def decoder(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4'])

    return layer_4
def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


    
def test_4(x,n):
    x_1=x[:,0]
    x_2=x[:,1]
    x_1_max=x_1.max()
    x_1_min=x_1.min()
    x_2_max=x_2.max()
    x_2_min=x_2.min()
    c=np.zeros((n*n,2))
    a=np.linspace(x_1_min,x_1_max,n,dtype=np.float32)[:, np.newaxis]
    b=np.linspace(x_2_min,x_2_max,n,dtype=np.float32)[:, np.newaxis]
    for i in range(n*n):
        c[i]=[a[int(i/n)],b[int(i%n)]]

    return c
def test_5(x,n,time):
    f, a = plt.subplots(n,n, figsize=(n,n))
    for i in range(n*n):
        #a[int(i%n)][n-int(i/n)-1].imshow(np.reshape(x[i], (28, 28)))
        a[n-int(i%n)-1][int(i/n)].imshow(np.reshape(x[i], (28, 28)))
    plt.savefig('try%d.png'%(time))
    plt.clf()
def test_3(a,i,tt):
    plt.scatter(a[:, 0],a[:, 1], c=tt)
    plt.colorbar()
    plt.savefig('dem%d.png'%(i))
    plt.clf()
def test_1(g,b,tt,n):
    f, a = plt.subplots(6, 10, figsize=(10, 6))
    for i in range(10):
        a[0][i].imshow(np.reshape(b[i], (28, 28)))
        a[1][i].imshow(np.reshape(g[i], (28, 28)))
        a[2][i].imshow(np.reshape(b[i+10], (28, 28)))
        a[3][i].imshow(np.reshape(g[i+10], (28, 28)))
        a[4][i].imshow(np.reshape(b[i+20], (28, 28)))
        a[5][i].imshow(np.reshape(g[i+20], (28, 28)))
    plt.savefig('%dtrain%d.png'%(n,tt))
    plt.clf()
def test_6(x,time):
    f, a = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            a[j][i].imshow(np.reshape(x[i+10*j], (28, 28)))
        
        #a[1][i].imshow(np.reshape(x[i+10], (28, 28)))
    plt.savefig('wtf%d.png'%(time))
    plt.clf()
def test_2(temp,tt):
    tttt=np.zeros((64,1))
    tttt=tt.argmax(1)
    for i in range(64):
        plt.plot(temp[i],color = 'b')
        plt.savefig('test %d.png'%(tttt[i]))
        plt.clf()
def test_8(temp,tt):
    tttt=np.zeros((64,1))
    tttt=tt.argmax(1)
    for i in range(64):
        plt.plot(temp[i],color = 'b')
        plt.savefig('mu %d.png'%(tttt[i]))
        plt.clf()
def test_9(temp,tt):
    tttt=np.zeros((64,1))
    tttt=tt.argmax(1)
    for i in range(64):
        plt.plot(temp[i],color = 'b')
        plt.savefig('logvar %d.png'%(tttt[i]))
        plt.clf()      
z_mu,z_logvar = encoder(X)
z_sample = sample_z(z_mu, z_logvar)
decoder_op = decoder(z_sample)
decoder_tt = decoder(tt)
y_pred = decoder_op
y_true = X
#recon_loss=tf.reduce_mean(tf.pow(decoder_op - X, 2))
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=decoder_op, labels=X), 1)
#kl_loss = -0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_logvar - tf.square(z_mu) - tf.exp(2.0 * z_logvar), 1)
cost_2=tf.reduce_sum(tf.pow(tt - z_sample, 2))
cost = tf.reduce_mean(recon_loss + kl_loss+cost_2)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1,global_step,decay_steps=500,decay_rate=0.95)
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
sess.run(init)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
total_batch = int(mnist.train.num_examples/batch_size)
ttt=np.eye(10, dtype=int)
tttt=np.zeros((10,10))
for j in range (10):
    for i in range (10):
        if i<1:
            tttt[i]=(ttt[j]*(10-i)+ttt[j+1]*(i))/10
        else:
            tttt[i]=(ttt[j]*(10-i)+ttt[j+1]*(i))/10
    ttt=np.vstack((ttt,tttt))

i = 0
#noise_factor = 0.5
test_d=mnist.test.images[:30]

for epoch in range(training_epochs):
    add_global = global_step.assign_add(1)
    for i in range(total_batch):
        
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise=np.random.randint(2, size=batch_xs.shape)
        noise_x=batch_xs*noise
        _ = sess.run(optimizer, feed_dict={X: noise_x,tt:batch_ys})
    if epoch % 10 == 0:
        noise=np.random.randint(2, size=test_d.shape)
        noise_test=test_d*noise
        temp_1= sess.run(y_pred, feed_dict={X:noise_test})
        temp_1=sess.run(tf.nn.sigmoid(temp_1))
        test_1(temp_1,mnist.test.images,epoch,1)
        #test_1(temp_2,mnist.test.images,epoch,2)
        c=sess.run(cost, feed_dict={X: noise_x,tt:batch_ys})
        print("Epoch:", '%04d' % (epoch),"cost=", "{:.9f}".format(c))
        #print("Epoch:", '%04d' % (epoch),"cost_1=", "{:.9f}".format(c_1))
        #print("Epoch:", '%04d' % (epoch),"cost_2=", "{:.9f}".format(c_2))
        #print(sess.run(learning_rate,feed_dict={global_step:epoch}))
        temp_4=sess.run(decoder_tt, feed_dict={tt:ttt})
        temp_4=sess.run(tf.nn.sigmoid(temp_4))
        test_6(temp_4,epoch)
    if epoch % 500 == 0:
        temp_3,temp_5,temp_6=sess.run([z_sample,z_mu,z_logvar], feed_dict={X: mnist.test.images[:64]})
        test_8(temp_5,mnist.test.labels[:64])
        test_9(temp_6,mnist.test.labels[:64])
        test_2(temp_3,mnist.test.labels[:64])    


