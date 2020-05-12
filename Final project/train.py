#coding:utf-8
import sys
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
t0 = time.time()
para=17

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size,mean=0.0,stddev=xavier_stddev)
def lecun(x):
    in_dim = size[0]
def bh1(x,train):
    if train==True:
        mean, var = tf.nn.moments(x,axes=[0])
        b=tf.rsqrt(var)
        c=(x-mean)*b
        d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=0.001)
        return d
    else:
        return x
def bhy(x,train):
    if train==True:
        mean, var = tf.nn.moments(x,axes=[0])
        b=tf.rsqrt(var)
        c=(x-mean)*b
        d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=0.001)
        return d
    else:
        return x

def bht(x):
    mean, var = tf.nn.moments(x,axes=[0])
    b=tf.rsqrt(var)
    return mean,b
def bhx(x):
    mean, var = tf.nn.moments(x,axes=[0])
    d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=1e-8)
    return d

batch_size=1024

nu1=1297
nu2=768
nu3=512
nu4=392
nu5=256
nu6=para*9
#%%
rotk='4'
path2=os.path.abspath('..')
x_data = np.loadtxt(open("%s\\data\\n_x_data%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("%s\\data\\n_y_data%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
l_data = np.loadtxt(open("%s\\data\\n_l_data%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)

x_test = np.loadtxt(open("%s\\data\\n_x_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("%s\\data\\n_y_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
l_test = np.loadtxt(open("%s\\data\\n_l_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
#%%
'''
test_count=random.sample(range(len(x_testt)),20000)

x_test=x_testt[test_count]
y_test=y_testt[test_count]
l_test=l_testt[test_count]
'''
#%%
x = tf.placeholder(tf.float32, [None, para*9])
y= tf.placeholder(tf.float32, [None,para*9])
day_num=tf.placeholder(tf.float32, [None, 1])
x_reshape=tf.reshape(x, [-1,3,3,para])
train= tf.placeholder(tf.bool)

dict = {'rmax': 1, 'rmin': 1, 'dmax':0}

def cnn_fk(x,k,n):
    a1=tf.layers.conv2d(x,filters=n,kernel_size=(1,k))
    a2=tf.layers.conv2d(x,filters=n,kernel_size=(k,1))
    a3=tf.layers.conv2d(x,filters=n,kernel_size=(k,k))
    a4=tf.layers.conv2d(x,filters=n,kernel_size=(1,1))
    a1_=tf.reshape(a1, [-1,a1.shape[1]*a1.shape[2],a1.shape[3]])
    a2_=tf.reshape(a2, [-1,a2.shape[1]*a2.shape[2],a2.shape[3]])
    a3_=tf.reshape(a3, [-1,a3.shape[1]*a3.shape[2],a3.shape[3]])
    a4_=tf.reshape(a4, [-1,a4.shape[1]*a4.shape[2],a4.shape[3]])
    sum_filter=tf.nn.elu(tf.layers.batch_normalization(tf.reshape(tf.concat([a1_,a2_,a3_,a4_],1),[-1,k+1,k+1,n]),axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict))
    return sum_filter

cc1=cnn_fk(x_reshape,3,32) 
cc2=cnn_fk(cc1,4,32)
cc3=cnn_fk(cc2,5,64)
cc4=cnn_fk(cc3,6,64)
cc5=cnn_fk(cc4,7,128)
cc6=cnn_fk(cc5,8,128)

def res_net(x,aa,k):
    with tf.variable_scope('resnet%d'%(aa))as scope:
        r1=tf.layers.conv2d(x,filters=k,kernel_size=(1,1))
        b1=tf.nn.elu(tf.layers.batch_normalization(r1,axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict))
        r2=tf.layers.conv2d(b1,filters=k,kernel_size=(3,3),padding='same')
        b2=tf.nn.elu(tf.layers.batch_normalization(r2,axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict))
        r3=tf.layers.conv2d(b2,filters=int(k*4),kernel_size=(1,1))
        b3=tf.nn.elu(tf.layers.batch_normalization(r3,axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict)+x)
    return b3
dict = {'rmax': 1, 'rmin': 1, 'dmax':0}

rn0=res_net(cc6,0,32)
for i in range(5):
    exec("rn%d=res_net(rn%d,%d,32)"%(i+1,i,i+1))

def dow_net(x,aa,k):
    with tf.variable_scope('resnet%d'%(aa))as scope:
        r1=tf.layers.conv2d(x,filters=k,kernel_size=(1,1))
        b1=tf.nn.elu(tf.layers.batch_normalization(r1,axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict))
        r2=tf.layers.conv2d(b1,filters=k,kernel_size=(3,3),padding='same')
        b2=tf.nn.elu(tf.layers.batch_normalization(r2,axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict))
        r3=tf.layers.conv2d(b2,filters=int(k*2),kernel_size=(1,1))
        b3=tf.nn.elu(tf.layers.batch_normalization(r3,axis=3,training=train,fused=True,renorm=True,renorm_clipping=dict))
    return b3

rn21=dow_net(rn5,21,16)
rn24=dow_net(rn21,24,8)

flat=tf.reshape(rn24,[-1,1296])

x_center=x_reshape[:,1,1,:]

add_x=tf.concat([flat,day_num],1)
x_center=x_reshape[:,1,1,:]


#tf.keras.initializers.lecun_normal()
#tf.glorot_uniform_initializer()
wwtf=tf.contrib.layers.xavier_initializer()
for i in range(5):
    exec("W%d = tf.get_variable('W%d',shape=[nu%d,nu%d], initializer=wwtf)" % (i+1,i+1,i+1,i+2))
    exec("b%d = tf.Variable(tf.zeros(shape=[nu%d]))" % (i+1,i+2))


h1 = tf.matmul(add_x,W1) + b1
bn1 = tf.nn.elu(tf.layers.batch_normalization(h1, training=train,epsilon=1e-9,momentum=0.99,fused=True))
h2 = tf.matmul(bn1,W2) + b2
bn2 = tf.nn.elu(tf.layers.batch_normalization(h2, training=train,epsilon=1e-9,momentum=0.99,fused=True))
h3 = tf.matmul(bn2,W3) + b3
bn3 = tf.nn.elu(tf.layers.batch_normalization(h3, training=train,epsilon=1e-9,momentum=0.99,fused=True))
h4 = tf.matmul(bn3,W4) + b4
bn4 = tf.nn.elu(tf.layers.batch_normalization(h4, training=train,epsilon=1e-9,momentum=0.99,fused=True))
bn10 = tf.layers.batch_normalization(tf.matmul(bn4,W5) + b5, training=train,epsilon=1e-9,momentum=0.99,fused=True)

bn10_s=tf.reshape(bn10,[-1,3,3,para])
bn10_c=bn10_s[:,1,1,:]

ans=bn10+x
ans_s=tf.reshape(ans,[-1,3,3,para])
ans_c=ans_s[:,1,1,:]

y_reshape=tf.reshape(y, [-1,3,3,para])
y_center=y_reshape[:,1,1,:]

y_=y-x
y_gg=tf.reshape(y_, [-1,3,3,para])
y_ggc=y_gg[:,1,1,:]


loss= tf.reduce_sum(tf.square(y_-bn10))

test_loss1=y-ans
test_loss2=y_center-ans_c

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    step = tf.train.AdamOptimizer().minimize(loss)
    #step =tf.contrib.opt.NadamOptimizer().minimize(loss)
#%%
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


###################################
v_all = tf.trainable_variables()
g_list = tf.global_variables()
bn_mean = [g for g in g_list if 'moving_mean' in g.name]
bn_var = [g for g in g_list if 'moving_variance' in g.name]
rn_std=[g for g in g_list if 'renorm_stddev' in g.name]
rn_mean=[g for g in g_list if 'renorm_mean' in g.name]
gamma = [g for g in v_all if 'gamma' in g.name]
beta = [g for g in v_all if 'beta' in g.name]
we = [g for g in v_all if 'W' in g.name]
optimizer = tf.train.AdamOptimizer()
gradients = optimizer.compute_gradients(loss,we)
for tt1 in range(len(gradients)):
    exec("g%d=gradients[%d]" % (tt1,tt1))


def mv_check():
    bv=sess.run(bn_var)
    bm=sess.run(bn_mean)
    ga=sess.run(gamma)
    be=sess.run(beta)
    for temp in range(len(bm)):
        plt.plot(bv[temp],label='bn_var')
        plt.plot(ga[temp],label='gamma')
        plt.legend(loc='upper right')
        plt.title('gamma-bn_var-L%d'%(temp+1))
        plt.savefig('gamma%d'%(temp+1))
        plt.clf()
        plt.plot(bm[temp],label='bn_mean')
        plt.plot(be[temp],label='beta')
        plt.legend(loc='upper right')
        plt.title('beta-bn_mean-L%d'%(temp+1))
        plt.savefig('beta%d'%(temp+1))
        plt.clf()
    
     
def plott(a,b,c,i,wtf,tt,e):
            plt.scatter(a[:,i-1]*c, b[:,i-1]*c, s=0.5, c='b', alpha=.5)
            plt.plot(b[:,i-1]*c,b[:,i-1]*c, 'r--')
            ee=np.abs(e).mean()
            plt.title('%d_%s [r=%.3f e=%.3f]'%(i,tt,wtf,ee))
            plt.ylabel("correct")
            plt.xlabel("prediction")
            plt.savefig('%s%d_.png'%(tt,i))
            plt.clf()

def data_train(x_data,y_data,locl,a):
    size=len(x_data)
    test_count=random.sample(range(size),int(size*a))
    x_vail=x_data[test_count,:]
    y_vail=y_data[test_count,:]
    l_vail=locl[test_count,:]
    l_train=np.delete(locl, test_count, 0)
    x_train=np.delete(x_data, test_count, 0)
    y_train=np.delete(y_data, test_count, 0)
    return x_vail,y_vail,x_train,y_train,l_train,l_vail

def print_error(error,xx,yy,a,sss):
    cm = plt.cm.get_cmap('RdYlBu')
    sc=plt.scatter(xx,yy,s=75,alpha=0.5,c=error,cmap=cm, vmin=-0.3, vmax=0.3)
    plt.colorbar(sc)
    plt.title('location_%d_%s'%(a,sss))
    plt.savefig('errorlocal_%d_%s.png'%(a,sss))
    plt.clf()
def error_check(error,local,sss):
    a0=[]
    b0=[]
    for w in range(para):
        a0.append(error[:,w]>0.3)
        b0.append(error[:,w]<-0.3)
    total=a0[0]+b0[0]
    for tt in range(para-1):
        total+=a0[tt+1]
        total+=b0[tt+1]
    err_info=local[total]
    #errr=error[total]
    day=err_info[:,2]
    model=err_info[:,3]
    plt.hist(day,bins=33)
    plt.title('day_error_%s'%(sss))
    plt.ylabel("number")
    plt.savefig('day_error_%s.png'%(sss))
    plt.clf()
    plt.hist(model,bins=3)
    plt.title('model_error_%s'%(sss))
    plt.ylabel("number")
    plt.savefig('model_error_%s.png'%(sss))
    plt.clf()
    f, ax = plt.subplots(figsize = (4,10))
    ax.scatter(err_info[:,0],err_info[:,1],s=75,alpha=0.5)
    plt.title('location_%s'%(sss))
    plt.savefig('errorlocal_%s.png'%(sss))
    plt.clf()
    for gg20 in range(para):
        print_error(error[:,gg20],local[:,0],local[:,1],gg20+1,sss)
        
dx = tf.placeholder(tf.float32, [None,para*9])
dy= tf.placeholder(tf.float32, [None,para*9])
dl= tf.placeholder(tf.float32, [None,1])

loss_fff=[]
loss_vvv=[]

dataset = tf.data.Dataset.from_tensor_slices((dx,dy,dl)).batch(batch_size).shuffle(buffer_size=10000)
iterator = dataset.make_initializable_iterator()
x_in, y_in,l_in = iterator.get_next()

def error_data(a,b,c,d,e):
    m=a.mean(axis=0)
    v=a.var(axis=0)
    mt=a.mean()
    vt=a.var()
    b.append(mt)
    c.append(vt)
    plt.plot(b)
    plt.title('Bias_error')
    plt.xlabel("Iteration")
    plt.savefig('m_total.png')
    plt.clf()
    plt.plot(c)
    plt.title('Variance_error')
    plt.xlabel("Iteration")
    plt.savefig('v_total.png')
    plt.clf()
    d=np.vstack((d,m))
    e=np.vstack((e,v))
    for bb in range(para):
        plt.plot(d[:,bb],label="%d"%(bb+1)) 
    plt.legend(loc='upper right')
    plt.title('Bias_error 1-15')
    plt.xlabel("Iteration")
    plt.savefig('m1_15.png')
    plt.clf()
    for bb in range(para):
        plt.plot(e[:,bb],label="%d"%(bb+1))
    plt.title('Variance_error 1-15')
    plt.xlabel("Iteration")
    plt.legend(loc='upper right')
    plt.savefig('v1_15.png')
    plt.clf()
    
       
    return b,c,d,e
g_list = tf.global_variables()
saver = tf.train.Saver(var_list=g_list, max_to_keep=5)
#check=tf.test.compute_gradient_error(x_train,[100,54],y_train,[100,6],x_init_value=start)
########### pre train
#saver.restore(sess, "my_net/save_net.ckpt")
#%%

dict = {'rmax': 1, 'rmin': 1, 'dmax':0}
for i in range(int(10)):
    x_vail,y_vail,x_train,y_train,l_train,l_vail=data_train(x_data,y_data,l_data,0.1)
    
    sess.run(iterator.initializer,feed_dict={dx:x_train,dy:y_train,dl:l_train[:,2][:, np.newaxis]})
    long=int(len(x_train)/batch_size)
    print(i)
    for j in range(long):
        temp_x,temp_y,temp_l=sess.run([x_in,y_in,l_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,day_num:temp_l,train:True})

        
for i in range(int(100)):
    x_vail,y_vail,x_train,y_train,l_train,l_vail=data_train(x_data,y_data,l_data,0.1)
    
    sess.run(iterator.initializer,feed_dict={dx:x_train,dy:y_train,dl:l_train[:,2][:, np.newaxis]})
    long=int(len(x_train)/batch_size)
    for j in range(long):
        temp_x,temp_y,temp_l=sess.run([x_in,y_in,l_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,day_num:temp_l,train:True})
    if(i%10==0):
        print(i)
        loss_value = sess.run(loss,feed_dict={x:x_vail,y:y_vail,train:True,day_num:l_vail[:,2][:, np.newaxis]})
        loss_fff.append(loss_value)
        loss_value = sess.run(loss,feed_dict={x:x_test,y:y_test,train:True,day_num:l_test[:,2][:, np.newaxis]})
        loss_vvv.append(loss_value)

dict = {'rmax': 1, 'rmin': 0, 'dmax':0}
for i in range(int(300)):
    x_vail,y_vail,x_train,y_train,local=data_train(x_data,y_data,l_data,0.1)
    sess.run(iterator.initializer, feed_dict={dx:x_train,dy:y_train})
    long=int(len(x_train)/batch_size)
    for j in range(long):
        temp_x,temp_y=sess.run([x_in,y_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True})
    if(i%10==0):
        print(i)
        loss_value = sess.run(loss,feed_dict={x:x_vail,y:y_vail,train:True,day_num:l_vail[:,2][:, np.newaxis]})
        loss_fff.append(loss_value)
        loss_value = sess.run(loss,feed_dict={x:x_test,y:y_test,train:True,day_num:l_test[:,2][:, np.newaxis]})
        loss_vvv.append(loss_value)

dict = {'rmax': 3, 'rmin':0, 'dmax':5}  
for i in range(int(1e6)):
    x_vail,y_vail,x_train,y_train,l_train,l_vail=data_train(x_data,y_data,l_data,0.1)
    sess.run(iterator.initializer,feed_dict={dx:x_train,dy:y_train,dl:l_train[:,2][:, np.newaxis]})
    long=int(len(x_train)/batch_size)
    for j in range(long):
        temp_x,temp_y,temp_l=sess.run([x_in,y_in,l_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,day_num:temp_l,train:True})
    if(i%10==0):
        print(i)
        loss_value = sess.run(loss,feed_dict={x:x_vail,y:y_vail,day_num:l_vail[:,2][:, np.newaxis],train:True})
        loss_fff.append(loss_value)
        error = sess.run(test_loss2,feed_dict={x:x_vail,y:y_vail,day_num:l_vail[:,2][:, np.newaxis],train:False})
        error_check(error,l_vail,'train')
        f, ax = plt.subplots(figsize = (8,6))
        temp,gg = sess.run([ans_c,y_center],feed_dict={x:x_vail,y:y_vail,day_num:l_vail[:,2][:, np.newaxis],train:False})
        print('---------vail------------')
        print('loss:',loss_value)
        for uu in range(para):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e+2,%d,temp_%d[0],'vali',error[:,%d])"%(uu+1,uu,uu))
        print('-------------------------')
        plt.plot(loss_fff)
        plt.title('loss')
        plt.xlabel("Iteration")
        #plt.ylim(0,2e5)
        plt.savefig('loss_.png')
        plt.clf()
        loss_value = sess.run(loss,feed_dict={x:x_test,y:y_test,day_num:l_test[:,2][:, np.newaxis],train:True})
        loss_vvv.append(loss_value)
        print('loss:',loss_value)
        plt.plot(loss_vvv)
        plt.title('loss')
        plt.xlabel("Iteration")
        plt.savefig('loss_t.png') 
        plt.clf()
    if(i%50==0):
        temp,gg = sess.run([bn10_c,y_ggc],feed_dict={x:x_test,y:y_test,day_num:l_test[:,2][:, np.newaxis],train:False})
        for uu in range(para):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e+2,%d,temp_%d[0],'res_test',error[:,%d])"%(uu+1,uu,uu))
        
        error = sess.run(test_loss2,feed_dict={x:x_test,y:y_test,day_num:l_test[:,2][:, np.newaxis],train:False})
        error_check(error,l_test,'test')
        f, ax = plt.subplots(figsize = (8,6))
        temp,gg = sess.run([ans_c,y_center],feed_dict={x:x_test,y:y_test,day_num:l_test[:,2][:, np.newaxis],train:False})
        print('---------test------------')
        for uu in range(para):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e+2,%d,temp_%d[0],'test',error[:,%d])"%(uu+1,uu,uu))
        print('-------------------------')
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        
        '''
    if(i%20==0):
        loss_value = sess.run(loss,feed_dict={x:temp_x,y:temp_y,train:True})
        
        temp,gg = sess.run([ans,y],feed_dict={x:x_test,y:y_test,train:False})
        #temp=temp/y_var+y_mean
        #temp1,gg1 = sess.run([ans,y_ans],feed_dict={x:x_test,y:y_test,train:False})
        #temp=-np.log((temp**-1)-1)
        print('---------test------------')
        #print(int(batch_size*i/size))
        print('loss:',loss_value)
        temp_0=stats.pearsonr(gg[:,0],temp[:,0])[0]
        temp_1=stats.pearsonr(gg[:,1],temp[:,1])[0]
        temp_2=stats.pearsonr(gg[:,2],temp[:,2])[0]
        temp_3=stats.pearsonr(gg[:,3],temp[:,3])[0]
        temp_4=stats.pearsonr(gg[:,4],temp[:,4])[0]
        temp_5=stats.pearsonr(gg[:,5],temp[:,5])[0]
        print(temp_0)
        print(temp_1)
        print(temp_2)
        print(temp_3)
        print(temp_4)
        print(temp_5)
        #error(temp,gg)
        print('-------------------------')
        #scatter
        plott(temp,gg,1e-2,1)
        plott(temp,gg,1e-3,2)
        plott(temp,gg,1e-2,3)
        plott(temp,gg,1e-2,4)
        plott(temp,gg,1,5)#-6
        plott(temp,gg,1,6)#-5

'''
