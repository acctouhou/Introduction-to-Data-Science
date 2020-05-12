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

batch_size=2048

nu1=1297
nu2=768
nu3=512
nu4=392
nu5=256
nu6=para*9

rotk='4'
path2=os.path.abspath('..')
x_data = np.loadtxt(open("%s\\data\\n_x_data%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("%s\\data\\n_y_data%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
l_data = np.loadtxt(open("%s\\data\\n_l_data%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)

x_test = np.loadtxt(open("%s\\data\\n_x_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("%s\\data\\n_y_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
l_test = np.loadtxt(open("%s\\data\\n_l_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)

'''
test_count=random.sample(range(len(x_testt)),20000)

x_test=x_testt[test_count]
y_test=y_testt[test_count]
l_test=l_testt[test_count]
'''

x = tf.placeholder(tf.float32, [None, para*9])
y= tf.placeholder(tf.float32, [None,para*9])
day_num=tf.placeholder(tf.float32, [None, 1])
x_reshape=tf.reshape(x, [-1,3,3,para])
train= tf.placeholder(tf.bool)


def cnn_fk(x,k,n):
    a1=tf.layers.conv2d(x,filters=n,kernel_size=(1,k))
    a2=tf.layers.conv2d(x,filters=n,kernel_size=(k,1))
    a3=tf.layers.conv2d(x,filters=n,kernel_size=(k,k))
    a4=tf.layers.conv2d(x,filters=n,kernel_size=(1,1))
    a1_=tf.reshape(a1, [-1,a1.shape[1]*a1.shape[2],a1.shape[3]])
    a2_=tf.reshape(a2, [-1,a2.shape[1]*a2.shape[2],a2.shape[3]])
    a3_=tf.reshape(a3, [-1,a3.shape[1]*a3.shape[2],a3.shape[3]])
    a4_=tf.reshape(a4, [-1,a4.shape[1]*a4.shape[2],a4.shape[3]])
    sum_filter=tf.nn.elu(tf.reshape(tf.concat([a1_,a2_,a3_,a4_],1),[-1,k+1,k+1,n]))
    return sum_filter

cc1=cnn_fk(x_reshape,3,128) 
cc2=cnn_fk(cc1,4,128)
cc3=cnn_fk(cc2,5,128)
cc4=cnn_fk(cc3,6,128)
cc5=cnn_fk(cc4,7,128)
cc6=cnn_fk(cc5,8,128)
def res_net(x,aa,k):
    with tf.variable_scope('resnet%d'%(aa))as scope:
        r1=tf.layers.conv2d(x,filters=k,kernel_size=(1,1))
        b1=tf.nn.elu(tf.layers.batch_normalization(r1,axis=3,training=train,fused=True))
        r2=tf.layers.conv2d(b1,filters=k,kernel_size=(3,3),padding='same')
        b2=tf.nn.elu(tf.layers.batch_normalization(r2,axis=3,training=train,fused=True))
        r3=tf.layers.conv2d(b2,filters=int(k*4),kernel_size=(1,1))
        b3=tf.nn.elu(tf.layers.batch_normalization(r3,axis=3,training=train,fused=True)+x)
    return b3
dict = {'rmax': 1, 'rmin': 1, 'dmax':0}

rn0=res_net(cc6,0,32)
for i in range(10):
    exec("rn%d=res_net(rn%d,%d,32)"%(i+1,i,i+1))

def dow_net(x,aa,k):
    with tf.variable_scope('resnet%d'%(aa))as scope:
        r1=tf.layers.conv2d(x,filters=k,kernel_size=(1,1))
        b1=tf.nn.elu(tf.layers.batch_normalization(r1,axis=3,training=train,fused=True))
        r2=tf.layers.conv2d(b1,filters=k,kernel_size=(3,3),padding='same')
        b2=tf.nn.elu(tf.layers.batch_normalization(r2,axis=3,training=train,fused=True))
        r3=tf.layers.conv2d(b2,filters=int(k*2),kernel_size=(1,1))
        b3=tf.nn.elu(tf.layers.batch_normalization(r3,axis=3,training=train,fused=True))
    return b3

rn21=dow_net(rn10,21,16)
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
saver.restore(sess, "my_net/save_net.ckpt")
#%%

name=['x displacement',
      'y displacement',
      'pressure',
      'x strain',
      'y strain',
      'z strain',
      '1 principal stress',
      '2 principal stress',
      '3 principal stress',
      'flow',
      'x flow',
      'y flow',
      'concentration',
      "young's modulus",
      "poisson's ratio",
      "permeability",
      "s coefficient"]


x_tol=np.vstack((x_data,x_test))
y_tol=np.vstack((y_data,y_test))
l_tol=np.vstack((l_data,l_test))
def col_d(a,b,c,d):
    x_tar=[]
    y_tar=[]
    l_tar=[]
    for i in range(33):
        tar=np.logical_and(c[:,2]==i,c[:,3]==d)
        x_tar.append(a[tar])
        y_tar.append(b[tar])
        l_tar.append(c[tar])
    return x_tar,y_tar,l_tar
x_tar,y_tar,l_tar=col_d(x_tol,y_tol,l_tol,1)
day_list=[0,20,32]
rotk='4'
y_mean = np.loadtxt(open("%s\\data\\n_y_mean%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
y_scale = np.loadtxt(open("%s\\data\\n_y_scale%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
y_var = np.loadtxt(open("%s\\data\\n_y_var%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)

x_mean = np.loadtxt(open("%s\\data\\n_x_mean%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
x_scale = np.loadtxt(open("%s\\data\\n_x_scale%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)
x_var = np.loadtxt(open("%s\\data\\n_x_var%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0)

#%%
def tra(wtf):
    #size=wtf.shape
    #aaa=np.reshape(wtf,[int(size[0]*size[1]),size[2]])
    ttem=wtf/y_scale*y_var+y_mean
    #ans=np.reshape(ttem,[size[0],size[1],size[2]])
    return ttem
def pool(inputs,loc,times):
    xx=loc[:,0]
    yy=loc[:,1]
    a=np.reshape(tra(inputs),[len(inputs),9,17])
    now_matrix=[]
    xmax=int(xx.max())
    ymax=int(yy.max())
    for i in range(9):
        aa=a[:,i,:]
        neww=np.zeros([int(xmax),int(ymax),para])
        neww[:,:,12]=0
        neww[:,:,16]=-1
        neww[:,:,14]=0.3
        neww[:,:,13]=113e9
        for tt in range(len(xx)):
            neww[int(xx[tt]-1),int(yy[tt]-1)]=aa[tt,:]
        now_matrix.append(neww)
    mo=np.zeros([int(xmax+2),int(ymax+2),9,para])
    mo[:xmax,:ymax,0,:]=now_matrix[0]            
    mo[1:(xmax+1),:ymax,1,:]=now_matrix[3]    
    mo[2:(xmax+2),:ymax,2,:]=now_matrix[6]
    mo[:xmax,1:(ymax+1),3,:]=now_matrix[1]            
    mo[1:(xmax+1),1:(ymax+1),4,:]=now_matrix[4]*times
    mo[2:(xmax+2),1:(ymax+1),5,:]=now_matrix[7]
    mo[:xmax,2:(ymax+2),6,:]=now_matrix[2]            
    mo[1:(xmax+1),2:(ymax+2),7,:]=now_matrix[5]            
    mo[2:(xmax+2),2:(ymax+2),8,:]=now_matrix[8]
    new=mo.sum(axis=2)
    from scipy import signal
    ggc=np.ones([3,3])
    ggc[1,1]=times
    ttt=signal.convolve2d(np.ones([xmax,ymax]),ggc)
    tta=ttt[:,:,np.newaxis]
    init=tta
    for i in range(16):
        init=np.concatenate((init,tta),axis=2)
    ave=new/init
    return ave,xmax,ymax


def col_d(a,b,c,d):
    x_tar=[]
    y_tar=[]
    l_tar=[]
    for i in range(33):
        tar=np.logical_and(c[:,2]==i,c[:,3]==d)
        x_tar.append(a[tar])
        y_tar.append(b[tar])
        l_tar.append(c[tar])
    return x_tar,y_tar,l_tar
def cyka(tt,ss,par):
    print(tt.shape)
    xlist=[]
    ylist=[]
    nt=np.zeros([ss[0]-2,ss[1]-2,par*9])
    for i in range (1,ss[0]-1,1):
        for j in range (1,ss[1]-1 ,1):
            temp=np.hstack((tt[i-1,j-1,:],
                              tt[i,j-1,:],
                              tt[i+1,j-1,:],
                              tt[i-1,j,:],
                              tt[i,j,:],#############################
                              tt[i+1,j,:],
                              tt[i-1,j+1,:],
                              tt[i,j+1,:],
                              tt[i+1,j+1,:],
                              ))
            nt[i-1,j-1,:]=temp
            xlist.append(j)
            ylist.append(i)
    xlist=np.asarray(xlist)
    ylist=np.asarray(ylist)
    if np.isnan(nt).sum()==0:
        return np.reshape(nt,[(ss[0]-2)*(ss[1]-2),par*9]),np.reshape(xlist,[(ss[0]-2)*(ss[1]-2),1]),np.reshape(ylist,[(ss[0]-2)*(ss[1]-2),1])
    else:
        print('fuck')
def normal(x):
    return (x-x_mean)*x_scale/x_var
def checke(inputs):
    wtf=inputs[:,13]>1.13e11
    for i in range(8):
        wtf=np.logical_or(inputs[:,int((i+1)*17+13)]>1.13e11,wtf)
    return wtf
deb=[]
def iterr(inputs,i,ck,times):
    aa,xli,yli=cyka(inputs.swapaxes(1,0),[204,37],para)
    dday=np.full([aa.shape[0],1],i)
    pre_y = sess.run(ans,feed_dict={x:normal(aa),day_num:dday,train:False})
    
    cm,xmax,ymax=pool(pre_y,np.hstack((xli,yli)),times)
    cm[ck,:]=0
    cm[ck,16]=-1
    cm[ck,14]=0.3
    cm[ck,13]=113e9
    
    return cm,xmax,ymax

x_tar,y_tar,l_tar=col_d(x_tol,y_tol,l_tol,1)





#%%

def look_s(inputs,mod):
    aaaa=np.reshape(inputs,[7548,])
    def s_classification(s):
        ans=np.zeros(len(s))
        for i in range(len(s)):
            if  s[i]>=-mod and s[i]<=0.0103:
                ans[i]=1
            elif  s[i]<=0.266 and s[i]>0.0103:
                ans[i]=5
            elif s[i]<=1 and s[i]>0.266:
                ans[i]=4
            elif s[i]<=3 and s[i]>1:
                ans[i]=3
            elif s[i]>3:
                ans[i]=2
        return ans
    ss1=s_classification(aaaa)
    s1=np.reshape(ss1,[37,204])
    return s1

x_tar,y_tar,l_tar=col_d(x_tol,y_tol,l_tol,3)

i=28
pre_y = sess.run(ans,feed_dict={x:x_tar[i],day_num:l_tar[i][:,2][:, np.newaxis],train:False})
cm,xmax,ymax=pool(pre_y,l_tar[i],100)
co,_,_=pool(y_tar[i],l_tar[i],100)
aaa=co[:,:,16]<-0.5
'''
def loc_map(loc):
    x=loc[:,0]
    y=loc[:,1]
    a=np.zeros([xmax,ymax])
    for i in range(len(x)):
        a[int(x[i]-1),int(y[i]-1)]=1
    return a
aaa=loc_map(l_tar[i])
'''
for j in range(1):
    cm,_,_=iterr(cm,i+1+j,aaa,100000)

cy,_,_=pool(y_tar[i+j+1],l_tar[i+1+j],100)
aa=cm[:,:,16]
bb=cy[:,:,16]
cc=co[:,:,16]
fig,axes=plt.subplots(1,3,figsize=(8,8),dpi=100, squeeze=False)
vx,vy=np.meshgrid(np.linspace(1,ymax+2,num=ymax+2),np.linspace(1,xmax+2,num=xmax+2))
s1=look_s(cy[:,:,16],0)
s11=look_s(cm[:,:,16],0)
s111=look_s(co[:,:,16],0)

axes[0,0].scatter(vy,vx,c=s1, cmap=cmaps,vmin=0,vmax=6)
axes[0,1].scatter(vy,vx,c=s11, cmap=cmaps,vmin=0,vmax=6)
axes[0,2].scatter(vy,vx,c=s111,cmap=cmaps,vmin=0,vmax=6)

'''
axes[0,0].scatter(vy,vx,c=bb,vmin=aa.min(),vmax=aa.max())
axes[0,1].scatter(vy,vx,c=aa,vmin=aa.min(),vmax=aa.max())
axes[0,2].scatter(vy,vx,c=cc,vmin=aa.min(),vmax=aa.max())
'''

#%%
from matplotlib import colors
colorslist = ['k','#FFFFFF','#FF00FF','#7700FF','#00FF00','#006400']
cmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=6)

day_list=[0,4,9,14,19,24,29]
for tt in range(len(day_list)):
    for i in range(3):
        init=day_list[tt]
        if not os.path.exists('%d'%(init)):
            os.makedirs('%d'%(init))
        x_tar,y_tar,l_tar=col_d(x_tol,y_tol,l_tol,i+1)
        pre_y = sess.run(ans,feed_dict={x:x_tar[init],day_num:l_tar[init][:,2][:, np.newaxis],train:False})
        cm,xmax,ymax=pool(pre_y,l_tar[i],100)
        co,_,_=pool(y_tar[i],l_tar[i],100)
        aaa=co[:,:,16]<-0.5
        for j in range(init,32):
           cm,_,_=iterr(cm,j+1,aaa,100)
           cy,_,_=pool(y_tar[j+1],l_tar[1+j],100)
           fig,axes=plt.subplots(1,3,figsize=(8,8),dpi=100, squeeze=False)
           vx,vy=np.meshgrid(np.linspace(1,ymax+2,num=ymax+2),np.linspace(1,xmax+2,num=xmax+2))
           s1=look_s(cy[:,:,16],0.5)
           s11=look_s(cm[:,:,16],0.5)
           s111=look_s(co[:,:,16],0.5)
           axes[0,0].scatter(vy,vx,c=s1, cmap=cmaps,vmin=0,vmax=6)
           axes[0,1].scatter(vy,vx,c=s11, cmap=cmaps,vmin=0,vmax=6)
           axes[0,2].scatter(vy,vx,c=s111,cmap=cmaps,vmin=0,vmax=6)
           plt.savefig('%d\\%d_%d.png'%(init,i,j))
           plt.clf()
#%%
from sklearn.metrics import accuracy_score

day_list=[0,4,9,14,19,24,29]
ccy=[]
for i in range(3):
    ta=[]
    for tt in range(len(day_list)):
        tb=[]
        init=day_list[tt]
        if not os.path.exists('%d'%(init)):
            os.makedirs('%d'%(init))
        x_tar,y_tar,l_tar=col_d(x_tol,y_tol,l_tol,i+1)
        pre_y = sess.run(ans,feed_dict={x:x_tar[init],day_num:l_tar[init][:,2][:, np.newaxis],train:False})
        cm,xmax,ymax=pool(pre_y,l_tar[i],100)
        aaa=co[:,:,16]<-0.5
        for j in range(init,32):
               cm,_,_=iterr(cm,j+1,aaa,100)
               cy,_,_=pool(y_tar[j+1],l_tar[1+j],100)
               vx,vy=np.meshgrid(np.linspace(1,ymax+2,num=ymax+2),np.linspace(1,xmax+2,num=xmax+2))
               s1=look_s(cy[:,:,16],0.5)
               s11=look_s(cm[:,:,16],0.5)
               per=accuracy_score(s1.flatten(),s11.flatten())
               tb.append(per)
        ta.append(tb)
    ccy.append(ta)
#%%
ta=ccy[0]   
tb=ccy[1]   
tc=ccy[2]   

bba=[]
bbb=[]
bbc=[]
fig,axes=plt.subplots(figsize=(10,6),dpi=100, squeeze=False)
for i in range(len(ta)):
    bba.append(ta[i][-1])
    bbb.append(tb[i][-1])
    bbc.append(tc[i][-1])
plt.style.use('ggplot')
plt.plot(day_list,bba,label='model1')
plt.plot(day_list,bbb,label='model2')
plt.plot(day_list,bbc,label='model3')
plt.xticks(day_list)
plt.legend(fontsize=20)
plt.xlabel('day to final',fontsize=20)
plt.ylabel('correct area',fontsize=20)
#plt.savefig('day.png')
#%%
fig,axes=plt.subplots(figsize=(10,6),dpi=100, squeeze=False)

for i in range(len(ta)):
    plt.plot(ta[i],label='%d day'%(day_list[i]))
plt.legend(fontsize=20)
plt.xlabel('predict day',fontsize=20)
plt.ylabel('correct area',fontsize=20)
plt.title('model 1',fontsize=28)
#plt.savefig('gg1.png')
fig,axes=plt.subplots(figsize=(10,6),dpi=100, squeeze=False)

for i in range(len(ta)):
    plt.plot(tb[i],label='%d day'%(day_list[i]))
plt.legend(fontsize=20)
plt.xlabel('predict day',fontsize=20)
plt.ylabel('correct area',fontsize=20)
plt.title('model 2',fontsize=28)
#plt.savefig('gg2.png')
fig,axes=plt.subplots(figsize=(10,6),dpi=100, squeeze=False)

for i in range(len(ta)):
    plt.plot(tc[i],label='%d day'%(day_list[i]))
plt.legend(fontsize=20)
plt.xlabel('predict day',fontsize=20)
plt.ylabel('correct area',fontsize=20)
plt.title('model 3',fontsize=28)
#plt.savefig('gg3.png')

#%%

    
#plt.legend()
#%%
for i in range(1):
    cm,xmax,ymax=iterr(cm,i+1)
    cy,_,_=pool(y_tar[i+1],l_tar[i+1],1)
    s1=look_s(cy[:,:,16],5e-3)
    s11=look_s(cm[:,:,16],5e-3)
    
    fig,axes=plt.subplots(1,2,figsize=(8,8),dpi=100, squeeze=False)
    vx,vy=np.meshgrid(np.linspace(1,ymax+2,num=ymax+2),np.linspace(1,xmax+2,num=xmax+2))
    axes[0,0].scatter(vy,vx,c=s1, cmap=cmaps,vmin=0,vmax=6)
    axes[0,1].scatter(vy,vx,c=s11, cmap=cmaps,vmin=0,vmax=6)

    #axes[0,0].scatter(vy,vx,c=s11)
    #axes[0,1].scatter(vy,vx,c=s1)

    plt.savefig('%d.png'%(i))
    plt.clf()

#%%
from sklearn.metrics import accuracy_score

day_list=[0,4,9,14,19,24,29]
ccy=[]
for i in range(3):
    ta=[]
    for tt in range(len(day_list)):
        tb=[]
        init=day_list[tt]
        if not os.path.exists('%d'%(init)):
            os.makedirs('%d'%(init))
        x_tar,y_tar,l_tar=col_d(x_tol,y_tol,l_tol,i+1)
        co,xmax,ymax=pool(y_tar[init],l_tar[init],100)
        for j in range(init,30):
               cy,_,_=pool(y_tar[j+3],l_tar[3+j],100)
               vx,vy=np.meshgrid(np.linspace(1,ymax+2,num=ymax+2),np.linspace(1,xmax+2,num=xmax+2))
               s1=look_s(cy[:,:,16],0.5)
               s11=look_s(co[:,:,16],0.5)
               per=accuracy_score(s1.flatten(),s11.flatten())
               tb.append(per)
        ta.append(tb)
    ccy.append(ta)

#%%












