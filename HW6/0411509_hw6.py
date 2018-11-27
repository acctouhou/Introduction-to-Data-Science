import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


data = pd.read_csv('autos.csv')
acc=data["horsepower"]/data["curb.weight"]
plt.style.use('ggplot')
data['acc']=acc
aa=data.corr()
data.plot.scatter(x='engine.size',y='price')
x_t=np.array(data.iloc[:,14])[:, np.newaxis]
y_t=np.array(data.iloc[:,23])[:, np.newaxis]
from sklearn.cross_validation import train_test_split
x,x_,y,y_ =train_test_split(x_t,y_t,test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeRegressor

test_loss=[]
train_loss=[]
for i in range(15):
    reg = DecisionTreeRegressor(max_depth=i+1,random_state=42)
    reg.fit(x, y)
    y_pr=reg.predict(x_)
    y_tr=reg.predict(x)
    
    test_loss.append(mean_squared_error(y_,y_pr))
    train_loss.append(mean_squared_error(y,y_tr))
plt.clf()


plt.plot(np.arange(1,i+2,1),test_loss,'r',label='test')
plt.plot(np.arange(1,i+2,1),train_loss,'b',label='train')

plt.legend(prop={'size': 20})
plt.xlabel('tree_depth',fontsize=20)
plt.ylabel('mean_squared_error',fontsize=20)
plt.title('train test error',fontsize=26)
plt.savefig('tree_mse.png')
plt.clf()
reg = DecisionTreeRegressor(max_depth=7,random_state=42)
reg.fit(x, y)


x_test =np.arange(x.min(),x.max(),1)[:, np.newaxis]
y_pr = reg.predict(x_test)
plt.scatter(x_test,y_pr,c='r',s=15)
plt.scatter(x,y,c='b',s=10)
plt.xlabel('engine.size',fontsize=15)
plt.ylabel('price',fontsize=15)
plt.title('Tree Regression',fontsize=20)
plt.savefig('tree_scatter.png')
plt.clf()
#####################################################################
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x_t,y_t)
y_pr = lin.predict(x_t)
loss=mean_squared_error(y_t,y_pr)
x_test =np.arange(x_t.min(),x_t.max(),1)[:, np.newaxis]
y_pr = lin.predict(x_test)
plt.scatter(x_t,y_t,c='b',label='raw data')
plt.scatter(x_test,y_pr,c='r',label='linear')
plt.legend(prop={'size': 20})
plt.xlabel('engine.size',fontsize=15)
plt.ylabel('price',fontsize=15)
plt.title('linear Regression',fontsize=20)
plt.savefig('linear.png')
plt.clf()
#####################################
from sklearn.svm import SVR
import seaborn as sns

sv_test=[]
sv_train=[]

svr = SVR(kernel='rbf', gamma=1e-5,C=1e6)

svr.fit(x,y)

y_pr=svr.predict(x_test)
plt.scatter(x_test,y_pr,c='r',label='raw data')
plt.scatter(x,y,c='b',label='svm')
plt.xlabel('engine.size',fontsize=15)
plt.ylabel('price',fontsize=15)
plt.legend(prop={'size': 16})
plt.title('svm Regression',fontsize=20)
plt.savefig('svr.png')
plt.clf()
tree_tr=min(train_loss)
tree_te=min(test_loss)
linear_tr=loss
svr_tr=mean_squared_error(y_1,y)
svr_te=mean_squared_error(y_2,y_)
f, ax = plt.subplots(nrows=1, ncols=2,figsize = (12,6))
ax[0].bar(['linear','svm','tree'],[linear_tr,svr_tr,tree_tr],color=['r','g','b'])
ax[1].bar(['linear','svm','tree'],[linear_tr,svr_te,tree_te],color=['r','g','b'])
ax[0].set_title('train_mse',fontsize=20)
ax[1].set_title('test_mse',fontsize=20)
ax[0].set_ylabel('mse',fontsize=20)
ax[1].set_ylabel('mse',fontsize=20)
ax[0].set_xlabel('type',fontsize=20)
ax[1].set_xlabel('type',fontsize=20)
plt.savefig('compare.png')
plt.clf()
'''
for i in range (-10,10):
    for j in range(-10,10):
        svr = SVR(kernel='rbf', gamma=10**i,C=10**j)
        svr.fit(x,y)
        y_1=svr.predict(x)
        y_2=svr.predict(x_)
        sv_test.append(mean_squared_error(y_2,y_))
        sv_train.append(mean_squared_error(y_1,y))
        
y_pr=svr.predict(x_test)
plt.scatter(x_test,y_pr,c='r')
plt.scatter(x,y,c='b')
sv_test=np.reshape(np.array(sv_test),[20,20])
sv_train=np.reshape(np.array(sv_train),[20,20])
sv_total=sv_test+sv_train
f, ax = plt.subplots(nrows=1, ncols=3,figsize = (12,6))
sns.heatmap(sv_test, annot=False, ax=ax[0])
sns.heatmap(sv_train, annot=False, ax=ax[1])
sns.heatmap(sv_total, annot=False, ax=ax[2])
f.suptitle('Hyperparameter table',fontsize=28)

ax[1].set_xlabel('gamma(log)',fontsize=20)
ax[0].set_title('test_mse',fontsize=20)
label=list(range (-10,10,2))
label2=list(range (-10,10))

ax[0].set_xticklabels(label)
ax[0].set_yticklabels(label2)
ax[1].set_xticklabels(label)
ax[1].set_yticklabels(label2)
ax[2].set_xticklabels(label)
ax[2].set_yticklabels(label2)
ax[0].set_ylabel('penalty(log)',fontsize=20)
ax[1].set_title('train_mse',fontsize=20)
ax[2].set_title('total_mse',fontsize=20)
plt.savefig('param.png')
plt.clf()
'''
##############################
'''
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor()
'''
