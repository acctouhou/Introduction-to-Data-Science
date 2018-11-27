# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
a=0
data=np.loadtxt(open("data","rb"),delimiter=",",skiprows=0)


from sklearn import cluster
kmeans_fit = cluster.KMeans(n_clusters = 2).fit(data)
lab = kmeans_fit.labels_

x=data[lab==1,a]
x1=data[lab==0,a]
m1=x.mean()
m0=x1.mean()
data_o=data[:,a]
data_m=data[:,a]

size=len(data)
test_count=random.sample(range(size),int(size*0.2))

y=data[test_count,a]
yl=lab[test_count]

aa=np.zeros([len(y)])
data_m=np.delete(data_m,test_count)

for i in range(len(aa)):
    if yl[i]==0:
        aa[i]=m0
    elif yl[i]==1:
        aa[i]=m1
data_m=np.hstack([aa,data_m])
    

    

plt.clf()
plt.hist(y,alpha=0.5,color='red')
plt.hist(aa,alpha=0.5,color='green')
plt.savefig('1.png')
plt.clf()




plt.hist(data_m,alpha=0.5,color='red')
plt.hist(data_o,alpha=0.5,color='blue')
plt.savefig('2.png')
error=np.abs(data_m-data_o)
#aa=cluster_labels==train_y
