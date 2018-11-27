import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

x=np.loadtxt(open("data","rb"),delimiter=",",skiprows=0)
#x=data[:,0:4]
#y=data[:,4]
name=['variance','skewness','curtosis','entropy','class']
def iqr(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)
iqr_ls=[]
for i in range(5):
    iqr_ls.append(iqr(x[:,i]))
qqq=np.asarray(iqr_ls)
plt.style.use('ggplot')
plt.bar(range(len(qqq)),qqq,tick_label=name)
plt.title('IQR_list')
plt.rc('font', size=20)
plt.savefig('iqr.png')
plt.clf()

f,a=plt.subplots(nrows=5,ncols=2,figsize=(8,12),dpi=100)
for i in range(5):
    a[i][0].scatter(x[:,i],range(len(x)))
    a[i][0].set_title('%s'%(name[i]))
    a[i][1].scatter(np.sort(x[:,i]),range(len(x)))
    a[i][1].set_title('%s_sort'%(name[i]))
    a[i][0].set_xticks([])
    a[i][1].set_xticks([])
    a[i][0].set_yticks([])
    a[i][1].set_yticks([])
plt.savefig('sort.png')

plt.clf()
b=np.corrcoef(x,rowvar=0)
f, a = plt.subplots(figsize = (12,9))
sns.heatmap(b, annot=True, vmax=1, vmin=-1, fmt='.2f', ax=a,xticklabels=name,yticklabels=name)
plt.title('correlation coefficient', size=32)


#plt.ylabel("correct")
plt.savefig('相關係數.png')
plt.clf()

f,a = plt.subplots(figsize=(10,10))
a.boxplot(x[:,0:4],labels=name[0:4])
plt.title('box plot')
plt.savefig('boxplot.png')
plt.clf()

f, a = plt.subplots(4,1, figsize=(7,9), sharex=True)
for i in range(4):
    sns.distplot(x[:,i],kde=True,norm_hist=False,ax=a[i])
    a[i].set_title('%s'%(name[i]))
    #a[i].set_ticks([])
plt.savefig('hist.png')


