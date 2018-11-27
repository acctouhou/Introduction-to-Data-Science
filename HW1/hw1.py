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
'''
X_tsne = TSNE(n_components=2,learning_rate=20).fit_transform(x)
X_iso = Isomap(n_neighbors=50).fit_transform(x)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.savefig('sne.png')
plt.clf()
plt.scatter(X_iso[:, 0], X_iso[:, 1], c=y)
plt.savefig('iso.png')
plt.clf()
'''


for i in range (5):
    sns.violinplot(x[:,i])
    plt.title('violin_%s'%(name[i]))
    plt.savefig('%s.png'%(name[i]))
    plt.clf()
    
b=np.corrcoef(x,rowvar=0)
f, a = plt.subplots(figsize = (6,4))
sns.heatmap(b, annot=True, vmax=1, vmin=-1, fmt='.2f', ax=a)
plt.title('correlation coefficient')
#plt.ylabel("correct")
plt.savefig('相關係數.png')


