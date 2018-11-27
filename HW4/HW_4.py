# -*- coding:utf8 -*-

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
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

'''
l_data = np.loadtxt(open("n_l_data5","rb"),delimiter=" ",skiprows=0)
mean = np.loadtxt(open("n_y_mean5","rb"),delimiter=" ",skiprows=0)
scale = np.loadtxt(open("n_y_scale5","rb"),delimiter=" ",skiprows=0)
var = np.loadtxt(open("n_y_var5","rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("n_y_data5","rb"),delimiter=" ",skiprows=0)/scale*var+mean
'''

def s_classification(s):
    ans=np.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0.266:
            ans[i]=4
        elif s[i]<=1 and s[i]>0.266:
            ans[i]=3
        elif s[i]<=3 and s[i]>1:
            ans[i]=2
        elif s[i]>3:
            ans[i]=1
    return ans
s_class=s_classification(y_data[:,11])
s1_sum=[]
s2_sum=[]
mtype=l_data[:,1]
for i in range(4):
    temp=mtype==1
    s1_sum.append(sum(s_class[temp]==(i+1)))
    temp=mtype==2
    s2_sum.append(sum(s_class[temp]==(i+1)))
    
plt.style.use('ggplot')
label=['fibrous tissue','cartilage',' immature bone',' mature bone']
colors =['#FF00FF','#9400D3','#00FF00','#008000']

separated = (.1,0,0,0)
fig,axes=plt.subplots(nrows=1,ncols=2,dpi=100,figsize=(12,4))
axes[0].pie(s1_sum , labels = label,autopct='%1.1f%%',colors=colors,explode=separated,textprops={'fontsize': 14})
axes[0].axis('equal')
axes[0].set_title('optimization',fontsize=28,verticalalignment='baseline',y=-0.06)
axes[1].pie(s2_sum, labels = label,autopct='%1.1f%%',colors=colors,explode=separated,textprops={'fontsize': 14})
axes[1].axis('equal')
axes[1].set_title('market',fontsize=28,verticalalignment='baseline',y=-0.06)
fig.suptitle('compare of model', fontsize=32)
plt.savefig('pie.png')

plt.clf()
day1=np.logical_and(s_class==4,mtype==1)
day2=np.logical_and(s_class==4,mtype==2)
plt.style.use('ggplot')

plt.hist(l_data[day1,2],bins=34,alpha=0.5,label='optimization')
plt.hist(l_data[day2,2],bins=34,alpha=0.5,label='market')
plt.legend(loc='upper right', fontsize=16)
plt.xlabel('Days', fontsize=24)
plt.ylabel('mature bone nums', fontsize=24)
plt.title('Mature bone distribution ', fontsize=32)
plt.savefig('hist.png')
plt.clf()
plt.style.use('ggplot')

local=l_data[:,0]
y=np.int64(local/1000)
x=local%1000
#xx,yy=np.meshgrid(np.arange(2,96,1),np.arange(2,96,1))
fig,axes=plt.subplots(nrows=1,ncols=2
                      ,figsize=(6,10),dpi=80)
s_tar1=np.logical_and(l_data[:,2]==34,mtype==1)
s_tar2=np.logical_and(l_data[:,2]==34,mtype==2)

#colorslist = ['#FF00FF','#9400D3','#0343df','#00FF00','#006400']
#cmaps = colors.LinearSegmentedColormap.from_list('mylist',colors,N=4) 
import matplotlib.colors

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#00FF00','#00FF00','#00FF00','#008000'])

axes[0].scatter(x[s_tar1],y[s_tar1],c=s_class[s_tar1],cmap=cmap)
axes[1].scatter(x[s_tar2],y[s_tar2],c=s_class[s_tar2],cmap=cmap)
axes[0].set_yticks([])
axes[1].set_yticks([])
axes[0].set_xticks([])
axes[1].set_xticks([])
axes[1].set_title('market',fontsize=28,verticalalignment='baseline',y=-0.06)
axes[0].set_title('optimization',fontsize=28,verticalalignment='baseline',y=-0.06)
fig.suptitle('Mature bone\nlocation', fontsize=32)

plt.savefig('scatter.png')
plt.clf()

#fig,axes=plt.subplots(nrows=1,ncols=2,dpi=100,figsize=(12,4))
groups = ["optimization", "market"]
ddd=[y_data[s_tar1,11],y_data[s_tar2,11]]
dict={"groups":groups,"S factor":ddd}
df = pd.DataFrame(dict)
fig,axes=plt.subplots(1,2)

#axes[0].sns.violinplot(y_data[s_tar1,11])
sns.violinplot(y_data[s_tar1,11],ax=axes[0])
axes[0].set_title('optimization',fontsize=28,verticalalignment='baseline',y=-0.06)

sns.violinplot(y_data[s_tar2,11],ax=axes[1])
axes[1].set_title('market',fontsize=28,verticalalignment='baseline',y=-0.06)

#fig.set_xlabel('S factor')




'''   
s=y_data1[:,11]
day=l_data1[:,2]
plt.scatter(day,s)
data=np.vstack([s,day]).T
ax = sns.boxplot(x="day", y="total_bill", data=data)
'''

