import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
#from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import  ensemble, preprocessing, metrics
#data = pd.read_csv('winequality-white.csv.csv')
data=pd.read_csv('0411509_HW7.csv',delimiter=',')
plt.style.use('ggplot')

aa=data.corr()
x_t=np.array(data.iloc[:,0:12])
y_t=np.array(data.iloc[:,12])
for i in range(len(y_t)):
    if y_t[i]<0:
        y_t[i]=1
    else:
        y_t[i]=0

class1=y_t==1
class0=y_t==0

from sklearn.model_selection import train_test_split

x,x_,y,y_ =train_test_split(x_t,y_t,test_size=0.2, random_state=0)


from sklearn import ensemble, preprocessing, metrics
forest = ensemble.RandomForestClassifier(n_estimators = 150,oob_score=True)
forest_fit = forest.fit(x,y)
#print (forest.oob_score_)
y_predprob = forest.predict_proba(x_t)[:,1]
print(metrics.roc_auc_score(y_t,y_predprob))

ypp=forest.predict(x_t)
from sklearn.metrics import roc_curve, auc  
fpr,tpr,threshold = roc_curve(y_t, ypp)
roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr,color='darkorange',lw=2,label='ROC curve (area = %0.2f)' % (roc_auc))
plt.legend(loc="lower right",fontsize=15)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('ROC ccurve',fontsize=20)
plt.savefig('roc.png')
plt.clf()
#estimator = forest.estimators_[5]
test=[]
train=[]


for i in range(10,400,50):
    forest = ensemble.RandomForestClassifier(n_estimators =i,
                                             min_samples_split=100,min_samples_leaf=20,
                                             max_depth=8,max_features='sqrt' ,random_state=10)
    forest.fit(x,y)
    y_pred=forest.predict(x_)
    yy=forest.predict(x)
    test.append(accuracy_score(y_, y_pred))
    train.append(accuracy_score(y,yy))
    
plt.plot(list(range(10,400,50)),test,label='test')
plt.plot(list(range(10,400,50)),train,label='train')
plt.legend()
plt.ylabel('accuracy',fontsize=15)
plt.xlabel('n_estimators',fontsize=15)
plt.title('num of estimators',fontsize=20)
plt.savefig('n_estimators.png')
plt.clf()





test_y_predicted = forest.predict(x_t)

from sklearn.metrics import classification_report
print(classification_report(y_t, test_y_predicted))
from sklearn.metrics import confusion_matrix
aa=confusion_matrix(y_t, test_y_predicted)
aa=aa / aa.astype(np.float).sum(axis=1)
import seaborn as sn
df_cm = pd.DataFrame(aa,["unstable","stable"],["unstable","stable"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
plt.title('normalized confusion_matrix',fontsize=18)
plt.savefig('confuse.png')


#forest.fit(x_t,y_t)
from sklearn.tree import export_graphviz
export_graphviz(forest.estimators_[0], out_file='tree.dot', 
                rounded = True, proportion = False, 
                 filled = True)
from subprocess import call
call(['dot', '-Tpng','tree.dot', '-o', 'tree.png', '-Gdpi=600'])
#fpr, tpr, thresholds = metrics.roc_curve(y, test_y_predicted)
#auc = metrics.auc(fpr, tpr)
#print(auc)

'''KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x, y) 
test_y_predicted = neigh.predict(x_t)
from sklearn.metrics import classification_report
print(classification_report(y_t, test_y_predicted))
'''

plt.clf()
name=data.columns.values.tolist()[:12]
aa=forest.feature_importances_
plt.bar(list(range(len(aa))),aa)
plt.xticks(list(range(len(aa))),name,rotation=90)


plt.title('importances feature',fontsize=18)
plt.ylabel('importances',fontsize=15)
plt.xlabel('parameter',fontsize=15)
plt.savefig('import.png')
plt.clf()
np.sort
