setwd("C:/Users/Acc/Desktop/�ƾڬ�Ƿ���/HW2")
getwd()
data = read.csv("data", header = T) #read csv file

library(ggplot2)



x= data.frame((data[1:5]))
names(x)=c("variance", "skewness","curtosis","entropy","class")

y= data.frame((data[5]))
names(y)=c("class")
pca <- prcomp(x,scale = TRUE)
str(test_data_PCA_prcomp)
#plot(pca,type="line",main="pca")
p=qplot(c(1:4),pca$sdev,geom = c("point", "path")
      ,xlab ="index",ylab="variance")

p + theme(axis.text.x = element_text(size =32,color='blue'))+ theme(axis.text.y = element_text(size =32,color='blue'))+ theme(axis.title.x = element_text(size = 36))+theme(axis.title.y = element_text(size = 36))



vars <- (pca$sdev)^2
props <- vars / sum(vars) 
cumulative.props <- cumsum(props)
aa=qplot(c(1:4),cumulative.props,geom = c("point", "smooth")
      ,xlab ="index",ylab="accumulation of variance")
aa + theme(axis.text.x = element_text(size =32,color='blue'))+ theme(axis.text.y = element_text(size =32,color='blue'))+ theme(axis.title.x = element_text(size = 32))+theme(axis.title.y = element_text(size = 32))

#plot(cumulative.props)
pca$rotation
top2.pca.eigenvector <- pca$rotation[, 1:5]
top2.pca.eigenvector

first.pca <- top2.pca.eigenvector[, 1]   #  �Ĥ@�D����
second.pca <- top2.pca.eigenvector[, 2]  #  �ĤG�D����


first.pca[order(first.pca, decreasing=FALSE)]  
a=dotchart(first.pca[order(first.pca, decreasing=FALSE)] ,   # �Ƨǫ᪺�Y��
         main="index_1 of lpading",                      # �D���D
         xlab="Variable Loadings",                         # x�b�����D
         col="blue")

second.pca[order(second.pca, decreasing=FALSE)]  
dotchart(second.pca[order(second.pca, decreasing=FALSE)] ,  # �Ƨǫ᪺�Y��
         main="index_2 of lpading",                       # �D���D
         xlab="Variable Loadings",                          # x�b�����D
         col="blue") 
biplot(pca, choices=1:2)  