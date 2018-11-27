setwd("C:/Users/Acc/Desktop/數據科學概論/HW2")
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

first.pca <- top2.pca.eigenvector[, 1]   #  第一主成份
second.pca <- top2.pca.eigenvector[, 2]  #  第二主成份


first.pca[order(first.pca, decreasing=FALSE)]  
a=dotchart(first.pca[order(first.pca, decreasing=FALSE)] ,   # 排序後的係數
         main="index_1 of lpading",                      # 主標題
         xlab="Variable Loadings",                         # x軸的標題
         col="blue")

second.pca[order(second.pca, decreasing=FALSE)]  
dotchart(second.pca[order(second.pca, decreasing=FALSE)] ,  # 排序後的係數
         main="index_2 of lpading",                       # 主標題
         xlab="Variable Loadings",                          # x軸的標題
         col="blue") 
biplot(pca, choices=1:2)  
