setwd("C:/Users/Acc/Desktop/數據科學概論/HW6")
getwd()
data = read.csv("0411509_data", header = T) #read csv file

library(ggplot2)



x= data.frame((data[1:5]))
names(x)=c("variance", "skewness","curtosis","entropy","class")

y= data.frame((data[5]))
names(y)=c("class")

library(car) 


