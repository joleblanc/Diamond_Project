############################################################
# Capstone Project: Diamonds
# February 10, 2019
# JOhannes Le Blanc
############################################################
library(lattice)
library(munsell)
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(mlbench)
library(rpart)

############################################################
## analysis
############################################################

# load missing packages
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(munsell)) install.packages("munsell", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

# load dataset
data("diamonds")
head(diamonds)
str(diamonds)

############################################################
## clean the dataset
############################################################

# delete unnessary variables from set
diamonds2 <- diamonds[c(1,2,3,4,7)]

# sample 10000 rows
dataset <- diamonds2[sample(nrow(diamonds), 10000), ]

# convert price into categorical variable
dataset$price_cat <- cut(dataset$price, 
                         breaks = c(0, 3000, 6000, 9000, 12000, 15000, 18000, 21000), 
                         labels = c("A", "B", "C", "D", "E", "F", "G"))

# view dataset
head(dataset)
summary(dataset)
dim(dataset)

############################################################
## create train and test set
############################################################

# split dataset 
test_index <- createDataPartition(dataset$price_cat, p = 0.90, list = FALSE)
# test set gets 10% of the data
test_set <- dataset[-test_index,]
# train set gets 90% of the data
train_set <- dataset[test_index,]

# dimensions of dataset
dim(train_set)
dim(test_set)

############################################################
## data exploration
############################################################

# percentage of stones by price bucket
price_percent <- prop.table(table(train_set$price_cat)) * 100
cbind(freq = table(train_set$price_cat), price_percent = price_percent)

# show distribution of price cat
plot(train_set$price_cat)

############################################################
## visually inspect the data
############################################################

# split input and output
x <- train_set[,1:5]
y <- train_set[,6]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(train_set)[i])
}
# show distribution of price and carat as continuous variables
plot_pcc_c <- dataset %>% ggplot(aes(y = price, x = carat)) +
  geom_point(aes(color = color), alpha = 1/1)  +
  theme(legend.position="bottom") 
plot_pcc_c

# show distribution of price and carat
plot_pcc_cat <- train_set %>% ggplot(aes(y = price_cat, x = carat)) + 
  geom_point(aes(color = color), position="jitter") +
  theme(legend.position="bottom") 
plot_pcc_cat

# create a regression tree for price
fit_tree <- rpart(price_cat ~ ., data = train_set)
plot(fit_tree, margin = 0.1)
text(fit_tree, cex = 0.75)

############################################################
## creating the model
############################################################

# k = 10 cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

## use six different methods on the training set
# LDA
set.seed(7)
fit.lda <- train(price_cat~., data = train_set, method = "lda", metric = metric, trControl = control)
# Random Forest
set.seed(7)
fit.rf <- train(price_cat~., data = train_set, method = "rf", metric = metric, trControl = control)
# GBM
set.seed(7)
fit.gbm <- train(price_cat~., data = train_set, method = "gbm", metric = metric, trControl = control)
# kNN
set.seed(7)
fit.knn <- train(price_cat~., data = train_set, method = "knn", metric = metric, trControl = control)
# SVM
set.seed(7)
fit.svm <- train(price_cat~., data = train_set, method="svmRadial", metric = metric, trControl = control)

# compare models by accuracy
results <- resamples(list(lda = fit.lda, GBM = fit.gbm, knn = fit.knn, svm = fit.svm, rf = fit.rf))
summary(results) 

# compare accuracy of models
bwplot(results)

# summarize Random Forest
print(fit.rf)

############################################################
## results
############################################################

# train set accuracy
pred.train <- predict(fit.rf, train_set)
print(confusionMatrix(pred.train, train_set$price_cat))

# test set accuracy
pred.train <- predict(fit.rf, test_set)
print(confusionMatrix(pred.train, test_set$price_cat))