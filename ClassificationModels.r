## Exploratory Data Analysis (EDA)

charity <- read.csv('/Users/z013nx1/Documents/charity.csv') # load the "charity.csv" file
head(charity)

data.train <- charity[charity$part=="train",]
data.valid <- charity[charity$part=="valid",]
data.test <- charity[charity$part=="test",]
length(data.train[,1])
length(data.valid[,1])
length(data.test[,1])

data.training<- data.train[, 2:24] #remove ID 

### Feature Transformation before modeling

wrat_cat<- as.numeric(ifelse(charity$wrat <= 1, 0, 1))
charity2<- cbind(charity, wrat_cat)
plot(x = charity2$wrat_cat, charity2$wrat)

charity.f<- charity2
charity.f$incm <- log(charity2$incm)
charity.f$inca <- log(charity2$inca)
charity.f$avhv <- log(charity2$avhv)
charity.f$tgif <- log(charity2$tgif)
charity.f$lgif <- log(charity2$lgif)
charity.f$rgif <- log(charity2$rgif)
charity.f$agif <- log(charity2$agif)

#Now that we have our features transformed, we can create our final training, validation, and testing datasets 
#to be used for modeling.  Through this process we will seperate our predictor variables from our 
#response variable.  Additionally, we will scale the features so that they are providing an equal weight to the model predictions. 


data.train <- charity.f[charity.f$part=="train",]
x.train <- data.train[,c(2:21,25)]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.f[charity.f$part=="valid",]
x.valid <- data.valid[,c(2:21,25)]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.f[charity.f$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,c(2:21, 25)]


x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
#apply(x.train.std, 2, mean) # check zero mean
#apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

#Linear Discriminant Analysis

library(MASS)

#learning an LDA model with leave one out cross validation
model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + wrat_cat, 
                  data.train.std.c, cv = TRUE) 

#creating probability esimates for the validation dataset
post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

#install.packages("ROCR")
library(ROCR)
pred <- prediction(post.valid.lda1,data.valid.std.c$donr)
perf <- performance(pred,"tpr","fpr")
plot(perf, main ="ROC Curve for LDA")

## Quadractic Discriminant Analysis (QDA)

model.qda1 <- qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + wrat_cat, 
data.train.std.c, cv = TRUE) 

#creating probability esimates for the validation dataset
post.valid.qda1 <- predict(model.qda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

pred <- prediction(post.valid.qda1,data.valid.std.c$donr)
perf <- performance(pred,"tpr","fpr")
plot(perf, main ="ROC Curve for QDA")

  
## Logistic Regression

model.lr <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + plow + npro + tgif +
lgif + rgif + tdon + tlag + agif + wrat_cat, data = data.train.std.c, family = binomial) 

#creating probability esimates for the validation dataset
post.valid.lr=predict(model.lr,data.valid.std.c,type="response")

pred <- prediction(post.valid.lr,data.valid.std.c$donr)
perf <- performance(pred,"tpr","fpr")
plot(perf, main ="ROC Curve for Logistic Regression")

### Logistic Regression | L2 Regularization

set.seed (1)
library(glmnet)
#glmnet needs a model matrix
x<- model.matrix(donr~ ., data.train.std.c)[,-1]

#10-fold cross validated grid search for lambda
#alpha = 0 for L2/Ridge penalty
cv.out=cv.glmnet(x,data.train.std.c$donr, type.measure ="deviance", alpha = 0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

valid_x<- model.matrix(donr~ ., data.valid.std.c)[,-1]
ridge.lr<- glmnet(x,data.train.std.c$donr, family = "binomial", alpha = 0, lambda = 0.0291278)
pred.ridge.lr=predict(ridge.lr, type = "response", newx=valid_x)

predl2 <- prediction(pred.ridge.lr,data.valid.std.c$donr)
perfl2 <- performance(predl2,"tpr","fpr")
plot(perfl2, main ="ROC Curve for Logistic Regression | L2")
ridge_est<- ridge.lr$beta
lr_est<- model.lr$coefficients
ridge_est
lr_est

### Logistic Regression | L1 Regularization

set.seed (1)
#10-fold cross validated grid search for lambda
#alpha = 1 for L1/Lasso penalty
cv.out=cv.glmnet(x,data.train.std.c$donr, type.measure ="deviance", alpha = 1)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

lasso.lr<- glmnet(x,data.train.std.c$donr, family = "binomial", alpha = 1, lambda = 0.001591041)
pred.lasso.lr=predict(lasso.lr, type = "response", newx=valid_x)

predl1 <- prediction(pred.lasso.lr,data.valid.std.c$donr)
perfl1 <- performance(predl1,"tpr","fpr")
plot(perfl1, main ="ROC Curve for Logistic Regression | L1")

lasso_est<- lasso.lr$beta
lasso_est

### k-Nearest Neighbors Classification (KNN)

library(class)
data.train.std.knn <- subset(data.train.std.c, select = -c(donr))
data.valid.std.knn <- subset(data.valid.std.c, select = -c(donr))

knn.pred1=knn(data.train.std.knn,data.valid.std.knn,data.train.std.c$donr,k=1, prob = TRUE)
table(knn.pred1,data.valid.std.c$donr)

knn.pred3=knn(data.train.std.knn,data.valid.std.knn,data.train.std.c$donr,k=3, prob = TRUE)
table(knn.pred3,data.valid.std.c$donr)

knn.pred5=knn(data.train.std.knn,data.valid.std.knn,data.train.std.c$donr,k=5, prob = TRUE)
table(knn.pred5,data.valid.std.c$donr)

knn.pred10=knn(data.train.std.knn,data.valid.std.knn,data.train.std.c$donr,k=10, prob = TRUE)
table(knn.pred10,data.valid.std.c$donr)

knn.pred50=knn(data.train.std.knn,data.valid.std.knn,data.train.std.c$donr,k=50, prob = TRUE)
table(knn.pred50,data.valid.std.c$donr)

knn.pred100=knn(data.train.std.knn,data.valid.std.knn,data.train.std.c$donr,k=100, prob = TRUE)
table(knn.pred100,data.valid.std.c$donr)
## Random Forest Classification

library(randomForest)
#random forest with m=5
rf=randomForest(data.train.std.knn, as.factor(data.train.std.c$donr), mtry=5,prox = TRUE, importance = TRUE)
pred.rf<- predict(rf, data.valid.std.knn, type = "prob")

predrf <- prediction(pred.rf[,2],data.valid.std.c$donr)
perfrf <- performance(predrf,"tpr","fpr")
plot(perfrf, main ="ROC Curve for Random Forest")

plot(perfl2, main = "ROC Curve for Logistic Regression | L2 and Random Forest", col = 'red')
plot(perfrf, add = TRUE)

#plotting feature importance
varImpPlot (rf)

### Gradient Boosted Trees (GBM)

install.packages("gbm")
library(gbm)
set.seed (1)
boost=gbm.fit(data.train.std.knn, data.train.std.c$donr,distribution="bernoulli",n.trees=1000, interaction.depth=20)
pred.boost=predict(boost,newdata=data.valid.std.knn, n.trees=100)

predboost <- prediction(pred.boost,data.valid.std.c$donr)
perfboost <- performance(predboost,"tpr","fpr")
plot(perfboost, main ="ROC Curve for Gradient Boosted Trees")

plot(perfboost, main = "ROC Curve for Logistic Regression | L2, Random Forest, and GBM", col = 'red')
plot(perfrf, add = TRUE, col = 'green')
plot(perfl2, add = TRUE)

#plotting feature importance
summary(boost)

### Support Vector Machine | Linear

library(ROCR)
library(e1071)
dat=data.frame(x=data.train.std.knn, y=as.factor(data.train.std.c$donr))

tune.out=tune(svm,y~.,data=dat,kernel="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(tune.out)

#fit using best model from tune
bestmod=tune.out$best.model
summary(bestmod)

testdat=data.frame(x=data.valid.std.knn, y=as.factor(data.valid.std.c$donr))
svmfit=svm(y~., data=dat, kernel="linear", cost=.01, gamma = 0.04761905, scale=FALSE, decision.values = T, probability = TRUE)

pred.svm<- predict(svmfit, testdat, decision.values = TRUE, probability = TRUE)
prob.donr<- attr(pred.svm, "probabilities")[,"1"]

roc.pred<- prediction(prob.donr, data.valid.std.c$donr == "1")
perfsvm <- performance(roc.pred,"tpr","fpr")
plot(perfsvm, main ="ROC Curve for Linear SVM")

fitted=attributes(predict(svmfit,testdat,
                          decision.values=TRUE))$decision.values

### Support Vector Machine | Radial Basis Function (RBF)

tune.out.rbf=tune(svm,y~.,data=dat,kernel="radial",
ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(tune.out.rbf)

#fit using best model from tune
bestmod.rbf=tune.out.rbf$best.model
summary(bestmod.rbf)

svmfit.rbf=svm(y~., data=dat, kernel="radial", cost=5, gamma = 0.04761905, scale=FALSE, decision.values = T, probability = TRUE)

pred.svm.rbf<- predict(svmfit.rbf, testdat, decision.values = TRUE, probability = TRUE)
prob.donr.rbf<- attr(pred.svm.rbf, "probabilities")[,"1"]

roc.pred.rbf<- prediction(prob.donr.rbf, data.valid.std.c$donr == "1")
perfsvm.rbf <- performance(roc.pred.rbf,"tpr","fpr")
plot(perfsvm.rbf, main ="ROC Curve for SVM with RBF")

plot(perfsvm.rbf, main ="ROC Curve for SVM with RBF and Random Forest", col = "red")
plot(perfrf, add = TRUE)