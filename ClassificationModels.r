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

## Prediction Model

There are a variety of different models developed that can be used to predict a quantitative variable, and in our case that is the donation amount.  We will develop the following models for our prediction of donation amount: 
  
  * Linear Regression
* Linear Regression | Best Subset Model
* Linear Regression | L2 Regularization
* Linear Regression | L1 Regularization
* Principal Components Regression
* Random Forest Regression
* Gradient Boosted Tree Regression

As we did with the classification models, we will learn the models on our training dataset and predict values on our validation dataset.  We will use the mean squared error of the validation dataset in order to rank which models perform the best. 

### Linear Regression

Linear regression is one of the most popular and commonly used techniques to predict quantitative values.  We use a technique called ordinary least squares where we fit a linear line that produces the lowest total value of the squared value of each observation distance from our line.  This technique is reliant on a variety of assumptions and hence can be very robust when those assumptions are met, and not reliable when they are not.  

```{r }
#Fit an OLS regression model
model.lm <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                 avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + wrat, 
               data.train.std.y)

pred.valid.lm <- predict(model.lm, newdata = data.valid.std.y) # validation predictions

mean((y.valid - pred.valid.lm)^2) # mean prediction error

sd((y.valid - pred.valid.lm)^2)/sqrt(n.valid.y) # std error


```

Fitting a linear regression model on the training dataset and predicting donation amount of the validation dataset, we find a mean prediction error of 1.5564 and a standard error of our predictions is 0.161215.  The standard error of our predictions is helpful in identifying how accurate our predictions are.  For example if we over predict and under predict wildly, the mean squared error might not give us this insight if they cancel each other out.  Using the standard error helps evalaute how tight the prediction bands are.

As we mentioned earlier, linear regression is robust if the assumptions are met.  In order to meet the assumptions our residuals need to be independently and identically distributed.  So we can plot our residuals against our fitted values to see if this is the case.  

```{r }

layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(model.lm)
```

The graphic in the upper right is showing us that for the most part, the residuals are random and we are seeing no pattern.  The QQ plot confirms this, but we do see some skewness as the residuals are not all following on the dotted line.  These plots would indicate to us that using this type of regression is acceptable and as a result the coefficients and ensuing predictions will be for the most part reliable. 

However, this model is including every variable and in hopes of using a simpler model because simple tend to "travel well" and predict more types of datasets we will examine a few types of dimension reduction and regularization. 

### Linear Regression | Best Subset Model

In hopes of finding a simpler subset model that predicts as well as the full model, it's possible to create a model with every combination of variables.  The approach is called the best subset model, and it is computationally very expensive since the amount of potential models are 2**p.  In our example, we have 21 features, so there are 2,097,152 different model combinations.  We will attempt this model because our dataset is not too large.

```{r }

#install.packages("leaps")
library(leaps)
bss= regsubsets(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + wrat, 
data.train.std.y)

reg.summary<- summary(bss)
reg.summary 

par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",
type="l")

plot(reg.summary$adjr2 ,xlab="Number of Variables ",
ylab="Adjusted RSq",type="l")

plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type='l')

plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",
type='l')

```

By examining the summary, we can see that the best subset model includes only 8 of the 21 features.  The features that are included have a star in the row for that feature.  The 8 features are:

* lgif
* reg4
* chld
* hinc
* reg3
* rgif
* agif
* tgif

After these 8 features are included, we don't see any meaningful difference in adjusted r-squared, RSS, BIC, or Mallows' Cp. A strong indicator that we shouldn't include any more features, because the prediction won't get any better but it becomes more complex. Now that we have our simpler model, we can predict our validation dataset with a model that only includes these 8 features.

```{r }

model.bss<- lm(damt ~ lgif + reg4 + chld + hinc + reg3 + rgif + agif + tgif, data.train.std.y)
pred.valid.bss <- predict(model.bss, newdata = data.valid.std.y) # validation predictions

mean((y.valid - pred.valid.bss)^2) # mean prediction error

sd((y.valid - pred.valid.bss)^2)/sqrt(n.valid.y) # std error

```

The mean squared error is slightly larger than the full model, but it's acheived with significantly less features.  The standard error is essentially the same, which again is promising considering the significantly smaller feature set.  


```{r }

layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(model.bss)

```

Plotting the residuals and fitted values we see very similar behavior as we did with the full model.  For the most part the residuals are random and the predictions don't appear have a pattern based on the size of the fitted value.

While this model was simpler, and in theory a model that would accomodate additional datasets than a more complicated one, it didn't predict as well on our validation dataset and therefore will not chose this model.

### Linear Regression | L2 Regularization

As we saw during our classification example with logistic regression, the L2 regularizing term applies shrinkage to the coefficients.  This shrinkage is to create a trade-off for training set performance and overfitting. This same concept applies to linear regression, but the loss function that we are optimizing is the least squares loss instead of the deviance loss.  Again, we can adjust the tuning paramer, lambda to create more or less shrinkage.  We will again use a cross validated search to determine the optimal value for lambda.  

```{r }
set.seed (1)
#library(glmnet)
#glmnet needs a model matrix
x_reg<- model.matrix(damt~ ., data.train.std.y)[,-1]

#10-fold cross validated grid search for lambda
#alpha = 0 for L2/Ridge penalty
cv.out=cv.glmnet(x_reg,data.train.std.y$damt, type.measure ="mse", alpha = 0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam
```

The cross validated search produces the optimal lambda value of 0.1252.  This will be the amount of regularization that we will use to fit our model.

```{r }
valid_x_reg<- model.matrix(damt~ ., data.valid.std.y)[,-1]

ridge.reg<- glmnet(x_reg,data.train.std.y$damt, family = "gaussian", alpha = 0, lambda = 0.1252296)

pred.ridge.reg=predict(ridge.reg, newx=valid_x_reg)

mean((y.valid - pred.ridge.reg)^2) # mean prediction error

sd((y.valid - pred.ridge.reg)^2)/sqrt(n.valid.y) # std error

```

The shrinkage here appears to be beneficial as we have acheived the lower mean squared error yet of any of the previous models.  Now let's examine the other regularization term, L1

### Linear Regression | L1 Regularization

As we saw before, the L1 regularizing term can not only apply shrinkage but all dimension reduction as it will let the coefficients drop all the way to 0.  This is a beneficial aspect if a few of the features are explaining the majority of the variance in the response variable, but again if a lot of the features are predictive it will usually underperform the L2 reguralizing term.

```{r }

set.seed (1)

#10-fold cross validated grid search for lambda
#alpha = 1 for L1/Lasso penalty
cv.out=cv.glmnet(x_reg,data.train.std.y$damt, type.measure ="mse", alpha = 1)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

```

After a cross validate search, we find a small value for lamda (0.002697991), telling us that the model does not need a lot of regularization.  Using this tuned value for lambda, we fit the L1 model.

```{r }

lasso.reg<- glmnet(x_reg,data.train.std.y$damt, family = "gaussian", alpha = 0, lambda = 0.002697991)

pred.lasso.reg=predict(lasso.reg, newx=valid_x_reg)

mean((y.valid - pred.lasso.reg)^2) # mean prediction error

sd((y.valid - pred.lasso.reg)^2)/sqrt(n.valid.y) # std error

```

After learning the model with our optimal value for lambda on the training set and predicting donation amounts from our validation set, we find the lowest mean squared error we have produced so far.  As we discussed above, L1 outperforms L2 when a few of the features are playing a large impact in the variation of the response variable.  This lines up nicely with our best subset model, which produce a model with only 8 features.  

### Principal Components Regression

As we've discussed a few times throughout this report, there appears to be common themes within the feature set, location, income, and previous donation history.  Principal components regression is a mathematical way to attempt to estimate these components.  The principal components are the created by assessing what creates the largest possible variance when we project the observations along a particular component (Section 6.3.1., James, Witten, Hastie, Tibshirani)  The additional components are created by being uncorrelated to the prior component and explaining the most additional variance.  Once these components have been established, we can use these components are features in our regression, rather than the features themselves. 

```{r }
#install.packages("pls")
library(pls)

set.seed (1)
#principal components regression
model.pcr=pcr(damt~., data=data.train.std.y ,scale=TRUE,
              validation ="CV")

summary(model.pcr)

validationplot(model.pcr,val.type="MSEP")
MSEP(model.pcr)
```

We plot the mean squared error the of the predictions against the number of principal components.  This shows us what is the optimal value for the number of principal components.  The heuristic rule that is advised to use, is to select the number of principal components at the "elbow" of the scree plot.  In this case, we will use 7 principal components.

```{r }

#test MSE at minimum of prior plot
pcr.pred=predict(model.pcr,data.valid.std.y,ncomp=7)
mean((y.valid - pcr.pred)^2) # mean prediction error
sd((y.valid - pcr.pred)^2)/sqrt(n.valid.y) # std error

```

We see that the mean squared error is the highest that we have observed so far.  Even though the standard error is small, signalling that our predictions are accurate, the mean squared error is largest we have seen so far.

Now that we have examined a variety of parametric approaches, we will now consider two tree-based models, random forest and gradient boosted trees.

### Random Forest

Running a random forest for regression is very similar to the random forest for classification.  We still run a series of decorrelated bootstrap aggregations, but instead of trying to create the biggest decrease in node impurity for misclassification, we are trying to get the biggest decrease in mean squared error.  Essentially the same process, but with a different optimizing parameter.  We will use the same value for m (5 ~ sqrt(21)), as the amount of features we have has stayed the same. 

When we fit the random forest, we will run through a series of tree count, which varies the amount of trees to build in the random forest.  Based on the mean squared error of the validation, we will choose the tree count that produces the minimum value. 

```{r }

data.train.std.rf <- subset(data.train.std.y, select = -c(damt))
data.valid.std.rf <- subset(data.valid.std.y, select = -c(damt))

tree_count<- c(10, 50, 100, 500, 1000)

for (i in tree_count) {
  model.rf=randomForest(data.train.std.rf, data.train.std.y$damt, mtry=5,prox = TRUE, importance = TRUE, ntree = i)
  pred.rf<- predict(model.rf, data.valid.std.rf)  
  print(i)
  print(mean((y.valid - pred.rf)^2)) # mean prediction error
  print(sd((y.valid - pred.rf)^2)/sqrt(n.valid.y)) # std error
}

```

The model with 500 trees has the lowest mean squared error and second lowest standard error of predictions.  This will be the model that we chose when we evalue this model type with regards to the other models we've developed.  Unfortunately, all of these mean squared errors are higher than the linear regression models we've developed. 

As we saw with the classification tree, we can output the feature importances which is the mean decrease in node impurity average over all the trees for a specific feature.  This is insight into which features produce the best splits in the dataset.

```{r }

#plotting feature importance
varImpPlot (model.rf)

```

Again the children feature is ranked highly and it appears that the past donation features are proving predictive.

Next let's look at a gradient boosted tree regression.

### Gradient Boosted Tree Regression

Like the random forest, the regression version of the gradient boosted tree approach is exactly the same as the classification approach.  We build small sequential trees and the hope is to decrease node impurity as measured by mean squared error.  We will use the same technique we did with the random forest and fit different models with different tree sizes.  The model that produces the minimum mean squared error on the validation set will be the one we chose.


```{r }

tree_count_gbm<- c(10, 100, 1000, 2000, 5000)

for (i in tree_count_gbm) {
model.gbm=gbm.fit(data.train.std.rf, data.train.std.y$damt, distribution = "gaussian", interaction.depth = 20, n.trees = i, 
verbose = FALSE)
pred.gbm<- predict(model.gbm, data.valid.std.rf, n.trees = i)  
print(i)
print(mean((y.valid - pred.gbm)^2)) # mean prediction error
print(sd((y.valid - pred.gbm)^2)/sqrt(n.valid.y)) # std error
}


```

Since the gradient boosted model learns "slowly" we need to give it enough time to learn the relationship.  We do this by making it fit more trees.  Here we see that when we fit a large amount of trees, the mean squared error drops down precipitiously.  The 5000 tree model produces a mean squared error of only 1.438698.  

Feature importance:

```{r }

summary(model.gbm)

```

We see that past donation history again is proving predictive of future donations.  Again we notice the amount of children in the home as a highly ranked feature. 

### Regression Summary

Let's examine the mean squared prediction error and standard error of our predictions for all of the models we have fit.

Model | Mean Squared Error | Standard Error of Predictions
:-------------------------------------|:-------------------------:|:-----------------------------------:
  Gradient Boosted Tree Regression | 1.438698 | 0.1652758
Linear Regression, L1 Regularization | 1.486639 | 0.1591288
Linear Regression, L2 Regularization | 1.504044 | 0.1608033
Linear Regression | 1.556378 | 0.161215
Random Forest Regression | 1.658179 | 0.1724815
Linear Regression, Best Subset Model | 1.658895 | 0.1607532
Principal Components Regression | 1.725119 | 0.1593986

All of the models are fairly close, with the gradient boosted tree with 5000 trees performing the best.  Interestingly, the standard error for principal components regression is approximately the smallest, but the mean is the highest.  As a result, we will chose the gradient boosted tree regression for our final donation amount predictions.


### Test Set Predictions

We need to now use the optimal models for classification and regression on our test dataset.  We will fit the support vector machine with radial basis function kernel for our classification, and our 5000 tree gradient boosted regression.  Additionally, we need to adjust for the weighted oversampling in the training and validation datasets for our classification model.

```{r }

post.test <- predict(rf, data.test.std, type = "prob")[, "1"] # post probs for test data

# Oversampling adjustment for calculating number of mailings for test set
n.mail.valid <- which.max(profit.rf)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)

```

Here we would mail to 310 of the 2007 potential donors in the test set.  Given the fact that the typical response rate is 0.10, that would tell us that the optimal mailing would be ~200 (2007 * 0.10).  Our model will "waste" a bit of circulation in order to get as many responders as possible.

Now that we have created our donor predictions with our classification model, we will now create our donation amount predictions.


```{r }

# select 5000 tree GBM  since it has minimum mean prediction error in the validation sample
yhat.test <- predict(model.gbm, newdata = data.test.std, n.trees = 5000) # test predictions


```

Now that we have generated our predictions, let's run some quality checks to make sure that we not only generated them without error, but that they make qualitative sense.

```{r }

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] 
yhat.test[1:10] 

densityplot(yhat.test)
densityplot(data.train.std.y$damt, add = TRUE, col = "red")
```

We can see that we generated the correct number of predictions and the predictions of our donation amount appear to be in the range that we saw in our training dataset.

Now that we have completed the modeling, let's export the predictions as a .csv file for submission.

```{r }
ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="RST.csv", row.names=FALSE) # use your initials for the file name
```

### Bibliography

James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. An Introduction to Statistical Learning with Applications   in R. New York, NY: Springer, 2015. Print.