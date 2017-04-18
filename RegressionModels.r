
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

### Linear Regression

#Fit an OLS regression model
model.lm <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                 avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif + wrat, 
               data.train.std.y)

pred.valid.lm <- predict(model.lm, newdata = data.valid.std.y) # validation predictions

mean((y.valid - pred.valid.lm)^2) # mean prediction error

sd((y.valid - pred.valid.lm)^2)/sqrt(n.valid.y) # std error

layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(model.lm)

### Linear Regression | Best Subset Model

install.packages("leaps")
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

model.bss<- lm(damt ~ lgif + reg4 + chld + hinc + reg3 + rgif + agif + tgif, data.train.std.y)
pred.valid.bss <- predict(model.bss, newdata = data.valid.std.y) # validation predictions

mean((y.valid - pred.valid.bss)^2) # mean prediction error

sd((y.valid - pred.valid.bss)^2)/sqrt(n.valid.y) # std error

layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(model.bss)

### Linear Regression | L2 Regularization

set.seed (1)
library(glmnet)
#glmnet needs a model matrix
x_reg<- model.matrix(damt~ ., data.train.std.y)[,-1]

#10-fold cross validated grid search for lambda
#alpha = 0 for L2/Ridge penalty
cv.out=cv.glmnet(x_reg,data.train.std.y$damt, type.measure ="mse", alpha = 0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

valid_x_reg<- model.matrix(damt~ ., data.valid.std.y)[,-1]
ridge.reg<- glmnet(x_reg,data.train.std.y$damt, family = "gaussian", alpha = 0, lambda = 0.1252296)
pred.ridge.reg=predict(ridge.reg, newx=valid_x_reg)

mean((y.valid - pred.ridge.reg)^2) # mean prediction error
sd((y.valid - pred.ridge.reg)^2)/sqrt(n.valid.y) # std error

### Linear Regression | L1 Regularization

set.seed (1)
#10-fold cross validated grid search for lambda
#alpha = 1 for L1/Lasso penalty
cv.out=cv.glmnet(x_reg,data.train.std.y$damt, type.measure ="mse", alpha = 1)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

lasso.reg<- glmnet(x_reg,data.train.std.y$damt, family = "gaussian", alpha = 0, lambda = 0.002697991)
pred.lasso.reg=predict(lasso.reg, newx=valid_x_reg)

mean((y.valid - pred.lasso.reg)^2) # mean prediction error
sd((y.valid - pred.lasso.reg)^2)/sqrt(n.valid.y) # std error

### Principal Components Regression

install.packages("pls")
library(pls)
set.seed (1)

#principal components regression
model.pcr=pcr(damt~., data=data.train.std.y ,scale=TRUE,
              validation ="CV")

summary(model.pcr)

validationplot(model.pcr,val.type="MSEP")
MSEP(model.pcr)

#test MSE at minimum of prior plot
pcr.pred=predict(model.pcr,data.valid.std.y,ncomp=7)
mean((y.valid - pcr.pred)^2) # mean prediction error
sd((y.valid - pcr.pred)^2)/sqrt(n.valid.y) # std error

### Random Forest

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

#plotting feature importance
varImpPlot (model.rf)

### Gradient Boosted Tree Regression

tree_count_gbm<- c(10, 100, 1000, 2000, 5000)

for (i in tree_count_gbm) {
model.gbm=gbm.fit(data.train.std.rf, data.train.std.y$damt, distribution = "gaussian", interaction.depth = 20, n.trees = i, 
verbose = FALSE)
pred.gbm<- predict(model.gbm, data.valid.std.rf, n.trees = i)  
print(i)
print(mean((y.valid - pred.gbm)^2)) # mean prediction error
print(sd((y.valid - pred.gbm)^2)/sqrt(n.valid.y)) # std error
}

summary(model.gbm)